"""
Implements the TransFuser vision backbone.
"""

import copy
import math

import timm
import torch
import torch.nn.functional as F
from torch import nn
from navsim.agents.uead_main.transfuser_config import TransfuserConfig
from navsim.agents.uead_main.edge_detection import *


class TransfuserBackbone(nn.Module):
    """Multi-scale Fusion Transformer for image + LiDAR feature fusion."""

    def __init__(self, config: TransfuserConfig):

        super().__init__()
        self.config = config

        self.image_encoder = timm.create_model(config.image_architecture, pretrained=False, features_only=True)
        # 手动加载本地权重，忽略分类头（fc）部分
        state_dict = torch.load(config.bkb_path, map_location='cpu')
        self.image_encoder.load_state_dict(state_dict, strict=False)
        self.edge_encoder = timm.create_model(
            config.edge_architecture,
            pretrained=False,
            features_only=True,
        )

        self.edge_encoder.conv1 = nn.Identity()
        self.edge_encoder.bn1 = nn.Identity()
        self.edge_encoder.act1 = nn.Identity()

        if self.config.fuse_type == "concat":
            concat_cov_in_channels_list = [64, 128, 256, 512]
            self.concat_cov = nn.ModuleList([
                nn.Conv2d(concat_cov_in_channels_list[i] * 2, concat_cov_in_channels_list[i], kernel_size=1)
                for i in range(4)
            ]
            )
        elif self.config.fuse_type == "attention":
            concat_cov_in_channels_list = [[32*128, 128], [16*64, 256], [8*32, 512]]
            self.fuse_modules = nn.ModuleList([
                CrossAttentionFusion(concat_cov_in_channels_list[i])
                for i in range(3)
            ]
            )
        #edge
        self.fuseplanes = [64, 128]
        self.dil = 32
        self.attentions = nn.ModuleList()
        self.dilations = nn.ModuleList()
        self.conv_reduces = nn.ModuleList()
        for i in range(2):
            self.dilations.append(CDCM(self.fuseplanes[i], self.dil))
            self.attentions.append(CSAM(self.dil))
            self.conv_reduces.append(MapReduce(self.dil))


        self.G = GeneratorResNet(512)
        self.img_fpn = FPN(in_channels=[64, 128, 256, 512])
        num_classes: int = 512
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)


    def forward(self, image):
        """
        Image + LiDAR feature fusion using transformers
        Args:
            image_list (list): list of input images
            lidar_list (list): list of input LiDAR BEV
        """
        image_features = image

        # Generate an iterator for all the layers in the network that one can loop through.
        image_layers = iter(self.image_encoder.items())
        edge_layers = iter(self.edge_encoder.items())
        edge_late_features= []
        img_feature_list = []
        # Stem layer.
        # In some architectures the stem is not a return layer, so we need to skip it.
        if len(self.image_encoder.return_layers) > 4:
            image_features = self.forward_layer_block(image_layers, self.image_encoder.return_layers, image_features)
        if len(self.edge_encoder.return_layers) > 4:
            edge_features = self.forward_layer_block(edge_layers, self.edge_encoder.return_layers, image_features)

        # Loop through the 4 blocks of the network.
        for i in range(4):
            image_features = self.forward_layer_block(image_layers, self.image_encoder.return_layers, image_features)
            edge_features = self.forward_layer_block(edge_layers, self.edge_encoder.return_layers, edge_features)

            image_features = self.fuse_features(image_features, edge_features, i, self.config.fuse_type)

            edge_late_features.append(edge_features)
            img_feature_list.append(image_features)
        img_feature_upscale = self.img_fpn(img_feature_list)
        img_feature_upscale = F.avg_pool2d(img_feature_upscale, kernel_size=(1, 4), stride=(1, 4))
        # Edge Detection
        edges = []
        H, W = 256, 1024
        for i, xi in enumerate(edge_late_features[:2]):
            attedge = self.attentions[i](self.dilations[i](xi))
            edges.append(F.interpolate(self.conv_reduces[i](attedge),
                                       (H, W), mode="bilinear", align_corners=False))

        edge_g = self.G(edge_late_features[-1])
        edges.append(F.interpolate(edge_g, (H, W), mode="bilinear", align_corners=False))

        x = self.avgpool(image_features)
        x = torch.flatten(x, 1)
        x = self.fc(x)


        return x, img_feature_upscale, edges

    def forward_layer_block(self, layers, return_layers, features):
        """
        Run one forward pass to a block of layers from a TIMM neural network and returns the result.
        Advances the whole network by just one block
        :param layers: Iterator starting at the current layer block
        :param return_layers: TIMM dictionary describing at which intermediate layers features are returned.
        :param features: Input features
        :return: Processed features
        """
        for name, module in layers:
            features = module(features)
            if name in return_layers:
                break
        return features

    def fuse_features(self, image_features, edge_features, layer_idx, fuse_type='multiply'):
        """
        Perform a TransFuser feature fusion block using a Transformer module.
        :param image_features: Features from the image branch
        :param lidar_features: Features from the LiDAR branch
        :param layer_idx: Transformer layer index.
        :return: image_features and lidar_features with added features from the other branch.
        """
        if fuse_type == 'multiply':
            image_features = image_features * edge_features
        elif fuse_type == 'add':
            image_features = image_features + edge_features
        elif fuse_type == 'concat':
            image_features = torch.cat([image_features, edge_features], dim=1)
            image_features = self.concat_cov[layer_idx](image_features)
        elif fuse_type == 'attention':
            if layer_idx == 0:
                image_features = image_features
            else:
                image_features = self.fuse_modules[layer_idx-1](image_features, edge_features)

        return image_features

class TransfuserBackbone_RGB(nn.Module):
    """Multi-scale Fusion Transformer for image + LiDAR feature fusion."""

    def __init__(self, config: TransfuserConfig):

        super().__init__()

        self.image_encoder = timm.create_model(config.image_architecture, pretrained=True, features_only=True,
                                                   pretrained_cfg_overlay=dict(file=config.bkb_path))

        self.img_fpn = FPN(in_channels=[64, 128, 256, 512])
        num_classes: int = 512
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)


    def forward(self, image):
        """
        Image + LiDAR feature fusion using transformers
        Args:
            image_list (list): list of input images
            lidar_list (list): list of input LiDAR BEV
        """
        image_features = image
        img_feature_list = self.image_encoder(image_features)

        img_feature_upscale = self.img_fpn(img_feature_list[1:])
        img_feature_upscale = F.avg_pool2d(img_feature_upscale, kernel_size=(1, 4), stride=(1, 4))

        x = self.avgpool(img_feature_list[4])
        x = torch.flatten(x, 1)
        x = self.fc(x)


        return x, img_feature_upscale


class FPN(nn.Module):
    """
    Compact Dilation Convolution based Module
    """

    def __init__(self, in_channels=[64, 128, 256, 512]):
        super(FPN, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        self.upsample1 = nn.ConvTranspose2d(in_channels[3], 256, kernel_size=2, stride=2, padding=0)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv1 = nn.Conv2d(256+in_channels[2], 256, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(256)


        self.upsample2 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2, padding=0)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(256+in_channels[1], 256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(256)

        self.upsample3 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2, padding=0)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv3 = nn.Conv2d(256+in_channels[0], 256, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(256)

    def forward(self, x):
        # p4:272,8,8;
        # p3:160,16,16;
        # p2:64,32,32;
        # p1:128,64,64
        p4 = self.relu(self.bn1(self.upsample1(x[3])))
        p3 = self.relu(self.bn2(self.conv1(torch.cat([x[2], p4], dim=1))))

        p3 = self.relu(self.bn3(self.upsample2(p3)))
        p2 = self.relu(self.bn4(self.conv2(torch.cat([x[1], p3], dim=1))))

        p2 = self.relu(self.bn5(self.upsample3(p2)))
        p1 = self.relu(self.bn6(self.conv3(torch.cat([x[0], p2], dim=1))))

        return p1

class CrossAttentionFusion(nn.Module):

    def __init__(self,  embed_config):
        super(CrossAttentionFusion, self).__init__()
        # embed_config = [num_embed, embed_dims]
        self.kv_embedding = nn.Embedding(embed_config[0], embed_config[1])
        self.query_embedding = nn.Embedding(embed_config[0], embed_config[1])

        attn_module_layer = nn.TransformerDecoderLayer(embed_config[1], 2, dim_feedforward=256, dropout=0.1,
                                                       batch_first=True)
        self.attn_module = nn.TransformerDecoder(attn_module_layer, 1)



    def forward(self, img_feature, edge_feature):

        bs, _, img_h, img_w = img_feature.shape
        img_feature = img_feature.flatten(-2, -1)  # bs, embed_dims, num_embed
        tgt = img_feature.permute(0, 2, 1).contiguous()  # bs,  num_embed, embed_dims

        edge_feature = edge_feature.flatten(-2, -1)  # bs, embed_dims, num_embed
        memory = edge_feature.permute(0, 2, 1).contiguous()  # bs,  num_embed, embed_dims

        query = tgt + self.query_embedding.weight[None, ...] #bs,  num_embed, embed_dims
        memory = memory + self.kv_embedding.weight[None, ...]

        query = self.attn_module(query, memory)  # [bs, num_embed, embed_dims]

        fuse_feature = query.view(bs, img_h, img_w, -1).permute(0, 3, 1, 2).contiguous()

        return fuse_feature

class GeneratorResNet_TCP(nn.Module):
    def __init__(self, input_channel):   ## (input_shape = (3, 256, 256), num_residual_blocks = 9)
        super(GeneratorResNet_TCP, self).__init__()

        ## 初始化网络结构
        out_features = 256                              ## 输出特征数out_features = 64
        encoder = [                                           ## model = [Pad + Conv + Norm + ReLU]              ## ReflectionPad2d(3):利用输入边界的反射来填充输入张量
            nn.Conv2d(input_channel, out_features, kernel_size=3, stride=2, padding=1),           ## Conv2d(3, 64, 7)
            nn.BatchNorm2d(out_features),                ## InstanceNorm2d(64):在图像像素上对HW做归一化，用在风格化迁移
            nn.ReLU(inplace=True),                          ## 非线性激活
        ]
        in_features = out_features                          ## in_features = 64

        ## 下采样，循环2次
        for _ in range(2):
            out_features = in_features                          ## out_features = 128 -> 256
            encoder += [                                                          ## (Conv + Norm + ReLU) * 2
                nn.Conv2d(in_features, out_features, 3, stride=1, padding=1),
                nn.BatchNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features
        self.encoder = nn.Sequential(*encoder)
        decoder = []
        in_features = out_features
        # 上采样两次
        for _ in range(4):
            out_features = int(in_features/2)

            decoder += [                                                          ## model += [Upsample + conv + norm + relu]
                nn.Upsample(scale_factor=2),
                nn.Conv2d(in_features, out_features, 3, stride=1, padding=1),
                nn.BatchNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features                                          ## out_features = 64

        ## 网络输出层                                                            ## model += [pad + conv + tanh]
        decoder += [nn.ReflectionPad2d(3), nn.Conv2d(out_features, 1, 7)]    ## 将(3)的数据每一个都映射到[-1, 1]之间

        self.decoder = nn.Sequential(*decoder)

    def forward(self, x):           ## 输入(1, 3, 256, 256)
        h = self.encoder(x)
        out = self.decoder(h)
        return out, h
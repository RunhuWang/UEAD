import torch.nn as nn
import torch
import torch.nn.functional as F
class CSAM(nn.Module):
    """
    Compact Spatial Attention Module
    """

    def __init__(self, channels):
        super(CSAM, self).__init__()

        mid_channels = 4
        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv2d(channels, mid_channels, kernel_size=1, padding=0)
        self.conv2 = nn.Conv2d(mid_channels, 1, kernel_size=3, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()
        nn.init.constant_(self.conv1.bias, 0)

    def forward(self, x):
        y = self.relu1(x)
        y = self.conv1(y)
        y = self.conv2(y)
        y = self.sigmoid(y)

        return x * y


class CDCM(nn.Module):
    """
    Compact Dilation Convolution based Module
    """

    def __init__(self, in_channels, out_channels):
        super(CDCM, self).__init__()

        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        self.conv2_1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, dilation=5, padding=5, bias=False)
        self.conv2_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, dilation=7, padding=7, bias=False)
        self.conv2_3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, dilation=9, padding=9, bias=False)
        self.conv2_4 = nn.Conv2d(out_channels, out_channels, kernel_size=3, dilation=11, padding=11, bias=False)
        nn.init.constant_(self.conv1.bias, 0)

    def forward(self, x):
        x = self.relu1(x)
        x = self.conv1(x)
        x1 = self.conv2_1(x)
        x2 = self.conv2_2(x)
        x3 = self.conv2_3(x)
        x4 = self.conv2_4(x)
        return x1 + x2 + x3 + x4

class MapReduce(nn.Module):
    """
    Reduce feature maps into a single edge map
    """
    def __init__(self, channels):
        super(MapReduce, self).__init__()
        self.conv = nn.Conv2d(channels, 1, kernel_size=1, padding=0)
        nn.init.constant_(self.conv.bias, 0)

    def forward(self, x):
        return self.conv(x)

class ConvDownsample(nn.Module):

    def __init__(self, channels, last_layer=False):
        super(ConvDownsample, self).__init__()
        self.last_layer = last_layer
        self.conv = nn.Conv2d(channels, 16, kernel_size=3, padding=1, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.LeakyReLU()
        nn.init.constant_(self.conv.bias, 0)

    def forward(self, x):
        if self.last_layer:
            return self.relu(self.bn1(self.conv(x)))
        else:
            return self.sigmoid(self.bn1(self.conv(x)))

class GeneratorResNet(nn.Module):
    def __init__(self, input_channel):   ## (input_shape = (bs, 512, 8, 32), num_residual_blocks = 9)
        super(GeneratorResNet, self).__init__()
        decoder = []
        in_features = 512
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
        out = self.decoder(x)
        return out


def cross_entropy_loss_RCF(prediction, labelf, beta):
    label = labelf.long()
    mask = labelf.clone()
    num_positive = torch.sum(label==1).float()
    num_negative = torch.sum(label==0).float()

    mask[label == 1] = 1.0 * num_negative / (num_positive + num_negative)
    mask[label == 0] = beta * num_positive / (num_positive + num_negative)
    mask[label == 2] = 0
    cost = F.binary_cross_entropy_with_logits(
            prediction, labelf, weight=mask, reduction='sum')
    return cost
if __name__ == '__main__':
    import time
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img = torch.rand((1, 512, 8, 29)).to(device)

    agent = GeneratorResNet().to(device)
    for i in range(10):
        t2 = time.time()
        outputs = agent(img)
        print(time.time()-t2)
    print(outputs.size())
    total = sum([param.nelement() for param in agent.parameters()])
    print(total/1e6)
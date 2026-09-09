# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
from typing import Optional, Dict
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from navsim.agents.uead_main.blocks import linear_relu_ln,bias_init_with_prob, GridSampleCrossBEVAttention
from navsim.common.enums import StateSE2Index
from navsim.agents.uead_main.transfuser_config import TransfuserConfig
class PlanningRefinementModule(nn.Module):
    def __init__(
            self,
            embed_dims=256,
            ego_fut_ts=8,
            ego_fut_mode=20,
            if_zeroinit_reg=True,
    ):
        super(PlanningRefinementModule, self).__init__()
        self.embed_dims = embed_dims
        self.ego_fut_ts = ego_fut_ts
        self.ego_fut_mode = ego_fut_mode
        self.plan_cls_branch = nn.Sequential(
            *linear_relu_ln(embed_dims, 1, 2),
            nn.Linear(embed_dims, 1),
        )
        self.plan_reg_branch = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, ego_fut_ts * 3),
        )
        self.if_zeroinit_reg = False

        self.init_weight()

    def init_weight(self):
        if self.if_zeroinit_reg:
            nn.init.constant_(self.plan_reg_branch[-1].weight, 0)
            nn.init.constant_(self.plan_reg_branch[-1].bias, 0)

        bias_init = bias_init_with_prob(0.01)
        nn.init.constant_(self.plan_cls_branch[-1].bias, bias_init)

    def forward(
            self,
            traj_feature,
    ):
        bs, ego_fut_mode, _ = traj_feature.shape

        # 6. get final prediction
        traj_feature = traj_feature.view(bs, ego_fut_mode, -1)
        plan_cls = self.plan_cls_branch(traj_feature).squeeze(-1)
        traj_delta = self.plan_reg_branch(traj_feature)
        plan_reg = traj_delta.reshape(bs, ego_fut_mode, self.ego_fut_ts, 3)

        return plan_reg, plan_cls

class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self,  tgt, lidar_token, traj_points,
                     lidar_pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        output = tgt

        poses_reg_list = []
        poses_cls_list = []
        for layer in self.layers:
            output, poses_reg, poses_cls = layer(output, lidar_token, traj_points,
                           lidar_pos=lidar_pos,
                           query_pos=query_pos)
            poses_reg_list.append(poses_reg)
            poses_cls_list.append(poses_cls)
            traj_points = poses_reg[..., :2].clone().detach()

        return poses_reg_list, poses_cls_list


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, config=None):
        super().__init__()
        self.cross_bev_attention = nn.MultiheadAttention(
            config.tf_d_model,
            config.tf_num_head,
            dropout=config.tf_dropout,
            batch_first=True,
        )
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self.task_decoder = PlanningRefinementModule(
            embed_dims=256,
            ego_fut_ts=8,
            ego_fut_mode=20,
        )


    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, lidar_token, traj_points,
                     lidar_pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):


        tgt2 = self.cross_bev_attention(self.with_pos_embed(tgt, query_pos), self.with_pos_embed(lidar_token, lidar_pos), lidar_token)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        poses_reg, poses_cls = self.task_decoder(tgt)  # bs,20,8,3; bs,20
        poses_reg[..., :2] = poses_reg[..., :2] + traj_points
        poses_reg[..., StateSE2Index.HEADING] = poses_reg[..., StateSE2Index.HEADING].tanh() * np.pi

        return tgt, poses_reg, poses_cls



class TrajectoryHead(nn.Module):
    """Trajectory prediction head."""

    def __init__(self, num_poses: int, d_ffn: int, d_model: int, plan_anchor_path: str, config: TransfuserConfig):
        """
        Initializes trajectory head.
        :param num_poses: number of (x,y,θ) poses to predict
        :param d_ffn: dimensionality of feed-forward network
        :param d_model: input dimensionality
        """
        super(TrajectoryHead, self).__init__()

        self._num_poses = num_poses
        self._d_model = d_model
        self._d_ffn = d_ffn
        self.diff_loss_weight = 2.0
        self.ego_fut_mode = 20

        self.query_embedding = nn.Embedding(20, config.tf_d_model)
        self.bev_embedding = nn.Embedding(4096, config.tf_d_model)
        plan_anchor = np.load(plan_anchor_path)
        self.plan_anchor = nn.Parameter(
            torch.tensor(plan_anchor, dtype=torch.float32),
            requires_grad=False,
        )  # 20,8,2
        self.anchor_branch = nn.Sequential(
            *linear_relu_ln(config.tf_d_model, 1, 2, input_dims=272),
            nn.Linear(config.tf_d_model, config.tf_d_model),
        )
        decoder_layer = TransformerDecoderLayer(d_model=256, dim_feedforward=1024, config=config)
        self.traj_fdecoder = TransformerDecoder(decoder_layer=decoder_layer, num_layers=3)


    def forward(self, bev_feature, status_encoding) -> Dict[str, torch.Tensor]:
        """Torch module forward pass."""
        bs = bev_feature.shape[0]
        # 1. add truncated noise to the plan anchor
        plan_anchor = self.plan_anchor.unsqueeze(0).repeat(bs, 1, 1, 1)
        plan_query = plan_anchor.flatten(-2, -1)  # bs, 20, 16
        query = torch.concatenate([plan_query, status_encoding[:, None].expand(-1, 20, -1)], dim=2)  # bs, 20, 272
        query = self.anchor_branch(query)  # bs, 20, 256
        query_pos= self.query_embedding.weight[None, ...].repeat(bs, 1, 1)  # bs, 1, 256

        bev_pos = self.bev_embedding.weight[None, ...].repeat(bs, 1, 1)  # bs, 1, 256
        bev_feature = bev_feature.flatten(-2, -1) #bs, 256, 4096
        bev_feature = bev_feature.permute(0, 2, 1) #bs,  4096, 256

        # 4. begin the stacked decoder
        poses_reg_list, poses_cls_list = self.traj_fdecoder(query, bev_feature,  plan_anchor, bev_pos, query_pos)
        reg_0 = poses_reg_list[0]
        reg_1 = poses_reg_list[1]
        reg_2 = poses_reg_list[2]
        cls_0 = poses_cls_list[0]
        cls_1 = poses_cls_list[1]
        cls_2 = poses_cls_list[2]

        mode_idx = poses_cls_list[-1].argmax(dim=-1)
        mode_idx = mode_idx[..., None, None, None].repeat(1, 1, self._num_poses, 3)
        best_reg = torch.gather(poses_reg_list[-1], 1, mode_idx).squeeze(1)

        if self.training:
            return {"trajectory": best_reg, "reg_0": reg_0, "reg_1": reg_1, "reg_2": reg_2, "cls_0": cls_0, "cls_1": cls_1, "cls_2": cls_2, "plan_anchor": plan_anchor}

        else:
            return {"trajectory": best_reg}


class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """
    def __init__(self, num_pos_feats=256):
        super().__init__()
        self.row_embed = nn.Embedding(64, num_pos_feats)
        self.col_embed = nn.Embedding(64, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, x):
        h, w = 64, 64
        i = torch.arange(w, device=x.device)
        j = torch.arange(h, device=x.device)
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        pos = torch.cat([
            x_emb.unsqueeze(0).repeat(h, 1, 1),
            y_emb.unsqueeze(1).repeat(1, w, 1),
        ], dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
        return pos



def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

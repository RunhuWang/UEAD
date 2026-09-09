"""Scene graph trajectory reconstruction module.

This module is designed as a lightweight add-on for trajectory reliability
assessment. It reconstructs the planning outputs from agent, BEV map, and ego
context tokens. The reconstruction residual can be used as an anomaly or
reliability score for the original planner output.
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MLP(nn.Module):
    """Two-layer MLP with LayerNorm."""

    def __init__(
        self, in_dim: int, hidden_dim: int = 256, out_dim: int = 256, dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ConvBNReLU(nn.Module):
    """Convolution block used by the BEV map token encoder."""

    def __init__(
        self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride,padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class AgentTokenEncoder(nn.Module):
    """Encodes object-query states into agent graph tokens."""

    def __init__(self, box_dim: int, d_model: int = 256) -> None:
        super().__init__()
        self.state_proj = MLP(box_dim + 1, d_model, d_model)
        self.fusion = MLP(d_model * 2, d_model, d_model)
        self.type_embed = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.type_embed, std=0.02)

    def forward(
        self,
        agent_states: torch.Tensor,
        agent_labels: torch.Tensor,
        agents_query: torch.Tensor,
    ) -> torch.Tensor:
        """Returns agent tokens with shape [B, N_agent, D]."""

        state = torch.cat([agent_states, agent_labels.unsqueeze(-1)], dim=-1)

        state_feat = self.state_proj(state)
        tokens = self.fusion(torch.cat([state_feat, agents_query], dim=-1))
        return tokens + self.type_embed


class MapPatchTokenEncoder(nn.Module):
    """Encodes BEV semantic logits and BEV features into map patch tokens.

    Expected shapes:
      bev_semantic_map:  [B, num_bev_classes, 256, 128]
      cross_bev_feature: [B, cross_bev_channels, 64, 64]

    Default output:
      map tokens: [B, 32 * 16, D]
    """

    def __init__(
        self,
        d_model: int = 256,
        num_bev_classes: int = 8,
        cross_bev_channels: int = 64,
        map_size: Tuple[int, int] = (16, 32),
    ) -> None:
        super().__init__()
        self.map_size = map_size
        num_patches = map_size[0] * map_size[1]

        self.semantic_encoder = nn.Sequential(
            ConvBNReLU(num_bev_classes, 64, stride=2),
            ConvBNReLU(64, 128, stride=2),
            ConvBNReLU(128, d_model, stride=2),
            ConvBNReLU(d_model, d_model, stride=1),
        )

        self.cross_bev_encoder = nn.Sequential(
            ConvBNReLU(cross_bev_channels, 128, stride=2),
            ConvBNReLU(128, d_model, stride=(2, 1), padding=1),
            ConvBNReLU(d_model, d_model, stride=1),
        )

        self.fusion = nn.Sequential(
            ConvBNReLU(d_model * 2, d_model, kernel_size=1, padding=0),
            ConvBNReLU(d_model, d_model, kernel_size=3, padding=1),
        )

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, d_model))
        self.type_embed = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.type_embed, std=0.02)

    def forward(
        self,
        bev_semantic_map: torch.Tensor,
        cross_bev_feature: torch.Tensor,
    ) -> torch.Tensor:
        """Returns map patch tokens with shape [B, H_map * W_map, D]."""

        sem = torch.softmax(bev_semantic_map, dim=1)

        sem_feat = self.semantic_encoder(sem)
        bev_feat = self.cross_bev_encoder(cross_bev_feature)

        if sem_feat.shape[-2:] != self.map_size:
            raise ValueError(
                "semantic_encoder produced spatial shape "
                f"{tuple(sem_feat.shape[-2:])}, expected {self.map_size}."
            )
        if bev_feat.shape[-2:] != self.map_size:
            raise ValueError(
                "cross_bev_encoder produced spatial shape "
                f"{tuple(bev_feat.shape[-2:])}, expected {self.map_size}."
            )

        fused = self.fusion(torch.cat([sem_feat, bev_feat], dim=1))
        tokens = fused.flatten(2).permute(0, 2, 1)
        return tokens + self.pos_embed + self.type_embed


class EgoTokenEncoder(nn.Module):
    """Encodes ego/status features into one ego graph token."""

    def __init__(
        self,
        status_dim: int,
        encoding_dim: int,
        d_model: int = 256,
    ) -> None:
        super().__init__()
        self.proj = MLP(status_dim + encoding_dim, d_model, d_model)
        self.type_embed = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.type_embed, std=0.02)

    def forward(
        self,
        status_feature: torch.Tensor,
        status_encoding: torch.Tensor,
    ) -> torch.Tensor:
        """Returns ego token with shape [B, 1, D]."""

        ego = torch.cat([status_feature, status_encoding], dim=-1)
        return self.proj(ego).unsqueeze(1) + self.type_embed


class SceneGraphTransformer(nn.Module):
    """Graph Transformer over trajectory, agent, map, and ego tokens."""

    def __init__(
        self,
        d_model: int = 256,
        num_layers: int = 3,
        nhead: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.encoder(tokens)


class TrajectoryReconstructionHeads(nn.Module):
    """Decodes trajectory tokens through a reconstructed trajectory feature."""

    def __init__(
        self,
        d_model: int = 256,
        num_poses: int = 8,
        feat_dim: int = 256,
    ) -> None:
        super().__init__()
        self.num_poses = num_poses

        self.feature_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, feat_dim),
            nn.LayerNorm(feat_dim),
        )

        self.reg_head = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, num_poses * 3),
        )

        self.im_score_head = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, 1),
        )

        self.sim_score_head = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, 5),
        )

    def forward(
        self,
        traj_tokens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, num_modes, _ = traj_tokens.shape

        pred_feature = self.feature_head(traj_tokens)

        pred_traj = self.reg_head(pred_feature)
        pred_traj = pred_traj.view(bsz, num_modes, self.num_poses, 3)

        pred_im_scores = self.im_score_head(pred_feature).squeeze(-1)
        pred_sim_scores = self.sim_score_head(pred_feature).permute(0, 2, 1)

        return pred_traj, pred_im_scores, pred_sim_scores, pred_feature


class SceneGraphTrajectoryReconstructor(nn.Module):
    """Multi-granularity scene graph trajectory reconstructor.

    The first ``num_modes`` graph tokens are learnable trajectory query nodes.
    They do not receive the original trajectory feature as input. The module
    reconstructs the original planner's trajectory distribution from agent,
    BEV-map, and ego context.

    Default token layout:
      20 trajectory query nodes
      30 agent nodes
      512 map patch nodes
      1 ego node
    """

    def __init__(
        self,
        box_dim: int = 5,
        status_dim: int = 8,
        encoding_dim: int = 256,
        d_model: int = 256,
        num_bev_classes: int = 8,
        cross_bev_channels: int = 64,
        map_size: Tuple[int, int] = (16, 32),
        num_modes: int = 256,
        num_poses: int = 8,
        traj_feat_dim: int = 256,
        graph_layers: int = 3,
        nhead: int = 8,
        dropout: float = 0.1,
        plan_anchor: str = None,
    ) -> None:
        super().__init__()
        self.num_modes = num_modes
        self.num_poses = num_poses

        self.agent_encoder = AgentTokenEncoder(box_dim=box_dim, d_model=d_model)
        self.map_encoder = MapPatchTokenEncoder(
            d_model=d_model,
            num_bev_classes=num_bev_classes,
            cross_bev_channels=cross_bev_channels,
            map_size=map_size,
        )
        self.ego_encoder = EgoTokenEncoder(
            status_dim=status_dim,
            encoding_dim=encoding_dim,
            d_model=d_model,
        )

        self.traj_query = nn.Parameter(torch.zeros(1, num_modes, d_model))
        nn.init.normal_(self.traj_query, std=0.02)

        if plan_anchor is not None:
            # plan_anchor shape: [num_modes, num_poses, 2 or 3]
            plan_anchor = np.load(plan_anchor)

            self.plan_anchor = nn.Parameter(
                torch.tensor(plan_anchor, dtype=torch.float32),
                requires_grad=False,
            )  # 20,8,2
            self.anchor_proj = MLP(16, d_model, d_model)
        else:
            self.plan_anchor = None
            self.anchor_proj = None

        self.traj_type_embed = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.traj_type_embed, std=0.02)

        self.graph = SceneGraphTransformer(
            d_model=d_model,
            num_layers=graph_layers,
            nhead=nhead,
            dropout=dropout,
        )

        self.heads = TrajectoryReconstructionHeads(
            d_model=d_model,
            num_poses=num_poses,
            feat_dim=traj_feat_dim,
        )

    def forward(
        self,
        agent_states: torch.Tensor,
        agent_labels: torch.Tensor,
        agents_query: torch.Tensor,
        bev_semantic_map: torch.Tensor,
        cross_bev_feature: torch.Tensor,
        status_feature: torch.Tensor,
        status_encoding: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        bsz = agent_states.shape[0]

        traj_tokens = self.traj_query.expand(bsz, -1, -1)
        if self.plan_anchor is not None:
            anchor = self.plan_anchor.reshape(self.num_modes, -1)
            anchor_embed = self.anchor_proj(anchor).unsqueeze(0)
            traj_tokens = traj_tokens + anchor_embed
        traj_tokens = traj_tokens + self.traj_type_embed

        agent_tokens = self.agent_encoder(agent_states, agent_labels, agents_query)
        map_tokens = self.map_encoder(bev_semantic_map, cross_bev_feature)
        ego_token = self.ego_encoder(status_feature, status_encoding)

        graph_tokens = torch.cat(
            [traj_tokens, agent_tokens, map_tokens, ego_token],
            dim=1,
        )
        graph_tokens = self.graph(graph_tokens)

        traj_out = graph_tokens[:, : self.num_modes]
        pred_all_trajectory, pred_im_scores, pred_sim_scores, pred_feature = self.heads(traj_out)


        return {
            "pred_all_trajectory": pred_all_trajectory,
            "pred_trajectory_im_scores": pred_im_scores,
            "pred_trajectory_sim_scores": pred_sim_scores,
            "pred_trajectory_feature": pred_feature,
            "traj_tokens": traj_out,
            "graph_tokens": graph_tokens,
        }


def trajectory_reconstruction_loss(
    pred_all_trajectory: torch.Tensor,
    pred_trajectory_im_scores: torch.Tensor,
    pred_trajectory_sim_scores: torch.Tensor,
    target_all_trajectory: torch.Tensor,
    target_trajectory_im_scores: torch.Tensor,
    target_trajectory_sim_scores: torch.Tensor,
    beta_score: float = 5.0,
) -> Dict[str, torch.Tensor]:
    """Training loss for trajectory reconstruction."""

    loss_reg = F.smooth_l1_loss(
        pred_all_trajectory,
        target_all_trajectory.detach(),
    )

    im_target = target_trajectory_im_scores.detach()

    # 2. 对生成器的 Logits 取 Log-Softmax（内部自动处理数值稳定性，绝无 NaN）
    im_log_probs = F.log_softmax(pred_trajectory_im_scores, dim=-1)
    # 3. 计算 KL 散度 / 交叉熵（对类别维度求和，对 Batch 取平均）
    im_loss_score = -torch.sum(im_target * im_log_probs, dim=-1).mean()

    sim_target = target_trajectory_sim_scores.detach()
    bce_loss_per_element = F.binary_cross_entropy_with_logits(
        pred_trajectory_sim_scores,  # 生成器的原始分数
        sim_target,  # 冻结模型输出的概率 [0~1]
        reduction='none'  # 先不降维，保留形状以便乘以 5
    )

    # 3. 与你的旧代码对齐：对所有元素取平均，再乘以 5（指标数量）
    sim_reward_loss = torch.mean(bce_loss_per_element)

    # print("loss_reg", loss_reg)
    # print("im_loss_score", im_loss_score)
    # print("sim_reward_loss", sim_reward_loss)

    loss = 500 * loss_reg +  im_loss_score + beta_score * sim_reward_loss
    return {
        "loss": loss,
        "loss_reg": loss_reg,
        "loss_score": im_loss_score
    }


@torch.no_grad()
def trajectory_reconstruction_anomaly_score(
    pred_all_trajectory: torch.Tensor,
    pred_trajectory_im_scores: torch.Tensor,
    pred_trajectory_sim_scores: torch.Tensor,
    target_all_trajectory: torch.Tensor,
    target_trajectory_im_scores: torch.Tensor,
    target_trajectory_sim_scores: torch.Tensor,
    beta_score: float = 5.0,
) -> Dict[str, torch.Tensor]:
    """Per-sample reconstruction residual for reliability assessment."""

    reg_error = F.smooth_l1_loss(
        pred_all_trajectory,
        target_all_trajectory,
        reduction="none",
    ).mean(dim=(1, 2, 3))

    im_target = target_trajectory_im_scores.detach()

    # 2. 对生成器的 Logits 取 Log-Softmax（内部自动处理数值稳定性，绝无 NaN）
    im_log_probs = F.log_softmax(pred_trajectory_im_scores, dim=-1)
    # 3. 计算 KL 散度 / 交叉熵（对类别维度求和，对 Batch 取平均）
    im_score_error = -torch.sum(im_target * im_log_probs, dim=-1).mean()

    sim_target = target_trajectory_sim_scores.detach()
    bce_loss_per_element = F.binary_cross_entropy_with_logits(
        pred_trajectory_sim_scores,  # 生成器的原始分数
        sim_target,  # 冻结模型输出的概率 [0~1]
        reduction='none'  # 先不降维，保留形状以便乘以 5
    )
    sim_score_error = torch.mean(bce_loss_per_element)

    anomaly_score = reg_error + im_score_error + beta_score * sim_score_error
    return {
        "anomaly_score": anomaly_score,
        "reg_error": reg_error,
        "score_error": im_score_error,
    }


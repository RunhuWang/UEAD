from typing import Dict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from navsim.agents.uead_main.transfuser_config import TransfuserConfig
from navsim.agents.uead_main.transfuser_backbone import TransfuserBackbone, TransfuserBackbone_RGB
from navsim.agents.uead_main.transfuser_features import BoundingBox2DIndex
from navsim.common.enums import StateSE2Index
from navsim.agents.uead_main.edge_detection import cross_entropy_loss_RCF

from navsim.agents.uead_main.blocks import linear_relu_ln,bias_init_with_prob, LossComputer
from navsim.agents.uead_main.transformer import TrajectoryHead
class TransfuserModel(nn.Module):
    """Torch module for Transfuser."""

    def __init__(self, config: TransfuserConfig):
        """
        Initializes TransFuser torch module.
        :param config: global config dataclass of TransFuser.
        """

        super().__init__()

        self._config = config
        if config.edge:
            self._backbone = TransfuserBackbone(config)
        else:
            self._backbone = TransfuserBackbone_RGB(config)

        self.trajectory_head = TrajectoryHead(
            num_poses=config.trajectory_sampling.num_poses,
            d_ffn=config.tf_d_ffn,
            d_model=config.tf_d_model,
            plan_anchor_path=config.plan_anchor_path,
            config=config)

        self.speed_branch = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.Dropout(p=0.5),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
        )

        self.measurements = nn.Sequential(
            nn.Linear(4 + 2 + 2, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
        )

        if config.detect_boxes:
            self._keyval_embedding = nn.Embedding(64 * 64 + 1, config.tf_d_model)
            self._query_embedding = nn.Embedding(config.num_bounding_boxes, config.tf_d_model)
            tf_decoder_layer = nn.TransformerDecoderLayer(
                d_model=config.tf_d_model,
                nhead=config.tf_num_head,
                dim_feedforward=config.tf_d_ffn,
                dropout=config.tf_dropout,
                batch_first=True,
            )

            self._tf_decoder = nn.TransformerDecoder(tf_decoder_layer, config.tf_num_layers)
            self._agent_head = AgentHead(
                num_agents=config.num_bounding_boxes,
                d_ffn=config.tf_d_ffn,
                d_model=config.tf_d_model,
            )

            self._bev_semantic_head = nn.Sequential(
                nn.Conv2d(
                    256,
                    config.bev_features_channels,
                    kernel_size=(3, 3),
                    stride=1,
                    padding=(1, 1),
                    bias=True,
                ),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    config.bev_features_channels,
                    config.num_bev_classes,
                    kernel_size=(1, 1),
                    stride=1,
                    padding=0,
                    bias=True,
                ),
                nn.Upsample(
                    size=(config.lidar_resolution_height // 2, config.lidar_resolution_width),
                    mode="bilinear",
                    align_corners=False,
                ),
            )

    def forward(self, features: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]=None) -> Dict[str, torch.Tensor]:
        """Torch module forward pass."""
        camera_feature: torch.Tensor = features["camera_feature"]
        status_feature: torch.Tensor = features["status_feature"]

        if self._config.edge:
            feature_emb, bev_feature, pred_edges = self._backbone(camera_feature)
        else:
            feature_emb, bev_feature = self._backbone(camera_feature)

        measurement_feature = self.measurements(status_feature)
        pred_speed = self.speed_branch(feature_emb)

        pred_wp = self.trajectory_head(bev_feature, measurement_feature)

        output: Dict[str, torch.Tensor] = {"speed": pred_speed}
        output.update(pred_wp)

        if self._config.detect_boxes:

            bev_semantic_map = self._bev_semantic_head(bev_feature)
            output.update({"bev_semantic_map": bev_semantic_map})

            batch_size = status_feature.shape[0]
            bev_feature = bev_feature.flatten(-2, -1).permute(0, 2, 1)
            keyval = torch.concatenate([bev_feature, measurement_feature[:, None]], dim=1)
            keyval += self._keyval_embedding.weight[None, ...]
            query = self._query_embedding.weight[None, ...].repeat(batch_size, 1, 1)
            query_out = self._tf_decoder(query, keyval)
            agents = self._agent_head(query_out)

            output.update(agents)


        if self._config.edge:
            output.update({"edge0": torch.sigmoid(pred_edges[0])})
            output.update({"edge1": torch.sigmoid(pred_edges[1])})
            output.update({"edge2": torch.sigmoid(pred_edges[2])})


            edge_feature0: torch.Tensor = features["laplacian_edge_feature0"]
            edge_feature1: torch.Tensor = features["laplacian_edge_feature1"]
            edge_feature2: torch.Tensor = features["laplacian_edge_feature2"]
            edge_targets_list = [edge_feature0, edge_feature1, edge_feature2]

            edge_loss = 0

            for j in range(len(pred_edges)):
                edge_loss += cross_entropy_loss_RCF(pred_edges[j], edge_targets_list[j], beta=1.1)


            output.update({"edge_loss": edge_loss})


        speed = status_feature[:, 4:6]
        speed = torch.sum(speed**2, dim=1)
        speed_loss = F.l1_loss(pred_speed, speed.view(-1, 1))
        output.update({"speed_loss": speed_loss})

        return output

class AgentHead(nn.Module):
    """Bounding box prediction head."""

    def __init__(
        self,
        num_agents: int,
        d_ffn: int,
        d_model: int,
    ):
        """
        Initializes prediction head.
        :param num_agents: maximum number of agents to predict
        :param d_ffn: dimensionality of feed-forward network
        :param d_model: input dimensionality
        """
        super(AgentHead, self).__init__()

        self._num_objects = num_agents

        self._d_ffn = d_ffn
        self._d_model = d_model
        self._mlp_states = nn.Sequential(
            nn.Linear(self._d_model, self._d_ffn),
            nn.ReLU(),
            nn.Linear(self._d_ffn, BoundingBox2DIndex.size()),
        )

        self._mlp_label = nn.Sequential(
            nn.Linear(self._d_model, 1),
        )

    def forward(self, agent_queries) -> Dict[str, torch.Tensor]:
        """Torch module forward pass."""

        agent_states = self._mlp_states(agent_queries)
        agent_states[..., BoundingBox2DIndex.POINT] = agent_states[..., BoundingBox2DIndex.POINT].tanh() * 32
        agent_states[..., BoundingBox2DIndex.HEADING] = agent_states[..., BoundingBox2DIndex.HEADING].tanh() * np.pi

        agent_labels = self._mlp_label(agent_queries).squeeze(dim=-1)

        return {"agent_states": agent_states, "agent_labels": agent_labels}

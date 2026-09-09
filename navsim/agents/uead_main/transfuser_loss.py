from typing import Dict, List
from scipy.optimize import linear_sum_assignment

import torch
import torch.nn.functional as F

from navsim.agents.uead_main.transfuser_config import TransfuserConfig
from navsim.agents.uead_main.transfuser_features import BoundingBox2DIndex
from navsim.agents.uead_main.blocks import LossComputer

planning_loss = LossComputer()

def transfuser_loss(
    targets: Dict[str, torch.Tensor], predictions: Dict[str, torch.Tensor], config: TransfuserConfig
):
    """
    Helper function calculating complete loss of Transfuser
    :param targets: dictionary of name tensor pairings
    :param predictions: dictionary of name tensor pairings
    :param config: global Transfuser config
    :return: combined loss value
    """


    if "plan_anchor" in predictions:
        trajectory_loss = _traj_loss(targets, predictions)
    else:
        trajectory_loss = F.l1_loss(predictions["trajectory"], targets["trajectory"])

    if config.edge:

        loss = (
            config.trajectory_weight * trajectory_loss
            + config.edge_weight * predictions["edge_loss"]
            + config.speed_weight * predictions["speed_loss"]
        )

        loss_dict = {
            'loss': loss,
            'trajectory_loss': config.trajectory_weight*trajectory_loss,
            'edge_loss': config.edge_weight * predictions["edge_loss"],
            'speed_loss': config.speed_weight * predictions["speed_loss"]
        }
    else:
        if config.detect_boxes:
            agent_class_loss, agent_box_loss = _agent_loss(targets, predictions, config)
            bev_semantic_loss = F.cross_entropy(
                predictions["bev_semantic_map"], targets["bev_semantic_map"].long())
            loss = (
                    config.trajectory_weight * trajectory_loss
                    + config.speed_weight * predictions["speed_loss"]
                    + config.agent_class_weight * agent_class_loss
                    + config.agent_box_weight * agent_box_loss
                    + config.bev_semantic_weight * bev_semantic_loss
            )

            loss_dict = {
                'loss': loss,
                'trajectory_loss': config.trajectory_weight * trajectory_loss,
                'speed_loss': config.speed_weight * predictions["speed_loss"],
                'agent_class_loss': config.agent_class_weight * agent_class_loss,
                'agent_box_loss': config.agent_box_weight * agent_box_loss,
                'bev_semantic_loss': config.bev_semantic_weight * bev_semantic_loss,
            }
        else:
            loss = (
                    config.trajectory_weight * trajectory_loss
                    + config.speed_weight * predictions["speed_loss"]
            )

            loss_dict = {
                'loss': loss,
                'trajectory_loss': config.trajectory_weight * trajectory_loss,
                'speed_loss': config.speed_weight * predictions["speed_loss"]
            }

    return loss_dict

def _traj_loss(targets: Dict[str, torch.Tensor],  predictions: Dict[str, torch.Tensor]):

    reg_list = [predictions['reg_0'], predictions['reg_1'], predictions['reg_2']]
    cls_list = [predictions['cls_0'], predictions['cls_1'], predictions['cls_2']]
    plan_anchor = predictions["plan_anchor"]
    trajectory_loss_dict = {}
    ret_traj_loss = 0
    for idx, (poses_reg, poses_cls) in enumerate(zip(reg_list, cls_list)):
        trajectory_loss = planning_loss(poses_reg, poses_cls, targets, plan_anchor)
        trajectory_loss_dict[f"trajectory_loss_{idx}"] = trajectory_loss
        ret_traj_loss += trajectory_loss

    return ret_traj_loss

def _agent_loss(
    targets: Dict[str, torch.Tensor], predictions: Dict[str, torch.Tensor], config: TransfuserConfig
):
    """
    Hungarian matching loss for agent detection
    :param targets: dictionary of name tensor pairings
    :param predictions: dictionary of name tensor pairings
    :param config: global Transfuser config
    :return: detection loss
    """

    gt_states, gt_valid = targets["agent_states"], targets["agent_labels"]
    pred_states, pred_logits = predictions["agent_states"], predictions["agent_labels"]

    if config.latent:
        rad_to_ego = torch.arctan2(
            gt_states[..., BoundingBox2DIndex.Y],
            gt_states[..., BoundingBox2DIndex.X],
        )

        in_latent_rad_thresh = torch.logical_and(
            -config.latent_rad_thresh <= rad_to_ego,
            rad_to_ego <= config.latent_rad_thresh,
        )
        gt_valid = torch.logical_and(in_latent_rad_thresh, gt_valid)

    # save constants
    batch_dim, num_instances = pred_states.shape[:2]
    num_gt_instances = gt_valid.sum()
    num_gt_instances = num_gt_instances if num_gt_instances > 0 else num_gt_instances + 1

    ce_cost = _get_ce_cost(gt_valid, pred_logits)
    l1_cost = _get_l1_cost(gt_states, pred_states, gt_valid)

    cost = config.agent_class_weight * ce_cost + config.agent_box_weight * l1_cost
    cost = cost.cpu()

    indices = [linear_sum_assignment(c) for i, c in enumerate(cost)]
    matching = [
        (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
        for i, j in indices
    ]
    idx = _get_src_permutation_idx(matching)

    pred_states_idx = pred_states[idx]
    gt_states_idx = torch.cat([t[i] for t, (_, i) in zip(gt_states, indices)], dim=0)

    pred_valid_idx = pred_logits[idx]
    gt_valid_idx = torch.cat([t[i] for t, (_, i) in zip(gt_valid, indices)], dim=0).float()

    l1_loss = F.l1_loss(pred_states_idx, gt_states_idx, reduction="none")
    l1_loss = l1_loss.sum(-1) * gt_valid_idx
    l1_loss = l1_loss.view(batch_dim, -1).sum() / num_gt_instances

    ce_loss = F.binary_cross_entropy_with_logits(pred_valid_idx, gt_valid_idx, reduction="none")
    ce_loss = ce_loss.view(batch_dim, -1).mean()

    return ce_loss, l1_loss


@torch.no_grad()
def _get_ce_cost(gt_valid: torch.Tensor, pred_logits: torch.Tensor) -> torch.Tensor:
    """
    Function to calculate cross-entropy cost for cost matrix.
    :param gt_valid: tensor of binary ground-truth labels
    :param pred_logits: tensor of predicted logits of neural net
    :return: bce cost matrix as tensor
    """

    # NOTE: numerically stable BCE with logits
    # https://github.com/pytorch/pytorch/blob/c64e006fc399d528bb812ae589789d0365f3daf4/aten/src/ATen/native/Loss.cpp#L214
    gt_valid_expanded = gt_valid[:, :, None].detach().float()  # (b, n, 1)
    pred_logits_expanded = pred_logits[:, None, :].detach()  # (b, 1, n)

    max_val = torch.relu(-pred_logits_expanded)
    helper_term = max_val + torch.log(
        torch.exp(-max_val) + torch.exp(-pred_logits_expanded - max_val)
    )
    ce_cost = (1 - gt_valid_expanded) * pred_logits_expanded + helper_term  # (b, n, n)
    ce_cost = ce_cost.permute(0, 2, 1)

    return ce_cost


@torch.no_grad()
def _get_l1_cost(
    gt_states: torch.Tensor, pred_states: torch.Tensor, gt_valid: torch.Tensor
) -> torch.Tensor:
    """
    Function to calculate L1 cost for cost matrix.
    :param gt_states: tensor of ground-truth bounding boxes
    :param pred_states: tensor of predicted bounding boxes
    :param gt_valid: mask of binary ground-truth labels
    :return: l1 cost matrix as tensor
    """

    gt_states_expanded = gt_states[:, :, None, :2].detach()  # (b, n, 1, 2)
    pred_states_expanded = pred_states[:, None, :, :2].detach()  # (b, 1, n, 2)
    l1_cost = gt_valid[..., None].float() * (gt_states_expanded - pred_states_expanded).abs().sum(
        dim=-1
    )
    l1_cost = l1_cost.permute(0, 2, 1)
    return l1_cost


def _get_src_permutation_idx(indices):
    """
    Helper function to align indices after matching
    :param indices: matched indices
    :return: permuted indices
    """
    # permute predictions following indices
    batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
    src_idx = torch.cat([src for (src, _) in indices])
    return batch_idx, src_idx

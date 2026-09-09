"""Measure parameter count and single-scene GPU inference latency of an agent.

Example:
python navsim/planning/script/run_gpu_agent_benchmark.py \
    agent=diffusiondrive_agent \
    agent.checkpoint_path=/path/to/checkpoint.pth \
    experiment_name=gpu_benchmark \
    +num_scenarios=100 +warmup_scenarios=5 +device=cuda:0
"""

from pathlib import Path

import hydra
from hydra.utils import instantiate
import numpy as np
from omegaconf import DictConfig
import torch
import time
from navsim.common.dataloader import SceneLoader


CONFIG_PATH = "config/pdm_scoring"
CONFIG_NAME = "default_run_pdm_score"


def infer_one_scene(agent, agent_input, device):
    """Build features on CPU, then time only the batch-size-one GPU forward pass."""

    features = {}
    for builder in agent.get_feature_builders():
        features.update(builder.compute_features(agent_input))
    features = {name: value.unsqueeze(0).to(device) for name, value in features.items()}

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    with torch.inference_mode():
        out = agent(features)
    end.record()
    torch.cuda.synchronize(device)
    return start.elapsed_time(end)


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig):
    device = torch.device(cfg.get("device", "cuda:0"))
    num_scenarios = int(cfg.get("num_scenarios", 100))
    warmup_scenarios = int(cfg.get("warmup_scenarios", 5))

    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("This script requires a CUDA device")
    if num_scenarios < 1 or warmup_scenarios < 0:
        raise ValueError("num_scenarios must be > 0 and warmup_scenarios must be >= 0")

    torch.cuda.set_device(device)
    agent = instantiate(cfg.agent)
    agent.initialize()
    agent.to(device).eval()

    total_params = sum(parameter.numel() for parameter in agent.parameters())
    scene_loader = SceneLoader(
        sensor_blobs_path=Path(cfg.sensor_blobs_path),
        data_path=Path(cfg.navsim_log_path),
        scene_filter=instantiate(cfg.train_test_split.scene_filter),
        sensor_config=agent.get_sensor_config(),
    )
    tokens = sorted(scene_loader.tokens)[:num_scenarios]
    if not tokens:
        raise RuntimeError("No scenes found")

    # Warm-up results are discarded so CUDA initialization does not skew timing.
    for token in tokens[: min(warmup_scenarios, len(tokens))]:
        agent_input = scene_loader.get_agent_input_from_token(token)
        infer_one_scene(agent, agent_input, device)

    latencies_ms = []
    for index, token in enumerate(tokens, start=1):
        agent_input = scene_loader.get_agent_input_from_token(token)
        latency_ms = infer_one_scene(agent, agent_input, device)
        latencies_ms.append(latency_ms)
        print(f"[{index:4d}/{len(tokens)}] {token}: {latency_ms:.3f} ms")
    print("\n===== GPU benchmark =====")
    print(f"Agent:              {agent.name()}")
    print(f"GPU:                {torch.cuda.get_device_name(device)}")
    print(f"Scenes / batch:     {len(tokens)} / 1")
    print(f"Parameters:         {total_params:,}")
    print(f"Mean latency:       {np.mean(latencies_ms):.3f} ms")
    print(f"Median latency:     {np.median(latencies_ms):.3f} ms")
    print(f"P95 latency:        {np.percentile(latencies_ms, 95):.3f} ms")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Evaluacion greedy de CartPole DQN / Dueling DQN."""
import argparse
import typing as tt

import gymnasium as gym
import numpy as np
import torch

from train_base import CartPoleDQN, NAME as BASE_NAME
from train_mejorado import DuelingDQN, NAME as DUELING_NAME


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_net(model: str, device: torch.device, ckpt: tt.Optional[str] = None) -> torch.nn.Module:
    env = gym.make("CartPole-v1")
    obs_size = env.observation_space.shape[0]
    n_actions = env.action_space.n
    env.close()
    if model == "dueling":
        net: torch.nn.Module = DuelingDQN(obs_size, n_actions)
        ckpt = ckpt or f"checkpoints/{DUELING_NAME}.pt"
    else:
        net = CartPoleDQN(obs_size, n_actions)
        ckpt = ckpt or f"checkpoints/{BASE_NAME}.pt"
    net.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    net.to(device)
    net.eval()
    return net


@torch.no_grad()
def evaluate(model: str, episodes: int, device: torch.device,
             ckpt: tt.Optional[str] = None) -> tt.Dict[str, float]:
    net = load_net(model, device, ckpt)
    env = gym.make("CartPole-v1")
    rewards = []
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        total = 0.0
        while not done:
            obs_v = torch.as_tensor(obs).unsqueeze(0).to(device)
            q_vals = net(obs_v)
            action = int(q_vals.max(1)[1].item())
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total += float(reward)
        rewards.append(total)
    env.close()
    arr = np.asarray(rewards, dtype=np.float32)
    return {"mean": float(arr.mean()), "std": float(arr.std()),
            "min": float(arr.min()), "max": float(arr.max())}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("base", "dueling"), default="dueling")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--ckpt", default=None, help="Ruta alternativa al .pt")
    args = parser.parse_args()
    device = get_device()
    stats = evaluate(args.model, args.episodes, device, ckpt=args.ckpt)
    print(f"model={args.model} episodes={args.episodes} "
          f"mean={stats['mean']:.1f} std={stats['std']:.1f} "
          f"min={stats['min']:.0f} max={stats['max']:.0f}")

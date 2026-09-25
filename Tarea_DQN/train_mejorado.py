#!/usr/bin/env python3
import csv
from dataclasses import dataclass
import os
import typing as tt

import gymnasium as gym
import ptan
import ptan.ignite
import torch
import torch.nn as nn
import torch.optim as optim
from ignite.engine import Engine

from lib import common

NAME = "06_dueling_cartpole"


@dataclass
class Hyperparams:
    env_name: str = "CartPole-v1"
    run_name: str = "dueling"
    tuner_mode: bool = False
    episodes_to_solve: int = 1000
    stop_reward: float = 475.0
    replay_size: int = 10_000
    replay_initial: int = 1_000
    target_net_sync: int = 100
    epsilon_frames: int = 5_000
    epsilon_start: float = 1.0
    epsilon_final: float = 0.02
    learning_rate: float = 1e-3
    gamma: float = 0.99
    batch_size: int = 32


# 1. Arquitectura Dueling DQN
class DuelingDQN(nn.Module):
    """Red Dueling DQN con streams de Valor y Ventaja."""
    def __init__(self, obs_size: int, n_actions: int):
        super().__init__()

        # Extracción compartida de características
        self.feature_layer = nn.Sequential(
            nn.Linear(obs_size, 128),
            nn.ReLU()
        )

        # Stream del Valor del Estado V(s)
        self.val_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        # Stream de la Ventaja de Acción A(s, a)
        self.adv_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.feature_layer(x.float())
        val = self.val_stream(feat)
        adv = self.adv_stream(feat)

        # Formula Dueling: Q(s, a) = V(s) + (A(s, a) - mean(A(s, a)))
        return val + (adv - adv.mean(dim=1, keepdim=True))


def make_env() -> gym.Env:
    return gym.make("CartPole-v1")


def train(params: Hyperparams, device: torch.device, _: dict) -> tt.Optional[int]:
    env = make_env()

    obs_size = env.observation_space.shape[0]
    n_actions = env.action_space.n

    # Instanciamos la red Dueling DQN
    net = DuelingDQN(obs_size, n_actions).to(device)
    tgt_net = ptan.agent.TargetNet(net)

    selector = ptan.actions.EpsilonGreedyActionSelector(epsilon=params.epsilon_start)
    epsilon_tracker = common.EpsilonTracker(selector, params)

    agent = ptan.agent.DQNAgent(net, selector, device=device)

    exp_source = ptan.experience.ExperienceSourceFirstLast(
        env, agent, gamma=params.gamma, steps_count=1
    )
    
    buffer = ptan.experience.ExperienceReplayBuffer(
        exp_source, buffer_size=params.replay_size
    )
    optimizer = optim.Adam(net.parameters(), lr=params.learning_rate)

    # Configuración de guardado de CSV
    os.makedirs("results", exist_ok=True)
    csv_file_path = os.path.join("results", "rewards_mejorado.csv")
    csv_file = open(csv_file_path, mode="w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["episode", "reward"])

    episode_count = 0

    def process_batch(engine: Engine, batch: tt.List[ptan.experience.ExperienceFirstLast]) -> dict:
        optimizer.zero_grad()
        loss_v = common.calc_loss_dqn(
            batch, net, tgt_net.target_model, gamma=params.gamma, device=device
        )
        loss_v.backward()
        optimizer.step()

        epsilon_tracker.frame(engine.state.iteration)
        if engine.state.iteration % params.target_net_sync == 0:
            tgt_net.sync()

        return {
            "loss": loss_v.item(),
            "epsilon": selector.epsilon,
        }

    engine = Engine(process_batch)
    common.setup_ignite(engine, params, exp_source, NAME)

    @engine.on(ptan.ignite.EpisodeEvents.EPISODE_COMPLETED)
    def log_csv(trainer: Engine) -> None:
        nonlocal episode_count
        episode_count += 1
        csv_writer.writerow([episode_count, trainer.state.episode_reward])
        csv_file.flush()
    try:
        r = engine.run(
            common.batch_generator(buffer, params.replay_initial, params.batch_size)
        )
    finally:
        csv_file.close()

    os.makedirs("checkpoints", exist_ok=True)
    torch.save(net.state_dict(), f"checkpoints/{NAME}.pt")
    if getattr(r, "solved", False):
        return r.episode
    return None


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


if __name__ == "__main__":
    import random
    import numpy as np
    random.seed(common.SEED)
    np.random.seed(common.SEED)
    torch.manual_seed(common.SEED)
    params = Hyperparams()
    device = get_device()
    train(params, device, {})
#!/usr/bin/env python3
from dataclasses import dataclass
import gymnasium as gym
import ptan
import typing as tt

import torch
import torch.nn as nn
import torch.optim as optim

from ignite.engine import Engine

from lib import common

NAME = "01_baseline_cartpole"

@dataclass
class Hyperparams:
    env_name: str = "CartPole-v1"
    run_name: str = "cartpole"
    tuner_mode: bool = False
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


class CartPoleDQN(nn.Module):
    """Red Neuronal Fully Connected estándar para CartPole."""
    def __init__(self, obs_size: int, n_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.float())


# 1. Definición de la función constructora del entorno
def make_env():
    return gym.make("CartPole-v1")


def train(params: Hyperparams, device: torch.device, _: dict) -> tt.Optional[int]:
    # Instanciamos el entorno usando make_env()
    env = make_env()

    obs_size = env.observation_space.shape[0]
    n_actions = env.action_space.n

    # Red y red objetivo
    net = CartPoleDQN(obs_size, n_actions).to(device)
    tgt_net = ptan.agent.TargetNet(net)

    # Selector y Agente actualizados
    selector = ptan.actions.EpsilonGreedyActionSelector(epsilon=params.epsilon_start)
    epsilon_tracker = common.EpsilonTracker(selector, params)

    agent = ptan.agent.DQNAgent(net, selector, device=device)

    # 3. Experience Source con steps_count=1 explícito
    exp_source = ptan.experience.ExperienceSourceFirstLast(
        env, agent, gamma=params.gamma, steps_count=1
    )
    
    buffer = ptan.experience.ExperienceReplayBuffer(
        exp_source, buffer_size=params.replay_size
    )
    optimizer = optim.Adam(net.parameters(), lr=params.learning_rate)

    def process_batch(engine, batch):
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
    r = engine.run(
        common.batch_generator(buffer, params.replay_initial, params.batch_size)
    )
    if r.solved:
        return r.episode


if __name__ == "__main__":
    params = Hyperparams()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train(params, device, {})
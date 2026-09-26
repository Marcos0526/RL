#!/usr/bin/env python3
"""DQN mejorado (Dueling) para CartPole-v1.

Espejo de train_base.py: mismo loop, mismas semillas, mismo CSV.
Unica diferencia: la red DQN (MLP) se reemplaza por DuelingDQN.
Las secciones modificadas estan marcadas con [MODIFICADO - Dueling].
Genera results/reward_mejorado_seed{seed}.csv con filas (episode, reward).
Uso: python train_mejorado.py --seed 42
"""
import csv
import os
import random
import typing as tt

import gymnasium as gym
import numpy as np
import ptan
from ptan.experience import ExperienceFirstLast
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Hiperparametros (identicos al baseline para comparar solo la arquitectura)
ENV_NAME = "CartPole-v1"
GAMMA = 0.99
HIDDEN_SIZE = 128
BATCH_SIZE = 32
REPLAY_SIZE = 10_000
REPLAY_INITIAL = 1_000
LEARNING_RATE = 1e-3
TARGET_NET_SYNC = 100
EPSILON_START = 1.0
EPSILON_FINAL = 0.02
EPSILON_FRAMES = 5_000
STOP_REWARD = 475.0
MAX_FRAMES = 200_000
# [MODIFICADO - Dueling] nombres propios para no mezclar CSV/ckpt con el base
CSV_TEMPLATE = "results/reward_mejorado_seed{seed}.csv"
CKPT_TEMPLATE = "checkpoints/dqn_dueling_cartpole_seed{seed}.pt"


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ---------------------------------------------------------------------------
# [MODIFICADO - Dueling] Arquitectura Dueling en lugar del MLP simple (DQN).
# Idea: separa Q(s,a) en valor del estado V(s) + ventaja A(s,a):
#   Q(s, a) = V(s) + (A(s, a) - mean_a A(s, a))
# Esto ayuda en CartPole porque muchas observaciones comparten el mismo
# valor V(s) y solo importa la ventaja relativa de mover izq/der.
# El resto del algoritmo (loss MSE, target net, epsilon-greedy, replay)
# NO cambia: Dueling solo cambia como la red parametriza Q.
# ---------------------------------------------------------------------------
class DuelingDQN(nn.Module):
    def __init__(self, obs_size: int, n_actions: int):
        super().__init__()
        # Tronco compartido de caracteristicas (equivale a la 1a capa del DQN base)
        self.features = nn.Sequential(
            nn.Linear(obs_size, HIDDEN_SIZE),
            nn.ReLU(),
        )
        # Stream de valor del estado: V(s), salida escalar
        self.value_stream = nn.Sequential(
            nn.Linear(HIDDEN_SIZE, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        # Stream de ventaja: A(s, a), una salida por accion
        self.advantage_stream = nn.Sequential(
            nn.Linear(HIDDEN_SIZE, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x.float())
        val = self.value_stream(feat)          # (B, 1)
        adv = self.advantage_stream(feat)      # (B, n_actions)
        # Restar la media centra las ventajas (identificabilidad V/A)
        return val + (adv - adv.mean(dim=1, keepdim=True))


def make_env(seed: tt.Optional[int] = None):
    env = gym.make("CartPole-v1")
    if seed is not None:
        env.reset(seed=seed)
        env.action_space.seed(seed)
    return env


def resolve_csv_path(csv_arg: tt.Optional[str], seed: int) -> str:
    """Si no se pasa --csv, genera results/reward_mejorado_seed{seed}.csv."""
    if csv_arg:
        return csv_arg
    return CSV_TEMPLATE.format(seed=seed)


def init_reward_csv(csv_path: str) -> tt.Tuple[tt.Any, csv.writer]:
    """Crea el CSV con cabecera (episode, reward).

    Retorna (file_handle, writer). Cada fila: fila=episode, reward.
    """
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    f = open(csv_path, mode="w", newline="")
    writer = csv.writer(f)
    writer.writerow(["episode", "reward"])
    return f, writer


def log_reward(writer: csv.writer, f: tt.Any, episode: int, reward: float) -> None:
    """Guarda una fila (episode, reward) y hace flush."""
    writer.writerow([episode, reward])
    f.flush()


@torch.no_grad()
def unpack_batch(batch: tt.List[ExperienceFirstLast],
                 net: DuelingDQN, gamma: float):
    # [MODIFICADO - Dueling] solo cambia el tipo anotado (DQN -> DuelingDQN).
    # El calculo del objetivo es el mismo DQN de 1 paso: r + gamma * max_a Q_tgt.
    states, actions, rewards, done_masks, last_states = [], [], [], [], []
    for exp in batch:
        states.append(exp.state)
        actions.append(exp.action)
        rewards.append(exp.reward)
        done_masks.append(exp.last_state is None)
        last_states.append(exp.state if exp.last_state is None else exp.last_state)
    states_v = torch.as_tensor(np.stack(states))
    actions_v = torch.tensor(actions)
    rewards_v = torch.tensor(rewards, dtype=torch.float32)
    # La target-net vive en device (mps/cuda); mover last_states ahi
    dev = next(net.parameters()).device
    last_states_v = torch.as_tensor(np.stack(last_states)).to(dev)
    last_q_v = net(last_states_v).cpu()
    best_last_q_v = last_q_v.max(dim=1)[0]
    best_last_q_v[torch.tensor(done_masks, dtype=torch.bool)] = 0.0
    return states_v, actions_v, rewards_v + gamma * best_last_q_v


def evaluate(net: DuelingDQN, device: torch.device, n_episodes: int = 5) -> float:
    """Evalua en modo greedy con ArgmaxActionSelector (sin exploracion)."""
    env = make_env()
    agent = ptan.agent.DQNAgent(
        net, ptan.actions.ArgmaxActionSelector(), device=device
    )
    exp_source = ptan.experience.ExperienceSourceFirstLast(
        env, agent, gamma=0.99, steps_count=1
    )
    _ = exp_source  # se usa el mismo patron make_env/agent/exp_source que en train
    total = 0.0
    net.eval()
    with torch.no_grad():
        for _ in range(n_episodes):
            obs, _ = env.reset()
            done, ep_r = False, 0.0
            while not done:
                obs_v = torch.as_tensor(np.array([obs])).to(device)
                action = int(net(obs_v).argmax(dim=1).item())
                obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                ep_r += reward
            total += ep_r
    net.train()
    env.close()
    return total / max(n_episodes, 1)


def train(device: torch.device, seed: int = 2024,
          csv_path: tt.Optional[str] = None,
          ckpt_path: tt.Optional[str] = None) -> tt.Optional[int]:
    set_seeds(seed)
    csv_path = resolve_csv_path(csv_path, seed)
    ckpt_path = ckpt_path or CKPT_TEMPLATE.format(seed=seed)

    env = make_env(seed)
    obs_size = env.observation_space.shape[0]
    n_actions = env.action_space.n

    # [MODIFICADO - Dueling] se instancia DuelingDQN en vez de DQN.
    net = DuelingDQN(obs_size, n_actions).to(device)
    tgt_net = ptan.agent.TargetNet(net)

    selector = ptan.actions.EpsilonGreedyActionSelector(epsilon=EPSILON_START)
    agent = ptan.agent.DQNAgent(net, selector, device=device)
    exp_source = ptan.experience.ExperienceSourceFirstLast(
        env, agent, gamma=0.99, steps_count=1, env_seed=seed
    )
    buffer = ptan.experience.ExperienceReplayBuffer(exp_source, buffer_size=REPLAY_SIZE)
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)

    csv_file, csv_writer = init_reward_csv(csv_path)

    frame = 0
    episode = 0
    try:
        while frame < MAX_FRAMES:
            frame += 1
            # decaimiento lineal de epsilon
            eps = max(EPSILON_FINAL, EPSILON_START - frame / EPSILON_FRAMES)
            selector.epsilon = eps
            buffer.populate(1)

            for reward, _ in exp_source.pop_rewards_steps():
                episode += 1
                log_reward(csv_writer, csv_file, episode, reward)
                print(f"frame={frame} episode={episode} reward={reward:.1f} eps={eps:.3f}")
                if reward >= STOP_REWARD:
                    print("Resuelto!")
                    os.makedirs(os.path.dirname(ckpt_path) or ".", exist_ok=True)
                    torch.save(net.state_dict(), ckpt_path)
                    print(f"CSV: {csv_path} | ckpt: {ckpt_path}")
                    return episode

            if len(buffer) < REPLAY_INITIAL:
                continue
            batch = buffer.sample(BATCH_SIZE)
            states_v, actions_v, tgt_q_v = unpack_batch(batch, tgt_net.target_model, GAMMA)
            states_v = states_v.to(device)
            actions_v = actions_v.to(device)
            tgt_q_v = tgt_q_v.to(device)
            optimizer.zero_grad()
            q_v = net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
            loss_v = F.mse_loss(q_v, tgt_q_v)
            loss_v.backward()
            optimizer.step()

            if frame % TARGET_NET_SYNC == 0:
                tgt_net.sync()
    finally:
        csv_file.close()
        env.close()
    os.makedirs(os.path.dirname(ckpt_path) or ".", exist_ok=True)
    torch.save(net.state_dict(), ckpt_path)
    print(f"CSV: {csv_path} | ckpt: {ckpt_path}")
    return None


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42,
                    help="Semilla (random, numpy, torch, env). Ej: --seed 0 --seed 42 --seed 123")
    ap.add_argument("--csv", default=None,
                    help="Ruta CSV. Si se omite: results/reward_mejorado_seed{seed}.csv")
    ap.add_argument("--ckpt", default=None,
                    help="Ruta checkpoint. Si se omite: checkpoints/dqn_dueling_cartpole_seed{seed}.pt")
    args = ap.parse_args()
    train(get_device(), seed=args.seed, csv_path=args.csv, ckpt_path=args.ckpt)

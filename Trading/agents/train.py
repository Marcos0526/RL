"""Entrena un agente PPO o A2C sobre ``data/train_data.csv``."""

from __future__ import annotations
import argparse
import sys
from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from envs.stock_trading_env import StockTradingEnv


def main() -> None:
    parser = argparse.ArgumentParser(description="Entrena un agente de trading.")
    parser.add_argument("--algo", choices=("PPO", "A2C"), default="PPO")
    parser.add_argument("--timesteps", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--initial-cash", type=float, default=10_000.0)
    parser.add_argument(
        "--train-csv", type=Path, default=PROJECT_ROOT / "data" / "train_data.csv"
    )
    args = parser.parse_args()
    if args.timesteps <= 0:
        parser.error("--timesteps debe ser mayor que cero.")

    from stable_baselines3 import A2C, PPO

    df = pd.read_csv(args.train_csv, index_col=0, parse_dates=True)
    env = StockTradingEnv(df, initial_cash=args.initial_cash)
    algorithm = PPO if args.algo == "PPO" else A2C
    model = algorithm("MlpPolicy", env, verbose=1, seed=args.seed)
    model.learn(total_timesteps=args.timesteps)

    model_path = PROJECT_ROOT / "models" / "ppo_stock_trading"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(model_path))
    print(f"Modelo guardado en: {model_path}.zip")
    env.close()


if __name__ == "__main__":
    main()

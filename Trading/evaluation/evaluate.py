"""Evalúa el agente guardado frente a una estrategia Buy and Hold."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from envs.stock_trading_env import StockTradingEnv


def total_return(portfolio: np.ndarray) -> float:
    return float((portfolio[-1] / portfolio[0] - 1) * 100)


def sharpe_ratio(portfolio: np.ndarray, periods_per_year: int = 252) -> float:
    returns = np.diff(portfolio) / np.maximum(portfolio[:-1], 1e-8)
    deviation = returns.std()
    return 0.0 if deviation < 1e-12 else float(
        returns.mean() / deviation * np.sqrt(periods_per_year)
    )


def maximum_drawdown(portfolio: np.ndarray) -> float:
    peaks = np.maximum.accumulate(portfolio)
    drawdowns = (portfolio - peaks) / np.maximum(peaks, 1e-8)
    return float(drawdowns.min() * 100)


def buy_and_hold(
    df: pd.DataFrame, initial_cash: float, commission: float = 0.001
) -> np.ndarray:
    price = float(df["Close"].iloc[0])
    shares = int(initial_cash // (price * (1 + commission)))
    cash = initial_cash - shares * price * (1 + commission)
    return cash + shares * df["Close"].to_numpy(dtype=float)


def run_agent(model, env: StockTradingEnv):
    observation, _ = env.reset()
    done = False
    while not done:
        action, _ = model.predict(observation, deterministic=True)
        observation, _, terminated, truncated, _ = env.step(int(action))
        done = terminated or truncated
    return (
        np.asarray(env.portfolio_history, dtype=float),
        np.asarray(env.price_history[1:], dtype=float),
        np.asarray(env.action_history, dtype=int),
    )


def print_metrics(name: str, portfolio: np.ndarray) -> None:
    print(
        f"{name:<11} | Retorno total: {total_return(portfolio):+.2f}% "
        f"| Sharpe: {sharpe_ratio(portfolio):+.3f} "
        f"| Maximum Drawdown: {maximum_drawdown(portfolio):.2f}%"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evalúa el agente frente a Buy and Hold.")
    parser.add_argument(
        "--model", type=Path, default=PROJECT_ROOT / "models" / "ppo_stock_trading.zip"
    )
    parser.add_argument(
        "--test-csv", type=Path, default=PROJECT_ROOT / "data" / "test_data.csv"
    )
    parser.add_argument("--initial-cash", type=float, default=10_000.0)
    parser.add_argument(
        "--save-fig", type=Path, default=PROJECT_ROOT / "evaluation" / "results.png"
    )
    args = parser.parse_args()

    if not args.test_csv.exists():
        raise FileNotFoundError(f"No se encontró el archivo de prueba: {args.test_csv}")
    model_path = args.model if args.model.exists() else args.model.with_suffix(".zip")
    if not model_path.exists():
        raise FileNotFoundError(f"No se encontró el modelo: {args.model}")

    from stable_baselines3 import A2C, PPO

    try:
        model = PPO.load(str(model_path))
    except Exception:
        model = A2C.load(str(model_path))

    df = pd.read_csv(args.test_csv, index_col=0, parse_dates=True)
    env = StockTradingEnv(df, initial_cash=args.initial_cash)
    portfolio, prices, actions = run_agent(model, env)
    benchmark = buy_and_hold(df, args.initial_cash)

    print("\n====== RESULTADOS ==========")
    print_metrics("Agente RL", portfolio)
    print_metrics("Buy & Hold", np.insert(benchmark, 0, args.initial_cash))
    buys = int((actions == StockTradingEnv.BUY).sum())
    sells = int((actions == StockTradingEnv.SELL).sum())
    print(f"Operaciones totales: {buys + sells} (compras={buys}, ventas={sells})")
    print(f"Valor final RL: ${portfolio[-1]:,.2f} | Buy & Hold: ${benchmark[-1]:,.2f}")

    dates = df.index[: len(prices)]
    rl_curve = portfolio[1 : len(prices) + 1]
    benchmark_curve = benchmark[: len(prices)]
    buy_mask = actions == StockTradingEnv.BUY
    sell_mask = actions == StockTradingEnv.SELL

    figure, (portfolio_axis, price_axis) = plt.subplots(
        2, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
    )
    portfolio_axis.plot(dates, rl_curve, label="Agente RL", linewidth=1.8)
    portfolio_axis.plot(
        dates, benchmark_curve, "--", label="Buy & Hold", linewidth=1.5
    )
    portfolio_axis.set_ylabel("Valor del portafolio ($)")
    portfolio_axis.set_title("Agente RL vs Buy & Hold")
    portfolio_axis.legend()
    portfolio_axis.grid(alpha=0.3)

    price_axis.plot(dates, prices, color="black", label="Precio Close")
    price_axis.scatter(dates[buy_mask], prices[buy_mask], marker="^", s=35, label="Compra")
    price_axis.scatter(dates[sell_mask], prices[sell_mask], marker="v", s=35, label="Venta")
    price_axis.set_ylabel("Precio ($)")
    price_axis.set_xlabel("Fecha")
    price_axis.legend()
    price_axis.grid(alpha=0.3)
    figure.tight_layout()
    args.save_fig.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.save_fig, dpi=150)
    print(f"Gráfico guardado en: {args.save_fig}")
    plt.show()


if __name__ == "__main__":
    main()

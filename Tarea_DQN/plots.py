#!/usr/bin/env python3
"""Compara DQN base vs Dueling: curva por par de semillas + resumen agregado.

Uso:
  python plots.py --base results/rewards_base_seed42.csv --dueling results/rewards_mejorado_seed42.csv --out results/comparacion_seed42.png
  python plots.py --all  # genera comparacion por semilla (42,123,2024) + resumen
"""
import argparse
import csv
import pathlib
import typing as tt

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEEDS = (42, 123, 2024)
WINDOW = 100
THRESHOLD = 300.0


def load_csv(path: str) -> tt.Tuple[tt.List[int], tt.List[float]]:
    eps, rews = [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            eps.append(int(row["episode"]))
            rews.append(float(row["reward"]))
    return eps, rews


def moving_avg(rews: tt.List[float], window: int = WINDOW) -> tt.List[float]:
    return [sum(rews[max(0, i - window + 1):i + 1]) / len(rews[max(0, i - window + 1):i + 1])
            for i in range(len(rews))]


def stats(rews: tt.List[float]) -> tt.Dict[str, float]:
    last = rews[-WINDOW:] if len(rews) >= WINDOW else rews
    first_hit = next((i + 1 for i, v in enumerate(moving_avg(rews)) if v >= THRESHOLD), float("nan"))
    return {"episodios": float(len(rews)), "media_ult100": sum(last) / len(last),
            "max": max(rews), f"ep_media100>={THRESHOLD:g}": first_hit}


def compare(base_csv: str, duel_csv: str, out: str) -> None:
    be, br = load_csv(base_csv)
    de, dr = load_csv(duel_csv)
    plt.figure()
    plt.plot(be, br, alpha=0.25, label="base")
    plt.plot(be, moving_avg(br), label="base media100")
    plt.plot(de, dr, alpha=0.25, label="dueling")
    plt.plot(de, moving_avg(dr), label="dueling media100")
    plt.xlabel("episodio")
    plt.ylabel("recompensa")
    plt.legend()
    plt.tight_layout()
    pathlib.Path(out).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out)
    plt.close()
    sb, sd = stats(br), stats(dr)
    print(f"[{pathlib.Path(out).name}] base={sb} dueling={sd}")
    print(f"  mejora media_ult100: {sd['media_ult100'] - sb['media_ult100']:+.1f}  -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="results/rewards_base.csv")
    ap.add_argument("--dueling", default="results/rewards_mejorado.csv")
    ap.add_argument("--out", default="results/comparacion.png")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    if args.all:
        for s in SEEDS:
            compare(f"results/rewards_base_seed{s}.csv",
                    f"results/rewards_mejorado_seed{s}.csv",
                    f"results/comparacion_seed{s}.png")
    else:
        compare(args.base, args.dueling, args.out)

#!/usr/bin/env python3
"""Compara los 3 modelos: base vs mejorado (Dueling) vs mejoradoV2 (Dueling+N-step).

Lee results/reward_{base,mejorado,mejoradoV2}_seed{seed}.csv, grafica por
semilla comun con banda de +/-1 desviacion estandar (media movil +/- std
movil) y reporta tabla:

    metrica | base | mejorado | mejorado_V2
    - Reward final medio (ultimas 100 ep)
    - Desviacion estandar (ultimas 100 ep)
    - Episodio del primer reward >= 475

Uso: python plots_v2.py [--results results] [--out results] [--window 20]
"""
import argparse
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PATTERNS = {
    "base": re.compile(r"reward_base_seed(\d+)\.csv$"),
    "mejorado": re.compile(r"reward_mejorado_seed(\d+)\.csv$"),
    "mejorado_V2": re.compile(r"reward_mejoradoV2_seed(\d+)\.csv$"),
}
COLORS = {"base": "tab:blue", "mejorado": "tab:orange", "mejorado_V2": "tab:green"}
LABELS = {"base": "base (DQN)",
          "mejorado": "mejorado (Dueling)",
          "mejorado_V2": "mejoradoV2 (Dueling+N-step)"}
METHODS = ["base", "mejorado", "mejorado_V2"]
THRESHOLD = 475.0


def find_runs(results_dir: str):
    """Retorna {seed: {metodo: path}} con semillas comunes a los 3 metodos."""
    found = {m: {} for m in METHODS}
    for f in os.listdir(results_dir):
        for m, pat in PATTERNS.items():
            mt = pat.search(f)
            if mt:
                found[m][int(mt.group(1))] = os.path.join(results_dir, f)
    seeds = sorted(set(found["base"]) & set(found["mejorado"]) & set(found["mejorado_V2"]))
    return {s: {m: found[m][s] for m in METHODS} for s in seeds}


def load_rewards(path: str) -> pd.Series:
    df = pd.read_csv(path)
    return df["reward"].astype(float).reset_index(drop=True)


def rolling_stats(s: pd.Series, window: int):
    """Media movil y std movil (banda +/-1 std)."""
    w = max(1, min(window, len(s)))
    mean = s.rolling(w, min_periods=1, center=True).mean()
    std = s.rolling(w, min_periods=1, center=True).std(ddof=0).fillna(0.0)
    return mean, std


def metrics(s: pd.Series, last_n: int = 100, threshold: float = THRESHOLD):
    """Reward final medio, std (ultimas N ep) y 1er episodio con reward >= umbral."""
    tail = s.iloc[-last_n:] if len(s) >= last_n else s
    final_mean = float(tail.mean())
    final_std = float(tail.std(ddof=0))
    hit = s[s >= threshold]
    first_ep = int(hit.index[0]) + 1 if len(hit) else None  # episodios base-1
    return final_mean, final_std, first_ep


def plot_seed(seed: int, series: dict, window: int, out: str):
    fig, ax = plt.subplots(figsize=(11, 5))
    for m in METHODS:
        s = series[m]
        mean, std = rolling_stats(s, window)
        ep = np.arange(1, len(s) + 1)
        c = COLORS[m]
        ax.plot(ep, s, color=c, alpha=0.22, linewidth=0.8)
        ax.plot(ep, mean, color=c, label=f"{LABELS[m]} (media movil)")
        ax.fill_between(ep, mean - std, mean + std, color=c, alpha=0.18,
                        label=f"{LABELS[m]} +/-1 std")
    ax.axhline(THRESHOLD, color="red", linestyle="--", linewidth=1,
               label=f"umbral {THRESHOLD:g}")
    ax.set_title(f"CartPole-v1 seed {seed}: base vs Dueling vs Dueling+N-step")
    ax.set_xlabel("episodio")
    ax.set_ylabel("reward")
    ax.legend(loc="upper left", fontsize="small")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def plot_average(all_series: dict, out: str):
    """Promedio entre semillas con banda +/-1 std (alineado al ep minimo)."""
    fig, ax = plt.subplots(figsize=(11, 5))
    for m in METHODS:
        arrs = [s.to_numpy() for s in all_series[m]]
        n = min(map(len, arrs))
        mat = np.stack([a[:n] for a in arrs])
        mean, std = mat.mean(axis=0), mat.std(axis=0)
        ep = np.arange(1, n + 1)
        c = COLORS[m]
        ax.plot(ep, mean, color=c, label=f"{LABELS[m]} (media entre semillas)")
        ax.fill_between(ep, mean - std, mean + std, color=c, alpha=0.18,
                        label=f"{LABELS[m]} +/-1 std")
    ax.axhline(THRESHOLD, color="red", linestyle="--", linewidth=1,
               label=f"umbral {THRESHOLD:g}")
    ax.set_title("CartPole-v1: promedio entre semillas (3 metodos)")
    ax.set_xlabel("episodio")
    ax.set_ylabel("reward")
    ax.legend(loc="upper left", fontsize="small")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def print_table(rows, title: str):
    w0 = max(len(r[0]) for r in rows + [("metrica",)])
    print(f"\n{title}")
    print(f"{'metrica':<{w0}} | {'base':>12} | {'mejorado':>12} | {'mejorado_V2':>12}")
    print("-" * (w0 + 45))
    for name, b, m, v in rows:
        print(f"{name:<{w0}} | {b:>12} | {m:>12} | {v:>12}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="results")
    ap.add_argument("--window", type=int, default=20,
                    help="ventana de media/std movil para la banda +/-1 std")
    ap.add_argument("--last-n", type=int, default=100)
    ap.add_argument("--threshold", type=float, default=THRESHOLD)
    ap.add_argument("--seeds", type=int, nargs="*", default=None,
                    help="semillas a comparar (default: todas las comunes)")
    args = ap.parse_args()

    runs = find_runs(args.results)
    if args.seeds:
        runs = {s: runs[s] for s in args.seeds if s in runs}
    if not runs:
        raise SystemExit(f"Sin trios base/mejorado/mejoradoV2 en {args.results}")
    os.makedirs(args.out, exist_ok=True)

    all_rows, agg = [], {m: [] for m in METHODS}
    for seed, paths in runs.items():
        series = {m: load_rewards(paths[m]) for m in METHODS}
        for m in METHODS:
            agg[m].append(series[m])
        plot_seed(seed, series, args.window,
                  os.path.join(args.out, f"comparacion3_seed{seed}.png"))

        met = {m: metrics(series[m], args.last_n, args.threshold) for m in METHODS}
        fmt = lambda v: f"{v:.1f}"
        rows = [
            (f"Reward final medio (ultimas {args.last_n} ep)",
             *[fmt(met[m][0]) for m in METHODS]),
            ("Desviacion estandar",
             *[fmt(met[m][1]) for m in METHODS]),
            (f"Episodio primer reward >= {args.threshold:g}",
             *[str(met[m][2]) if met[m][2] else "no alcanzado" for m in METHODS]),
        ]
        ns = ", ".join(f"n_{m}={len(series[m])}" for m in METHODS)
        print_table(rows, f"seed {seed} ({ns})")
        for r in rows:
            all_rows.append((seed,) + r)

    if len(runs) > 1:
        plot_average(agg, os.path.join(args.out, "comparacion3_promedio.png"))
        print(f"\nGrafica promedio entre {len(runs)} semillas: comparacion3_promedio.png")

    tab = pd.DataFrame(all_rows,
                       columns=["seed", "metrica", "base", "mejorado", "mejorado_V2"])
    tab_path = os.path.join(args.out, "tabla_comparativa3.csv")
    tab.to_csv(tab_path, index=False)
    print(f"\nTabla guardada en {tab_path}")
    print("Graficas: " + ", ".join(f"comparacion3_seed{s}.png" for s in runs))


if __name__ == "__main__":
    main()

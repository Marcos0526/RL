#!/usr/bin/env python3
"""Grafica y compara los CSV de train_base.py vs train_mejorado.py.

Lee results/reward_base_seed{seed}.csv y results/reward_mejorado_seed{seed}.csv,
grafica por semilla (base vs mejorado) con banda de +/-1 desviacion estandar
(media movil +/- std movil) y reporta tabla:

    metrica | base | mejorado
    - Reward final medio (ultimas 100 ep)
    - Desviacion estandar (ultimas 100 ep)
    - Episodio del primer reward >= 475

Uso: python plots.py [--results results] [--out results] [--window 20]
"""
import argparse
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_PATTERN = re.compile(r"reward_base_seed(\d+)\.csv$")
MEJ_PATTERN = re.compile(r"reward_mejorado_seed(\d+)\.csv$")
THRESHOLD = 475.0


def find_runs(results_dir: str):
    """Retorna {seed: {'base': path, 'mejorado': path}} con semillas comunes."""
    base, mej = {}, {}
    for f in os.listdir(results_dir):
        m = BASE_PATTERN.search(f)
        if m:
            base[int(m.group(1))] = os.path.join(results_dir, f)
        m = MEJ_PATTERN.search(f)
        if m:
            mej[int(m.group(1))] = os.path.join(results_dir, f)
    seeds = sorted(set(base) & set(mej))
    return {s: {"base": base[s], "mejorado": mej[s]} for s in seeds}


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


def plot_seed(seed: int, sb: pd.Series, sm: pd.Series, window: int, out: str):
    mb, sdb = rolling_stats(sb, window)
    mm, sdm = rolling_stats(sm, window)
    ep_b, ep_m = np.arange(1, len(sb) + 1), np.arange(1, len(sm) + 1)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(ep_b, sb, color="tab:blue", alpha=0.25, linewidth=0.8)
    ax.plot(ep_m, sm, color="tab:orange", alpha=0.25, linewidth=0.8)
    ax.plot(ep_b, mb, color="tab:blue", label="base (media movil)")
    ax.fill_between(ep_b, mb - sdb, mb + sdb, color="tab:blue", alpha=0.2,
                    label="base +/-1 std")
    ax.plot(ep_m, mm, color="tab:orange", label="mejorado (media movil)")
    ax.fill_between(ep_m, mm - sdm, mm + sdm, color="tab:orange", alpha=0.2,
                    label="mejorado +/-1 std")
    ax.axhline(THRESHOLD, color="red", linestyle="--", linewidth=1,
               label=f"umbral {THRESHOLD:g}")
    ax.set_title(f"CartPole-v1 seed {seed}: base vs mejorado (Dueling)")
    ax.set_xlabel("episodio")
    ax.set_ylabel("reward")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def plot_average(series: dict, out: str):
    """Promedio entre semillas con banda +/-1 std (alineado al ep minimo)."""
    fig, ax = plt.subplots(figsize=(10, 5))
    for name, color in (("base", "tab:blue"), ("mejorado", "tab:orange")):
        arrs = [s.to_numpy() for s in series[name]]
        n = min(map(len, arrs))
        mat = np.stack([a[:n] for a in arrs])
        mean, std = mat.mean(axis=0), mat.std(axis=0)
        ep = np.arange(1, n + 1)
        ax.plot(ep, mean, color=color, label=f"{name} (media entre semillas)")
        ax.fill_between(ep, mean - std, mean + std, color=color, alpha=0.2,
                        label=f"{name} +/-1 std")
    ax.axhline(THRESHOLD, color="red", linestyle="--", linewidth=1,
               label=f"umbral {THRESHOLD:g}")
    ax.set_title("CartPole-v1: promedio entre semillas")
    ax.set_xlabel("episodio")
    ax.set_ylabel("reward")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def print_table(rows, title: str):
    w0 = max(len(r[0]) for r in rows + [("metrica",)])
    print(f"\n{title}")
    print(f"{'metrica':<{w0}} | {'base':>12} | {'mejorado':>12}")
    print("-" * (w0 + 30))
    for name, b, m in rows:
        print(f"{name:<{w0}} | {b:>12} | {m:>12}")


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
        raise SystemExit(f"Sin pares base/mejorado en {args.results}")
    os.makedirs(args.out, exist_ok=True)

    all_rows, agg = [], {"base": [], "mejorado": []}
    for seed, paths in runs.items():
        sb, sm = load_rewards(paths["base"]), load_rewards(paths["mejorado"])
        agg["base"].append(sb)
        agg["mejorado"].append(sm)
        plot_seed(seed, sb, sm, args.window,
                  os.path.join(args.out, f"comparacion_seed{seed}.png"))

        b_mean, b_std, b_ep = metrics(sb, args.last_n, args.threshold)
        m_mean, m_std, m_ep = metrics(sm, args.last_n, args.threshold)
        rows = [
            (f"Reward final medio (ultimas {args.last_n} ep)",
             f"{b_mean:.1f}", f"{m_mean:.1f}"),
            ("Desviacion estandar", f"{b_std:.1f}", f"{m_std:.1f}"),
            (f"Episodio primer reward >= {args.threshold:g}",
             str(b_ep) if b_ep else "no alcanzado",
             str(m_ep) if m_ep else "no alcanzado"),
        ]
        print_table(rows, f"seed {seed} (n_base={len(sb)}, n_mej={len(sm)})")
        for r in rows:
            all_rows.append((seed,) + r)

    if len(runs) > 1:
        plot_average(agg, os.path.join(args.out, "comparacion_promedio.png"))
        print(f"\nGrafica promedio entre {len(runs)} semillas: comparacion_promedio.png")

    tab = pd.DataFrame(all_rows, columns=["seed", "metrica", "base", "mejorado"])
    tab_path = os.path.join(args.out, "tabla_comparativa.csv")
    tab.to_csv(tab_path, index=False)
    print(f"\nTabla guardada en {tab_path}")
    print("Graficas: " + ", ".join(f"comparacion_seed{s}.png" for s in runs))


if __name__ == "__main__":
    main()

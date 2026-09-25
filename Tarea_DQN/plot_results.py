#!/usr/bin/env python3
"""Genera results/reward_curve.png desde results/rewards_mejorado.csv."""
import csv
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC = pathlib.Path("results/rewards_mejorado.csv")
DST = pathlib.Path("results/reward_curve.png")

eps, rews = [], []
with open(SRC) as f:
    reader = csv.DictReader(f)
    for row in reader:
        eps.append(int(row["episode"]))
        rews.append(float(row["reward"]))

window = 100
avg = [sum(rews[max(0, i - window + 1):i + 1]) / len(rews[max(0, i - window + 1):i + 1])
       for i in range(len(rews))]

plt.figure()
plt.plot(eps, rews, alpha=0.3, label="reward")
plt.plot(eps, avg, label=f"media movil {window}")
plt.xlabel("episodio")
plt.ylabel("recompensa")
plt.legend()
plt.tight_layout()
DST.parent.mkdir(exist_ok=True)
plt.savefig(DST)
print(f"guardada {DST} con {len(eps)} episodios")

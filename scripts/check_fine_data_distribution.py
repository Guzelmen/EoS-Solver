"""
Quick diagnostic: inspect T, alpha, x distributions in the fine dataset
used to compute norm stats.
"""
import sys
import torch
import numpy as np

FINE_DATA_PATH = "/rds/general/user/gs1622/home/mini PINN/data/phase_4_solver_mega_fine.pt"

raw = torch.load(FINE_DATA_PATH, map_location="cpu")
inputs = raw["inputs"]  # [N, 3]: col0=x, col1=alpha, col2=T

x     = inputs[:, 0].numpy()
alpha = inputs[:, 1].numpy()
T     = inputs[:, 2].numpy()

N = len(T)
print(f"Dataset size: {N:,} points\n")

def summarise(name, arr, transforms=None):
    print(f"--- {name} ---")
    print(f"  raw range : [{arr.min():.4g}, {arr.max():.4g}]")
    pcts = np.percentile(arr, [5, 25, 50, 75, 95])
    print(f"  percentiles (5/25/50/75/95): {pcts}")
    if transforms:
        for label, fn in transforms.items():
            t = fn(arr)
            print(f"  {label}: mean={t.mean():.4g}, std={t.std():.4g}  "
                  f"  range=[{t.min():.4g}, {t.max():.4g}]")
    print()

summarise("T (keV)",  T,     {"log(T)":      lambda a: np.log(a + 1e-12)})
summarise("alpha",    alpha, {"log1p(alpha)": lambda a: np.log1p(a)})
summarise("x",        x,     {"log(x)":       lambda a: np.log(a + 1e-8)})

# T concentration: how many points fall in each decade
print("--- T decade breakdown ---")
edges = [0, 0.001, 0.01, 0.1, 1.0, 10.0, np.inf]
labels = ["<0.001", "0.001-0.01", "0.01-0.1", "0.1-1", "1-10", ">10"]
for lo, hi, lbl in zip(edges[:-1], edges[1:], labels):
    n = ((T >= lo) & (T < hi)).sum()
    print(f"  T in [{lbl:>12}] keV : {n:>8,}  ({100*n/N:.1f}%)")

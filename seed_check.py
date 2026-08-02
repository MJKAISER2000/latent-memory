"""Seed robustness check for the decisive L0 comparison.

The single-seed L0 run showed `pinned` beating `dissipative` by ~29x relative
ledger error at t=1024. One seed is not evidence. This repeats the decisive
pair (plus `conservative`, which is cheap) across seeds.

`free` is excluded: it took 921s vs ~85s for the others on seed 0 (Pade
scaling-and-squaring struggles as ||W|| grows during training), so a seed sweep
over it costs an hour. Run it separately if the free-vs-rest ordering matters.
"""

import numpy as np
import torch

import toy_ledger as T

SEEDS = [0, 1, 2]
MODES = ["dissipative", "conservative", "pinned"]
HORIZONS = [64, 128, 256, 512, 1024]

acc = {m: [] for m in MODES}
for seed in SEEDS:
    for mode in MODES:
        _, rel, _, _, dt = T.train_one(mode, seed=seed)
        acc[mode].append([rel[h].item() for h in HORIZONS])
        print(f"seed={seed} {mode:>13}  "
              + "  ".join(f"{rel[h].item():.4f}" for h in HORIZONS)
              + f"   ({dt:.0f}s)", flush=True)

print("\n" + "=" * 78)
print(f"MEAN +/- STD over seeds {SEEDS}   (trained only to t=64)")
print("=" * 78)
print(f"{'mode':>13} | " + " ".join(f"{('t=' + str(h)):>13}" for h in HORIZONS))
print("-" * 78)
for m in MODES:
    a = np.array(acc[m])
    cells = " ".join(f"{a[:, i].mean():>6.4f}+-{a[:, i].std():<6.4f}"
                     for i in range(len(HORIZONS)))
    print(f"{m:>13} | {cells}")
print("-" * 78)

p = np.array(acc["pinned"])[:, -1]
d = np.array(acc["dissipative"])[:, -1]
c = np.array(acc["conservative"])[:, -1]
print(f"\nt=1024 advantage of pinned:")
print(f"  vs dissipative : {d.mean() / p.mean():>6.1f}x")
print(f"  vs conservative: {c.mean() / p.mean():>6.1f}x")
print(f"  per-seed pinned-wins: dissipative {int((p < d).sum())}/{len(SEEDS)}, "
      f"conservative {int((p < c).sum())}/{len(SEEDS)}")

# The quantitative prediction: absolute ledger error random-walks (~sqrt t)
# while M_t grows linearly, so RELATIVE error should fall as 1/sqrt(t).
r = np.array(acc["pinned"])
print(f"\npinned rel_err(1024)/rel_err(64) = {(r[:, -1] / r[:, 0]).mean():.3f} "
      f"+- {(r[:, -1] / r[:, 0]).std():.3f}   (1/sqrt(16) = 0.25 predicted)")

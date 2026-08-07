"""
HARDENING ITEM 3: does the attractor track the training horizon?

The mechanism story predicts the learned decay converges to the interior
optimum of the K_train-horizon objective. If that optimum is set by the
training horizon, d*(K_train) * K_train should be roughly constant; if d* is
instead pinned by something else (init scale, task constants), it will not
move with K_train. This is the single cheapest test separating "horizon-myopic
attractor" from "init artifact".

Sweep: scalar arm on the frontier task, K_train in {32, 64, 128, 256},
K_test = 16 * K_train, seeds 0-2. Reports learned d* = softplus(s), the
product d* * K_train, and ledger skill at the 16x horizon.

Uses l0_frontier verbatim with module-level horizon overrides (the repo's
established driver pattern); nothing in l0_frontier is edited.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as TF

import l0_frontier as L

KS = [32, 64, 128, 256]
SEEDS = [0, 1, 2]


def main():
    print("d* vs 1/K_train scaling  (scalar arm; K_test = 16*K_train)\n")
    rows = {}
    for K in KS:
        L.K_TRAIN, L.K_TEST = K, 16 * K
        L.HORIZONS = [16 * K]
        ds, sks = [], []
        for seed in SEEDS:
            model, dt = L.train_one("scalar", seed)
            e = L.evaluate(model, seed)
            d = TF.softplus(model.gen.s).item()
            ds.append(d)
            sks.append(e["ledger"][L.K_TEST]["skill"])
            print(f"  K={K:>4} s{seed}: d*={d:.3e}  d*K={d*K:.4f}  "
                  f"skill@16K={sks[-1]:+.3f}  ({dt:.0f}s)", flush=True)
        rows[K] = (ds, sks)

    print(f"\n{'K_train':>8} {'d* med':>11} {'d*·K_train med':>15} "
          f"{'skill@16K med':>14}")
    prods = []
    for K in KS:
        ds, sks = rows[K]
        dm = float(np.median(ds))
        prods.append(dm * K)
        print(f"{K:>8} {dm:>11.3e} {dm*K:>15.4f} {float(np.median(sks)):>+14.3f}")

    spread = max(prods) / max(min(prods), 1e-12)
    print(f"\nd*·K_train spread across a {KS[-1]//KS[0]}x horizon range: "
          f"{spread:.2f}x")
    print("READING: if d* tracked ONLY the init (1e-3), d*·K would grow 8x "
          "across this range; if d* is horizon-set, d*·K stays ~flat. "
          "Spread <~2x supports the horizon-myopic attractor; >~4x argues "
          "d* is init- or task-anchored, and the mechanism wording must weaken.")


if __name__ == "__main__":
    main()

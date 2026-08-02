"""
L0-HARD: the ledger experiment on the NON-DEGENERATE task.

The original L0 used a deposit rate shared by every sample, so M_t concentrated
(SD/E ~ 1/sqrt(t)) and a constant predictor emitting 0.075t scored 7% relative
error at t=1024 without reading the input. That made effect sizes
uninterpretable: "pinned is 12.7x better than dissipative" mostly said that
dissipative had degraded to worse-than-trivial.

RATE_JITTER draws a per-sample rate p_i ~ U(0.05, 0.25). Measured spread:
    SD/E ~ 0.40 at EVERY horizon (vs 0.088 at t=1024 before)
    constant predictor -> 0.47 relative error at t=1024 (vs 0.071 before)
So any model beating ~0.47 is genuinely tracking the sample-specific total.

REPORTED, for every mode and horizon:
  rel    relative error (comparable to the old numbers, but on the hard task)
  corr   corr(pred, M_t) across samples -- the honest test of input-dependence
  skill  1 - MSE_model/MSE_const -- beat the oracle constant predictor?

KILL CRITERION
  If no mode achieves corr > 0.7 and skill > 0 beyond the training horizon, the
  generator structure is not buying long-horizon accumulation and the memory
  branch should be reconsidered.
"""

from __future__ import annotations

import numpy as np
import torch

import toy_ledger as T
from trivial_baseline import diagnose, HORIZONS

T.RATE_JITTER = True          # <-- the whole point

SEEDS = [0, 1, 2]
MODES = ["dissipative", "conservative", "pinned"]


def main():
    print("L0-HARD  (per-sample deposit rate; constant predictor ~0.47 rel err @ t=1024)")
    print(f"seeds={SEEDS}  modes={MODES}\n")

    res = {m: [] for m in MODES}
    for seed in SEEDS:
        for mode in MODES:
            model, _, _, _, dt = T.train_one(mode, seed=seed)
            g = torch.Generator(device=T.DEV).manual_seed(seed + 1)
            v = torch.randn(T.D_IN, generator=g, device=T.DEV)
            d = diagnose(model, g, v, T.K_TEST)
            res[mode].append(d)
            print(f"  seed={seed} {mode:>13}  "
                  + "  ".join(f"r={d[t]['rel']:.3f}/c={d[t]['corr']:+.2f}"
                              for t in HORIZONS) + f"  ({dt:.0f}s)", flush=True)

    hdr = " ".join(f"{('t=' + str(t)):>9}" for t in HORIZONS)
    for key, title, note in [
        ("corr", "CORRELATION corr(pred, M_t)", "the honest test: is it input-dependent?"),
        ("skill", "SKILL vs oracle constant predictor", "<=0 means no better than the population mean"),
        ("skill_cal", "SKILL AFTER ORACLE AFFINE RECALIBRATION",
         "the ceiling calibration could reach; gap to `skill` is pure gain drift"),
        ("gain", "fitted gain a in (a*pred + b)", "1.0 = self-calibrating; else the scale drifted"),
        ("rel", "relative error", "compare against the CONSTANT row"),
    ]:
        print("\n" + "=" * 76)
        print(f"{title}   -- {note}")
        print("=" * 76)
        print(f"{'mode':>13} | {hdr}")
        print("-" * 76)
        for m in MODES:
            a = np.array([[d[t][key] for t in HORIZONS] for d in res[m]])
            print(f"{m:>13} | " + " ".join(f"{v:9.3f}" for v in a.mean(axis=0)))
            print(f"{'  (std)':>13} | " + " ".join(f"{v:9.3f}" for v in a.std(axis=0)))
        if key == "rel":
            c = np.array([[d[t]["rel_const"] for t in HORIZONS] for d in res[MODES[0]]])
            print(f"{'CONSTANT':>13} | " + " ".join(f"{v:9.3f}" for v in c.mean(axis=0)))
        print("-" * 76)

    print("\nVERDICT @ t=1024")
    for m in MODES:
        c = np.mean([d[1024]["corr"] for d in res[m]])
        s = np.mean([d[1024]["skill"] for d in res[m]])
        tag = ("TRACKS THE SAMPLE" if c > 0.7 and s > 0 else
               "no better than constant" if s <= 0 else "partial")
        print(f"  {m:>13}: corr={c:+.3f}  skill={s:+.3f}  -> {tag}")


if __name__ == "__main__":
    main()

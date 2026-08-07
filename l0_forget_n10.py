"""
HARDENING ITEM 1: l0_forget Phase B at n=10 with the full arm set.

The committed Phase B (results/L0_forget.txt) is n=3 over three arms - below
the two-sided sign-test floor. This driver runs 5 arms x 10 seeds on the SAME
validated task (Phase A necessity already established, not re-run):

    scalar   learned uniform decay        (expected: attractor, fails both)
    scalar0  frozen-zero scalar           (expected: ledger unreliable - lottery)
    diag0    frozen-zero per-dim decay    (new arm: can rotation still steal the
                                           ledger when only SOME dims must clear?)
    ks       full K+S, learned scalar     (new arm: soft version of pinned)
    pinned   hard C^T W = 0               (expected: ledger 10/10, content viable)

GUARD: before the sweep, re-runs scalar seed 0 and asserts it reproduces the
committed log (skill -0.960, con_late 23.0019 at printed precision) - proving
the arm-extension edit did not perturb the established arms' RNG streams.

Pre-registered tests (committed with this file before any n=10 data existed):
    pinned vs each other arm, paired by seed, exact two-sided sign test,
    separately on ledger skill@1024 and content_late.
"""

from __future__ import annotations

import numpy as np

import l0_forget as F
from l0_stats import sign_test

ARMS = ["scalar", "scalar0", "diag0", "ks", "pinned"]
SEEDS = list(range(10))


def main():
    print("GUARD: reproduce committed scalar s0 before sweeping...", flush=True)
    m, _ = F.train_one("scalar", 0)
    e = F.evaluate(m, 0)
    got = (round(e["skill_1024"], 3), round(e["con_late"], 4))
    exp = (-0.960, 23.0019)
    print(f"  got skill={got[0]:+.3f} con_late={got[1]:.4f}  "
          f"expected {exp[0]:+.3f}/{exp[1]:.4f}")
    if got != exp:
        print("  GUARD FAILED - arm extension perturbed established arms; ABORT")
        return
    print("  guard passed (bit-exact at printed precision)\n")

    res = {a: [] for a in ARMS}
    for seed in SEEDS:
        for a in ARMS:
            model, dt = F.train_one(a, seed)
            e = F.evaluate(model, seed)
            res[a].append(e)
            print(f"  s{seed} {a:>8}: skill@1024={e['skill_1024']:+.3f} "
                  f"con_late={e['con_late']:.4f} decay_min={e['decay_min']:.2e} "
                  f"({dt:.0f}s)", flush=True)

    def col(a, k):
        return np.array([r[k] for r in res[a]])

    print(f"\n{'arm':>8} | {'ledger skill@1024 med [IQR]':>28} | "
          f"{'content late med [IQR]':>24} | {'ledger holds (skill>0)':>22}")
    print("-" * 92)
    for a in ARMS:
        sk, cl = col(a, "skill_1024"), col(a, "con_late")
        holds = int((sk > 0).sum())
        print(f"{a:>8} | {np.median(sk):+8.3f} [{np.percentile(sk,25):+.3f},"
              f"{np.percentile(sk,75):+.3f}] | {np.median(cl):8.3f} "
              f"[{np.percentile(cl,25):.3f},{np.percentile(cl,75):.3f}] | "
              f"{holds:>10}/10")
    print("-" * 92)

    print("\nPAIRED TESTS vs pinned (pre-registered; two-sided exact sign test)")
    for b in [a for a in ARMS if a != "pinned"]:
        for metric, better_high in [("skill_1024", True), ("con_late", False)]:
            dp, db = col("pinned", metric), col(b, metric)
            wins = int(((dp > db) if better_high else (dp < db)).sum())
            ties = int((dp == db).sum())
            p = sign_test(wins, len(dp) - ties) if ties < len(dp) else 1.0
            print(f"  pinned vs {b:>8} on {metric:>10}: wins {wins}/10 "
                  f"(ties {ties}), sign p={p:.4g}")


if __name__ == "__main__":
    try:
        from capture_env import banner
        banner("l0_forget_n10")
    except ImportError:
        pass
    main()

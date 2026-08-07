"""
HARDENING ITEM 2: statistics for the two under-powered rescue verdicts.

The committed rescue suite (results/L0_mechanism.txt Part B) supported
"3x budget does not rescue" on n=2 (one seed partially rescued) and
"regularization fails" on a single lambda (3.0) at n=2. This runs:

    long   (3x training budget)      seeds 0-4   (n=5)
    reg    lambda in {0.3, 3, 30}    seeds 0-2   (n=3 per lambda)

on the l0_frontier task, using the exact committed train_variant recipe.
Verdict wording in the docs is upgraded/downgraded strictly from these medians.
"""

from __future__ import annotations

import numpy as np

import l0_mechanism as M
import l0_frontier as L


def show(tag, evals):
    sk = np.array([e["ledger"][L.K_TEST]["skill"] for e in evals])
    cl = np.array([e["content_late"] for e in evals])
    print(f"  {tag:>14}: skill med {np.median(sk):+.3f} "
          f"[{sk.min():+.3f},{sk.max():+.3f}]  content med {np.median(cl):.3f}  "
          f"rescued(skill>0): {int((sk>0).sum())}/{len(sk)}")
    return sk


def main():
    print("LONG ARM (3x budget), seeds 0-4")
    longs = []
    for seed in range(5):
        e = M.train_variant("diag", seed, "long")
        longs.append(e)
        print(f"    s{seed}: skill={e['ledger'][L.K_TEST]['skill']:+.3f} "
              f"con_late={e['content_late']:.3f} dmin={e['decay_min']:.2e}",
              flush=True)
    sk_long = show("long n=5", longs)

    print("\nREG ARM lambda sweep, seeds 0-2")
    results = {}
    for lam in [0.3, 3.0, 30.0]:
        M.LAM_REG = lam
        evs = []
        for seed in range(3):
            e = M.train_variant("diag", seed, "reg")
            evs.append(e)
            print(f"    lam={lam:<4g} s{seed}: "
                  f"skill={e['ledger'][L.K_TEST]['skill']:+.3f} "
                  f"con_late={e['content_late']:.3f} dmin={e['decay_min']:.2e}",
                  flush=True)
        results[lam] = show(f"reg lam={lam:g}", evs)

    print("\nVERDICT INPUTS")
    print(f"  long: median {np.median(sk_long):+.3f}; "
          f"rescued {int((sk_long>0).sum())}/5 -> "
          f"{'budget alone CAN rescue sometimes - soften the verdict' if (sk_long>0).sum() >= 2 else 'budget does not reliably rescue - verdict stands'}")
    any_reg = max(np.median(v) for v in results.values())
    print(f"  reg: best median over lambdas {any_reg:+.3f} -> "
          f"{'some lambda rescues - REVISE soft-fixes-fail' if any_reg > 0.5 else 'no lambda rescues at these settings - verdict stands (scoped to this sweep)'}")


if __name__ == "__main__":
    try:
        from capture_env import banner
        banner("rescue_stats")
    except ImportError:
        pass
    main()

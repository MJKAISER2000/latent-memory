"""
n=10 STATISTICS RUN -- the package's formal backbone.

The audit's standing objection: n=3 seeds has a two-sided sign-test floor of
p=0.125, so nothing at n=3 is formally significant. This runs the frontier
comparison at n=10 across the five arms that matter for the final claim:

    scalar      learned uniform decay (practitioner init 1e-3)   [fails?]
    diag        learned per-dim decay (same init)                [fails?]
    ks          soft K+S (same init)                             [partial?]
    diag0       learned per-dim decay, init/frozen at ~0         [fix #2]
    pinned      hard C^T W = 0                                   [fix #1]

Reported per arm: median [IQR] for ledger skill @1024 and content-late, plus
exact two-sided sign tests for the pre-registered comparisons:
    pinned  vs scalar,  pinned vs diag,  pinned vs ks   (superiority)
    pinned  vs diag0                                     (parity of the fixes)
10/10 wins -> p = 2*(1/2)^10 = 0.00195. Wilcoxon added when scipy is present.

Runtime: 5 arms x 10 seeds x ~2.5 min ~ 2h on the 3060. Run AFTER the rescue
suite finishes -- do not share the GPU.
"""

from __future__ import annotations

import numpy as np
import torch

import l0_frontier as L
from l0_mechanism import train_variant

SEEDS = list(range(10))
ARMS = ["scalar", "diag", "ks", "diag0", "pinned"]


def run_arm(arm: str, seed: int):
    if arm == "diag0":
        return train_variant("diag", seed, "init0")
    model, _ = L.train_one(arm, seed)
    return L.evaluate(model, seed)


def sign_test(wins: int, n: int) -> float:
    """Exact two-sided sign test p-value for `wins` out of n (no ties)."""
    from math import comb
    tail = sum(comb(n, k) for k in range(min(wins, n - wins) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def main():
    print(f"n=10 stats run | arms={ARMS}\n", flush=True)
    res = {a: [] for a in ARMS}
    for seed in SEEDS:
        for arm in ARMS:
            e = run_arm(arm, seed)
            res[arm].append(e)
            print(f"  s{seed} {arm:>7}: skill={e['ledger'][L.K_TEST]['skill']:+.3f} "
                  f"late={e['content_late']:.3f} dmin={e['decay_min']:.2e}",
                  flush=True)

    def col(arm, key):
        if key == "skill":
            return np.array([r["ledger"][L.K_TEST]["skill"] for r in res[arm]])
        return np.array([r[key] for r in res[arm]])

    print("\n" + "=" * 78)
    print(f"MEDIAN [IQR] over {len(SEEDS)} seeds")
    print("=" * 78)
    print(f"{'arm':>8} | {'ledger skill @1024':>24} | {'content late':>22}")
    print("-" * 78)
    for a in ARMS:
        s, c = col(a, "skill"), col(a, "content_late")
        print(f"{a:>8} | {np.median(s):+8.3f} [{np.percentile(s,25):+.3f},"
              f"{np.percentile(s,75):+.3f}] | {np.median(c):8.3f} "
              f"[{np.percentile(c,25):.3f},{np.percentile(c,75):.3f}]")
    print("-" * 78)

    print("\nPAIRED TESTS (pre-registered)")
    try:
        from scipy.stats import wilcoxon
        have_scipy = True
    except ImportError:
        have_scipy = False
    for a, b, what in [("pinned", "scalar", "superiority"),
                       ("pinned", "diag", "superiority"),
                       ("pinned", "ks", "superiority"),
                       ("pinned", "diag0", "fix-parity")]:
        d = col(a, "skill") - col(b, "skill")
        wins = int((d > 0).sum())
        p = sign_test(wins, len(d))
        line = (f"  {a} vs {b:>7} ({what:>11}): wins {wins}/{len(d)}, "
                f"sign-test p={p:.4g}, median diff={np.median(d):+.3f}")
        if have_scipy and 0 < wins < len(d):
            try:
                line += f", wilcoxon p={wilcoxon(d).pvalue:.4g}"
            except ValueError:
                pass
        print(line)
    print("\n(fix-parity: a LOW win count / high p means the two fixes are "
          "statistically indistinguishable on raw skill -- which is the "
          "expected result; pinned's edge is exactness+auditability, not skill)")


if __name__ == "__main__":
    main()

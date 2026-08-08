"""
L1 DATA: haystack `count` corpora sized for the real-LM ledger experiment.

Three splits (different haystack seeds -> disjoint entities/fillers):
    l1_train      300 streams x ~4k tokens  (~60 blocks of 64)   seed 0
    l1_evalshort   60 streams x ~4k tokens  (train-horizon eval) seed 1
    l1_testlong    60 streams x ~64k tokens (~16x horizon)       seed 2

Deposit RATE is drawn per stream (deposits-per-block ~ U(0.05, 0.2)), mirroring
the toy's per-sample rate jitter, so the running total cannot be predicted from
stream length alone. The per-deposit amounts live in the text ("A shipment of
N units...") and are re-parsed downstream from the durable spans -- no schema
changes to data/haystack.py.

Uses make_sample directly so n_deposits can vary per stream.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict
from pathlib import Path

from data.haystack import HaystackConfig, make_sample

SPLITS = [
    # name        seed  n    n_filler  (≈15 tokens per filler sentence)
    ("l1_train",     0, 300,  260),
    ("l1_evalshort", 1,  60,  260),
    ("l1_testlong",  2,  60, 4300),
]
RATE_LO, RATE_HI = 0.05, 0.20          # deposits per 64-token block
TOKENS_PER_SENT = 15.5                 # measured average, for block estimates


def main():
    out_dir = Path("data")
    for name, seed, n, n_filler in SPLITS:
        rng = random.Random(seed)
        blocks_est = n_filler * TOKENS_PER_SENT / 64
        path = out_dir / f"{name}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for _ in range(n):
                rate = rng.uniform(RATE_LO, RATE_HI)
                n_dep = max(2, round(rate * blocks_est))
                cfg = HaystackConfig(task="count", n_filler=n_filler,
                                     n_deposits=n_dep, depths=(0.02,),
                                     seed=seed)
                s = make_sample(cfg, rng)
                f.write(json.dumps(asdict(s)) + "\n")
        print(f"{name}: {n} streams, ~{blocks_est:.0f} blocks each, "
              f"deposits/stream 2..{max(2, round(RATE_HI * blocks_est))} "
              f"-> {path}")


if __name__ == "__main__":
    main()

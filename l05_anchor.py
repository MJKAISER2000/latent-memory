"""
L0.5 -- ANCHOR TRANSPORT.

The sharpened version of the paper's Fig. 10 pullback ablation, and the one
claim in this project that is ours rather than borrowed.

THE CLAIM
  Supervise the ledger at ONE horizon (t = 64). Test at t = 8 ... 1024.
  The pinned model should be correct at every other horizon for free.

WHY IT SHOULD WORK
  The balance law C^T z_t = C^T z_0 + sum_{k<t} C^T g_k holds exactly, by
  construction. Supervising only at t=64 imposes a single scalar equation per
  sample:  sum_{k<64} C^T g_k = M_64.  That alone does NOT pin the per-block
  flux -- C^T g_k could be constant and still satisfy it.

  What pins it is that the encoder is SHARED across positions and the deposits
  a_k vary independently across positions and samples. One sum constraint,
  replicated over many random samples, identifies the per-block function
  C^T g(x) = a(x). Exact conservation then transports that to every horizon.

  The baselines have no balance law. Their readout at t=64 can be satisfied by
  whatever combination of decay/rotation happens to work after exactly 64
  steps, with nothing constraining any other t. So they should fail at
  unsupervised horizons in BOTH directions -- interior (t < 64) as well as
  exterior (t > 64). The interior failure is the cleaner signature, since it
  cannot be explained away as an extrapolation artifact.

IF THIS HOLDS
  Structure converts sparse supervision into dense supervision. That matters
  well beyond the toy: you rarely have per-token labels for what a memory
  should be retaining.

FALSIFIED IF
  pinned-sparse is no better than dissipative-sparse at interior horizons, or
  pinned-sparse collapses relative to pinned-dense.
"""

from __future__ import annotations

import numpy as np
import torch

import toy_ledger as T

SEEDS = [0, 1, 2]
MODES = ["dissipative", "conservative", "pinned"]
SUPS = ["dense", "sparse"]
# 64 is the anchor. Everything else is unsupervised under `sparse`.
HORIZONS = [8, 16, 32, 64, 128, 256, 512, 1024]


@torch.no_grad()
def evaluate_full(model, gen, v, K: int, B: int = 64):
    """Absolute AND relative ledger error. Absolute is the honest metric at
    small t, where M_t ~ 0 makes the relative version explode."""
    model.eval()
    x, a, M, content = T.make_batch(B, K, gen, v)
    zs, _ = model.rollout(x)
    pred = model.ledger(zs.reshape(-1, T.N_Z)).reshape(B, K + 1, -1)[..., 0]
    abs_err = (pred - M).abs().mean(dim=0)
    rel_err = ((pred - M).abs() / M.clamp(min=1e-3)).mean(dim=0)
    model.train()
    return abs_err.cpu().numpy(), rel_err.cpu().numpy(), M.mean(dim=0).cpu().numpy()


def main():
    print(f"L0.5 anchor transport | anchor at t={T.K_TRAIN} | seeds={SEEDS}")
    print("under `sparse`, ONLY t=64 is supervised; all other columns are free\n")

    res = {(m, s): [] for m in MODES for s in SUPS}
    scale = None

    for seed in SEEDS:
        for mode in MODES:
            for sup in SUPS:
                model, _, _, _, dt = T.train_one(mode, seed=seed, supervision=sup)
                g = torch.Generator(device=T.DEV).manual_seed(seed + 1)
                v = torch.randn(T.D_IN, generator=g, device=T.DEV)
                ae, re, Mm = evaluate_full(model, g, v, T.K_TEST)
                res[(mode, sup)].append([ae[h] for h in HORIZONS])
                if scale is None:
                    scale = [Mm[h] for h in HORIZONS]
                print(f"  seed={seed} {mode:>12} {sup:>6}  "
                      + " ".join(f"{ae[h]:7.3f}" for h in HORIZONS)
                      + f"  ({dt:.0f}s)", flush=True)

    hdr = " ".join(f"{('t='+str(h)):>9}" for h in HORIZONS)
    print("\n" + "=" * 104)
    print("MEAN ABSOLUTE LEDGER ERROR  (lower is better; ^ marks the supervised anchor)")
    print("=" * 104)
    print(f"{'mode':>13} {'sup':>7} | {hdr}")
    print(f"{'true M_t':>13} {'':>7} | " + " ".join(f"{m:9.2f}" for m in scale))
    print("-" * 104)
    for mode in MODES:
        for sup in SUPS:
            a = np.array(res[(mode, sup)]).mean(axis=0)
            mark = "".join("^" if h == T.K_TRAIN else "" for h in HORIZONS)
            print(f"{mode:>13} {sup:>7} | " + " ".join(f"{x:9.3f}" for x in a)
                  + (f"   {mark}" if sup == "sparse" else ""))
        print("-" * 104)

    print("\n" + "=" * 70)
    print("ANCHOR TRANSPORT RATIO  = sparse_error / dense_error")
    print("1.0 => one supervised horizon was as good as supervising all of them")
    print("=" * 70)
    print(f"{'mode':>13} | {hdr}")
    print("-" * 70)
    for mode in MODES:
        d = np.array(res[(mode, "dense")]).mean(axis=0)
        s = np.array(res[(mode, "sparse")]).mean(axis=0)
        print(f"{mode:>13} | " + " ".join(f"{x:9.2f}" for x in s / np.maximum(d, 1e-9)))
    print("-" * 70)

    # The decisive cut: interior horizons under sparse supervision. These are
    # unsupervised but require no extrapolation, so a failure there is purely
    # about the absence of a transport mechanism.
    interior = [i for i, h in enumerate(HORIZONS) if h < T.K_TRAIN]
    print("\ninterior (t<64, unsupervised, no extrapolation) mean abs error, sparse:")
    for mode in MODES:
        s = np.array(res[(mode, "sparse")]).mean(axis=0)[interior]
        print(f"  {mode:>13}: {s.mean():.4f}")


if __name__ == "__main__":
    main()

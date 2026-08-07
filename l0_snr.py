"""
ADAM-CORRECT MECHANISM PROBE: gradient signal-to-noise, not magnitude.

Adam normalises by the second moment, so a small-but-consistent gradient still
moves a parameter at ~lr per step. "The gradient is 237x smaller" therefore
does NOT explain why the decay parameter parked at its init. The right
quantity is the SIGNAL-TO-NOISE RATIO across minibatches:

    SNR(d) = |E_batch[ dL/ds ]| / SD_batch[ dL/ds ]

  SNR >> 1  ->  consistently signed gradient  ->  Adam descends at ~lr
  SNR << 1  ->  sign flips batch to batch     ->  Adam random-walks; the
                parameter stays where the init put it

PREDICTION (the Adam-proof version of F1's mechanism)
  SNR crosses ~1 near d ~ 1/K_train: super-critical decay rates are seen and
  corrected; sub-critical rates are invisible *to a stochastic optimiser* even
  though L_test depends on them enormously.

Also reported: the IMPLIED DRIFT. Over N steps with lr and SNR, expected
displacement of s is ~ lr*N*SNR/sqrt(1+SNR^2) (signal) vs lr*sqrt(N) (noise
walk). We print how many steps Adam would need to move the decay from init
1e-3 down to 1e-5 given the measured SNR -- if that exceeds the training
budget by orders of magnitude, the parking is explained quantitatively.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

import l0_frontier as L

DEV = L.DEV
GRID = np.logspace(-5, -0.5, 10)
N_BATCH = 64                      # minibatches per grid point
TARGET_FROM, TARGET_TO = 1e-3, 1e-5


def grad_samples(model, d: float, gen, n=N_BATCH):
    """Per-minibatch gradients dL_train/ds at pinned decay value d."""
    raw = L.softplus_inv(float(d))
    gs = []
    for _ in range(n):
        with torch.no_grad():
            model.gen.s.fill_(raw)
        model.gen.s.requires_grad_(True)
        if model.gen.s.grad is not None:
            model.gen.s.grad = None
        x, M, content = L.make_batch(L.BATCH, L.K_TRAIN, gen)
        zs = model.rollout(x)
        loss = F.mse_loss(model.ledger(zs), M)
        recon = model.dec(zs[:, 1:].reshape(-1, L.N_Z)).reshape(
            L.BATCH, L.K_TRAIN, L.D_IN)
        loss = loss + F.mse_loss(recon, content)   # SAME total loss as training
        loss.backward()
        gs.append(model.gen.s.grad.item())
    return np.array(gs)


def main(seed: int = 0):
    print("Training a scalar-mode model to convergence first (same recipe)...")
    model, dt = L.train_one("scalar", seed)
    learned = F.softplus(model.gen.s.detach()).item()
    print(f"  done ({dt:.0f}s). SGD/Adam converged to d = {learned:.3e}\n")

    gen = torch.Generator(device=DEV).manual_seed(seed + 999)
    print("=" * 74)
    print("GRADIENT SNR ACROSS MINIBATCHES  (total loss, exactly as trained)")
    print(f"critical scale 1/K_train = {1/L.K_TRAIN:.4f}")
    print("=" * 74)
    print(f"{'decay d':>10} {'d*K_train':>10} {'mean grad':>12} {'sd grad':>11} "
          f"{'SNR':>8}  regime")
    print("-" * 74)
    rows = []
    for d in GRID:
        g = grad_samples(model, d, gen)
        mu, sd = g.mean(), g.std()
        snr = abs(mu) / max(sd, 1e-12)
        regime = "VISIBLE to Adam" if snr > 1 else "noise-dominated"
        rows.append((d, mu, sd, snr))
        print(f"{d:>10.2e} {d*L.K_TRAIN:>10.4f} {mu:>12.4e} {sd:>11.4e} "
              f"{snr:>8.3f}  {regime}", flush=True)
    print("-" * 74)

    # implied steps to walk s from softplus_inv(1e-3) to softplus_inv(1e-5)
    dist = abs(L.softplus_inv(TARGET_FROM) - L.softplus_inv(TARGET_TO))
    sub = [r for r in rows if r[0] * L.K_TRAIN < 0.1]
    if sub:
        snr_sub = np.mean([r[3] for r in sub])
        # Adam step ~ lr * mu/sqrt(mu^2+sd^2) = lr * SNR/sqrt(1+SNR^2) toward signal
        drift = L.LR * snr_sub / math.sqrt(1 + snr_sub ** 2)
        n_signal = dist / max(drift, 1e-12)
        print(f"""
IMPLIED DRIFT BUDGET
  distance in s-space, d=1e-3 -> 1e-5 : {dist:.2f}
  mean sub-critical SNR               : {snr_sub:.3f}
  Adam signal drift per step          : ~{drift:.2e}
  steps needed by signal alone        : ~{n_signal:,.0f}
  actual training budget              : {L.STEPS:,} steps
  shortfall                           : {n_signal / L.STEPS:,.0f}x
""")
    print("If SNR < 1 sub-critically and the shortfall is large, the parking of")
    print("learned decay is quantitatively explained UNDER ADAM -- the flat-basin")
    print("mechanism survives the optimizer-normalisation objection.")


if __name__ == "__main__":
    try:
        from capture_env import banner
        banner("l0_snr")
    except ImportError:
        pass
    main()

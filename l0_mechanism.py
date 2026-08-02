"""
MECHANISM PROBE + BASELINE RESCUE.

F1 claims: decay rates below ~1/K_train sit in a FLAT BASIN of the training
loss, so SGD has no gradient signal to drive them to zero, and the residual
leak is fatal at test horizon. So far that is an inference from where the
decay parameters happened to land. This script tests the mechanism directly
and then tries hard to falsify F1.

--------------------------------------------------------------------------
PART A -- LANDSCAPE SCAN (direct evidence, no training involved)
  Take a trained model. Freeze everything. Sweep the decay parameter d over
  a log grid and measure:
     L_train(d)   ledger loss at K_train = 64
     L_test(d)    ledger loss at K_test  = 1024
     |dL_train/d(log d)|   the gradient SGD actually sees

  F1 predicts a REGIME SEPARATION:
     for d << 1/K_train : L_train flat and gradient ~ 0, but L_test already rising
     for d >> 1/K_train : both rise, gradient large
  If instead the training gradient is large throughout, F1's mechanism is
  wrong and the baselines failed for some other reason.

--------------------------------------------------------------------------
PART B -- BASELINE RESCUE (adversarial, against our own finding)
  The audit's standing demand: try to fix the baselines BEFORE claiming the
  constraint is needed. Variants applied to `diag` (the mode that CAN
  represent selective protection but did not find it):

    baseline   as in l0_frontier (init 1e-3)
    init0      initialise every decay at ~0. If this alone fixes it, F1
               collapses from "structure is needed" to "initialisation
               matters" -- THE decisive rescue.
    reg        add lambda * mean(softplus(d)) to the loss: explicit pressure
               toward zero decay. Tests whether the flat basin is escapable
               with an auxiliary gradient.
    long       3x training steps. Tests whether it is merely slow, not flat.

  For `scalar` we also run init0, but expect a DIFFERENT failure: scalar has
  one knob, so zero decay protects the ledger and destroys content. That is
  an expressivity limit, not an optimisation one -- and keeping the two
  claims separate is the point.

REPORTED: ledger skill @1024 AND content MSE late. A rescue only counts if it
wins BOTH, because winning one by sacrificing the other is what the
unconstrained modes already do.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

import l0_frontier as L

DEV = L.DEV
GRID = np.logspace(-6, -0.3, 22)     # decay rates to scan
LAM_REG = 3.0                        # weight for the explicit decay penalty


# ---------------------------------------------------------------- PART A
@torch.no_grad()
def _ledger_loss(model, x, M, K):
    zs = model.rollout(x[:, :K])
    return F.mse_loss(model.ledger(zs), M[:, :K + 1]).item()


def landscape(seed: int = 0, B: int = 256):
    """Sweep the scalar decay of a trained model; measure both losses + grad."""
    model, _ = L.train_one("scalar", seed)
    gen = torch.Generator(device=DEV).manual_seed(seed + 555)
    x, M, _ = L.make_batch(B, L.K_TEST, gen)

    learned = F.softplus(model.gen.s).item()
    rows = []
    for d in GRID:
        raw = L.softplus_inv(float(d))
        with torch.no_grad():
            model.gen.s.fill_(raw)
        lt = _ledger_loss(model, x, M, L.K_TRAIN)
        lT = _ledger_loss(model, x, M, L.K_TEST)

        # gradient SGD actually sees at the TRAIN horizon, in log-d units
        model.gen.s.requires_grad_(True)
        if model.gen.s.grad is not None:
            model.gen.s.grad = None
        zs = model.rollout(x[:, :L.K_TRAIN])
        loss = F.mse_loss(model.ledger(zs), M[:, :L.K_TRAIN + 1])
        loss.backward()
        # d(softplus(s))/ds = sigmoid(s); chain to d/d(log d) = d * d/dd
        dsig = torch.sigmoid(model.gen.s).item()
        g_d = model.gen.s.grad.item() / max(dsig, 1e-12)      # dL/dd
        rows.append((d, lt, lT, abs(g_d * d)))                # |dL/dlog d|
        model.gen.s.grad = None

    print("\n" + "=" * 78)
    print("PART A -- LOSS LANDSCAPE IN THE DECAY PARAMETER")
    print(f"K_train={L.K_TRAIN}  =>  critical scale 1/K_train = {1/L.K_TRAIN:.4f}")
    print(f"SGD actually converged to d = {learned:.2e}")
    print("=" * 78)
    print(f"{'decay d':>10} {'d*K_train':>10} {'L_train':>11} {'L_test':>12} "
          f"{'|dL_train/dlog d|':>18}")
    print("-" * 78)
    base_tr = rows[0][1]
    for d, lt, lT, g in rows:
        mark = "  <-- 1/K_train" if abs(math.log10(d * L.K_TRAIN)) < 0.18 else ""
        print(f"{d:>10.2e} {d*L.K_TRAIN:>10.4f} {lt:>11.4f} {lT:>12.3f} "
              f"{g:>18.3e}{mark}")
    print("-" * 78)

    flat = [r for r in rows if r[0] * L.K_TRAIN < 0.1]
    steep = [r for r in rows if r[0] * L.K_TRAIN > 1.0]
    if flat and steep:
        ftr = max(r[1] for r in flat) - min(r[1] for r in flat)
        fte = max(r[2] for r in flat) - min(r[2] for r in flat)
        print(f"\nIn the sub-critical region (d*K_train < 0.1):")
        print(f"  L_train varies by {ftr:.5f}   <- what SGD sees")
        print(f"  L_test  varies by {fte:.3f}   <- what actually matters")
        print(f"  ratio = {fte/max(ftr,1e-9):,.0f}x")
        print(f"  mean |grad| sub-critical  = {np.mean([r[3] for r in flat]):.3e}")
        print(f"  mean |grad| super-critical= {np.mean([r[3] for r in steep]):.3e}")
    return rows


# ---------------------------------------------------------------- PART B
def train_variant(mode: str, seed: int, variant: str):
    """Rescue variants. Returns the same eval dict as l0_frontier."""
    torch.manual_seed(seed)
    gen = torch.Generator(device=DEV).manual_seed(seed + 1)
    model = L.Model(mode).to(DEV)

    if variant == "init0":
        with torch.no_grad():
            model.gen.s.fill_(-20.0)                 # softplus(-20) ~ 2e-9
            model.gen.d.fill_(-20.0)

    steps = L.STEPS * 3 if variant == "long" else L.STEPS
    opt = torch.optim.Adam(model.parameters(), lr=L.LR)
    sched = torch.optim.lr_scheduler.StepLR(opt, max(steps // 3, 1), gamma=0.5)

    for _ in range(steps):
        x, M, content = L.make_batch(L.BATCH, L.K_TRAIN, gen)
        zs = model.rollout(x)
        loss = F.mse_loss(model.ledger(zs), M)
        recon = model.dec(zs[:, 1:].reshape(-1, L.N_Z)).reshape(
            L.BATCH, L.K_TRAIN, L.D_IN)
        loss = loss + F.mse_loss(recon, content)
        if variant == "reg":
            pen = (F.softplus(model.gen.d).mean() if mode == "diag"
                   else F.softplus(model.gen.s))
            loss = loss + LAM_REG * pen
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
    return L.evaluate(model, seed)


def rescue(seeds=(0, 1)):
    plan = [("diag", "baseline"), ("diag", "init0"), ("diag", "reg"),
            ("diag", "long"), ("scalar", "init0"), ("pinned", "baseline")]
    print("\n" + "=" * 88)
    print("PART B -- BASELINE RESCUE  (a rescue counts only if it wins BOTH columns)")
    print("=" * 88)
    print(f"{'mode':>8} {'variant':>9} | {'ledger skill@1024':>18} | "
          f"{'content late':>14} | {'min decay':>11} | {'align':>6}")
    print("-" * 88)
    out = {}
    for mode, var in plan:
        sk, cl, dm, al = [], [], [], []
        for s in seeds:
            e = train_variant(mode, s, var)
            sk.append(e["ledger"][L.K_TEST]["skill"])
            cl.append(e["content_late"])
            dm.append(e["decay_min"])
            al.append(e["align"])
        out[(mode, var)] = (np.mean(sk), np.mean(cl), np.mean(dm), np.mean(al))
        print(f"{mode:>8} {var:>9} | {np.mean(sk):>+9.3f} +-{np.std(sk):<6.3f} | "
              f"{np.mean(cl):>8.3f} +-{np.std(cl):<4.2f} | {np.mean(dm):>11.2e} | "
              f"{np.mean(al):>6.2f}", flush=True)
    print("-" * 88)

    d_base, d_i0 = out[("diag", "baseline")], out[("diag", "init0")]
    p = out[("pinned", "baseline")]
    print(f"""
INTERPRETATION
  diag init0 vs diag baseline : ledger {d_base[0]:+.2f} -> {d_i0[0]:+.2f}
  diag init0 vs pinned        : ledger {d_i0[0]:+.2f} vs {p[0]:+.2f}, """
          f"""content {d_i0[1]:.3f} vs {p[1]:.3f}

  If diag-init0 matches pinned on BOTH columns, F1 weakens to "initialise the
  decay at zero" -- a one-line fix, not an argument for hard structure.
  If diag-init0 recovers the ledger but drifts its decay back up or loses
  content, the flat basin is real and the constraint is doing work SGD will
  not do on its own.""")
    return out


if __name__ == "__main__":
    landscape(seed=0)
    rescue(seeds=(0, 1))

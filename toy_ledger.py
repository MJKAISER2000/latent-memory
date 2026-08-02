"""
Level-0 experiment: does the structured generator actually buy length
generalisation for a running ledger?

TASK ("running ledger with distractors")
  A stream of blocks x_k in R^d. Each block carries
     - a sparse non-negative "deposit" a_k  (nonzero w.p. 0.15), and
     - distractor content that changes every block.
  Two supervised objectives:
     - LEDGER : predict the running total  M_t = sum_{k<t} a_k
     - CONTENT: reconstruct the current block's content from z_t
  The content task is essential -- without it the model can dedicate the whole
  latent to the ledger and the comparison is vacuous.

THE ACTUAL QUESTION
  Train on K_train = 64 blocks. Test on K_test = 1024 (16x the training
  horizon). Does ledger accuracy survive extrapolation?

  This is the memory analogue of the paper's Fig. 5/6: a model with the right
  conserved structure should degrade gracefully past the horizon it was
  trained on, while an unconstrained one drifts.

VARIANTS (identical enc/dec, identical losses, identical param budget)
  free         unconstrained W
  dissipative  W = -BB^T - eps I          (the Mamba/S4 corner)
  conservative W = A - A^T                (the DeltaNet corner)
  pinned       W = K + S inside C-perp    (the paper's mixed case)

HONEST EXPECTATION
  "pinned" should NOT have zero error. Each block's flux estimate C^T g_k has
  some error, and those errors random-walk, so ledger error grows ~sqrt(t).
  The claim under test is sqrt(t) accumulation vs. compounding drift.
"""

from __future__ import annotations

import time

import torch
import torch.nn.functional as F

from latent_twin_memory import LatentTwinMemory, MemoryConfig

DEV = "cuda" if torch.cuda.is_available() else "cpu"
D_IN, N_Z, N_PIN = 32, 32, 1
K_TRAIN, K_TEST = 64, 1024
DEPOSIT_P = 0.15
STEPS, BATCH, LR = 3000, 64, 3e-3
MODES = ["free", "dissipative", "conservative", "pinned"]

# --- task degeneracy control -------------------------------------------------
# With a FIXED deposit rate shared by every sample, M_t concentrates:
#   E[M_t] = 0.075t, SD[M_t] = sqrt(0.0444t), so SD/E ~ 1/sqrt(t) -> 0.
# A constant predictor emitting 0.075t then scores 7% relative error at t=1024
# without reading the input at all, which makes effect sizes uninterpretable.
#
# RATE_JITTER draws a per-sample rate p_i ~ U(RATE_LO, RATE_HI). The spread of
# M_t across samples is then dominated by rate variation, which does NOT shrink
# with t:  SD/E = SD(p)/E(p) = const (~38%). A model must genuinely track the
# sample rather than emit the population mean.
RATE_JITTER = False
RATE_LO, RATE_HI = 0.05, 0.25


def make_batch(B: int, K: int, gen: torch.Generator, v: torch.Tensor):
    """x: (B,K,D_IN)  a: (B,K)  M: (B,K+1) running total  content: (B,K,D_IN)"""
    if RATE_JITTER:
        p = RATE_LO + (RATE_HI - RATE_LO) * torch.rand(
            B, 1, generator=gen, device=DEV)
    else:
        p = torch.full((B, 1), DEPOSIT_P, device=DEV)
    mask = (torch.rand(B, K, generator=gen, device=DEV) < p).float()
    a = mask * torch.rand(B, K, generator=gen, device=DEV)
    content = 0.5 * torch.randn(B, K, D_IN, generator=gen, device=DEV)
    x = a[..., None] * v + content
    M = torch.cat([torch.zeros(B, 1, device=DEV), a.cumsum(dim=1)], dim=1)
    return x, a, M, content


def evaluate(model, gen, v, K: int, B: int = 64):
    """Relative ledger error |pred - M_t| / M_t at a set of horizons."""
    model.eval()
    with torch.no_grad():
        x, a, M, content = make_batch(B, K, gen, v)
        zs, _ = model.rollout(x)                       # (B, K+1, n_z)
        pred = model.ledger(zs.reshape(-1, N_Z)).reshape(B, K + 1, -1)[..., 0]
        abs_err = (pred - M).abs()
        rel = (abs_err / M.clamp(min=1e-3)).mean(dim=0)   # (K+1,)
    model.train()
    return rel


def train_one(mode: str, seed: int = 0, supervision: str = "dense"):
    """supervision:
         dense  -- ledger supervised at EVERY prefix t = 0..K_TRAIN (L0 setting)
         sparse -- ledger supervised ONLY at t = K_TRAIN (the L0.5 anchor test)

    Under `sparse`, the pinned model should still be correct at every other
    horizon: the balance law C^T z_t = C^T z_0 + sum_k C^T g_k, plus a shared
    per-block encoder, means one supervised time point identifies the flux
    function -- which then transports to all t. Baselines have no such
    mechanism and should fail at unsupervised horizons in BOTH directions
    (interior t < 64 as well as exterior t > 64).
    """
    assert supervision in {"dense", "sparse"}
    torch.manual_seed(seed)
    gen = torch.Generator(device=DEV).manual_seed(seed + 1)
    v = torch.randn(D_IN, generator=gen, device=DEV)

    cfg = MemoryConfig(d_in=D_IN, n_z=N_Z, mode=mode, n_pinned=N_PIN, eps=1e-3)
    model = LatentTwinMemory(cfg).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=1000, gamma=0.5)

    t0 = time.time()
    for step in range(STEPS):
        x, a, M, content = make_batch(BATCH, K_TRAIN, gen, v)
        zs, g = model.rollout(x)

        pred = model.ledger(zs.reshape(-1, N_Z)).reshape(BATCH, K_TRAIN + 1, -1)[..., 0]
        if supervision == "dense":
            l_ledger = F.mse_loss(pred, M)              # every prefix
        else:
            l_ledger = F.mse_loss(pred[:, -1], M[:, -1])  # ONE horizon only

        # CONTENT loss: reconstruct the current block from the memory state
        recon = model.dec(zs[:, 1:].reshape(-1, N_Z)).reshape(BATCH, K_TRAIN, D_IN)
        l_content = F.mse_loss(recon, content)

        loss = l_ledger + l_content
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

    dt = time.time() - t0
    rel = evaluate(model, gen, v, K_TEST)
    return model, rel, l_ledger.item(), l_content.item(), dt


def main():
    print(f"device={DEV}  train horizon={K_TRAIN}  test horizon={K_TEST}\n")
    horizons = [16, 64, 128, 256, 512, 1024]
    results = {}

    for mode in MODES:
        model, rel, lled, lcon, dt = train_one(mode)
        results[mode] = rel
        rep = model.gen.structure_report()
        print(f"[{mode:>12}] trained in {dt:5.1f}s   "
              f"ledger_mse={lled:.2e}  content_mse={lcon:.2e}  "
              f"max_eig(W+W^T)={rep['max_sym_eig']:+.2e}")

    print("\n" + "=" * 74)
    print("RELATIVE LEDGER ERROR vs HORIZON   (trained only up to t=64)")
    print("=" * 74)
    print(f"{'mode':>13} | " + " ".join(f"{('t=' + str(h)):>10}" for h in horizons))
    print("-" * 74)
    for mode in MODES:
        row = " ".join(f"{results[mode][h].item():>10.4f}" for h in horizons)
        tag = "  <-- in-dist" if False else ""
        print(f"{mode:>13} | {row}{tag}")
    print("-" * 74)
    print("t=16,64 are inside the training horizon; t>=128 is extrapolation.")

    # sqrt(t) check for the pinned model: is the error a random walk of flux
    # errors (expected, benign) or compounding drift (bad)?
    p = results["pinned"]
    print(f"\npinned: err(1024)/err(64) = {(p[1024] / p[64]).item():.2f}  "
          f"(sqrt-law would predict ~{(1024 / 64) ** 0.5:.2f} if abs error "
          f"random-walks and M_t grows linearly, less since M_t grows too)")


if __name__ == "__main__":
    main()

"""
L0-FRONTIER: the audit-corrected experiment.

WHAT THE AUDIT KILLED
  The original L0 "pinned is 12.7x better" was an artifact. eps=1e-3 was applied
  as -eps*I to the dissipative baseline but -eps*P to pinned (exempting the
  protected direction), and eps*K_test = 1.024 -- the baseline was forced to
  leak on exactly the test timescale. With eps=0, dissipative + oracle
  recalibration matches pinned to 1.00x. Fatal confounds also included: 2x live
  generator params for pinned, no gain/bias on the ledger readout, and a task
  label isomorphic to the architecture's invariant.

THE QUESTION THAT SURVIVES
  eps=0 ("never forget anything") was only viable because the latent had spare
  capacity. Under CAPACITY PRESSURE -- where the model must forget filler to
  keep reconstructing recent content -- uniform retention and ledger retention
  conflict. Then:
    (a) can learned decay discover selective protection on its own?
    (b) if it can in principle (diag/full modes CAN represent the pinned
        solution), does SGD actually find it under train-short/test-long?
  If learned modes find it reliably, the constraint buys nothing -> honest
  negative result, close the branch. If they don't while pinned is reliable,
  the surviving claim is about OPTIMIZATION RELIABILITY, not expressivity.

AUDIT FIXES APPLIED
  1. Every mode gets the same learned affine ledger readout: gain * <r, z> + b.
  2. Matched parameterisations; the unpinned K+S control is included. All
     modes carry both A (rotation) and decay parameters; decay is LEARNED
     everywhere (softplus, same init) -- no fixed floors, no asymmetric eps.
  3. Non-degenerate task: per-sample rate jitter (constant predictor is bad at
     every horizon) AND gated deposits (flux = flag * amount, so the label is a
     nonlinear function of the block that the encoder must compute -- not a
     linear functional isomorphic to any mode's built-in invariant).
  4. Capacity pressure: n_z = 8 with d_in = 32 and a content-reconstruction
     loss evaluated at the TEST horizon too. With no decay, accumulated state
     random-walks (~sqrt(t)) and swamps the current block, so content recon at
     t=1024 REQUIRES forgetting; the ledger forbids it on one direction.

MODES
  scalar   W = (A - A^T) - softplus(s) I        one shared decay (IndexMem's
                                                learned-lambda analogue; CANNOT
                                                protect one direction selectively)
  diag     W = (A - A^T) - diag(softplus(d))    per-dimension decay; CAN mimic
                                                pinning if SGD zeroes one d_i
                                                and routes the ledger there
  ks       W = (A - A^T) - BB^T - softplus(s) I full K+S, unpinned (the audit's
                                                requested matched control)
  pinned   W = P(A-A^T)P - (PB)(PB)^T - softplus(s) P   protection by construction

METRICS (both axes matter -- this is a frontier, not a single number)
  ledger: corr(pred, M_t) and skill vs oracle constant, at t in {64..1024}
  content: reconstruction MSE of the current block at t~64 and t~1024
  A mode "wins" only by doing well on BOTH at t=1024.
"""

from __future__ import annotations

import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEV = "cuda" if torch.cuda.is_available() else "cpu"
D_IN, N_Z = 32, 8
K_TRAIN, K_TEST = 64, 1024
STEPS, BATCH, LR = 3000, 64, 3e-3
SEEDS = [0, 1, 2]
MODES = ["scalar", "diag", "ks", "pinned"]
HORIZONS = [64, 128, 256, 512, 1024]
RATE_LO, RATE_HI = 0.10, 0.50
DECAY_INIT = 1e-3          # softplus-inverse applied below; same for every mode


def softplus_inv(y: float) -> float:
    return math.log(math.expm1(y))


class Gen(nn.Module):
    """Matched generator family. All modes: rotation + LEARNED decay."""

    def __init__(self, mode: str, n_z: int = N_Z):
        super().__init__()
        self.mode, self.n_z = mode, n_z
        s = 0.03 / (2 * math.sqrt(n_z))
        self.A = nn.Parameter(torch.randn(n_z, n_z) * s)
        b = math.sqrt(0.03 / (4 * n_z))
        self.B = nn.Parameter(torch.randn(n_z, n_z) * b)
        raw = softplus_inv(DECAY_INIT)
        self.s = nn.Parameter(torch.tensor(raw))                    # scalar decay
        self.d = nn.Parameter(torch.full((n_z,), raw))              # diag decay
        if mode == "pinned":
            C0 = torch.linalg.qr(torch.randn(n_z, 1))[0]
            self.C_raw = nn.Parameter(C0)

    def pinned_basis(self):
        return torch.linalg.qr(self.C_raw)[0] if self.mode == "pinned" else None

    def forward(self) -> torch.Tensor:
        I = torch.eye(self.n_z, device=self.A.device)
        K = self.A - self.A.T
        if self.mode == "scalar":
            return K - F.softplus(self.s) * I
        if self.mode == "diag":
            return K - torch.diag(F.softplus(self.d))
        if self.mode == "ks":
            return K - self.B @ self.B.T - F.softplus(self.s) * I
        # pinned
        C = self.pinned_basis()
        P = I - C @ C.T
        PB = P @ self.B
        return P @ K @ P - PB @ PB.T - F.softplus(self.s) * P


class Model(nn.Module):
    def __init__(self, mode: str):
        super().__init__()
        self.gen = Gen(mode)
        h = 64
        self.enc = nn.Sequential(nn.Linear(D_IN, h), nn.SiLU(),
                                 nn.Linear(h, h), nn.SiLU(), nn.Linear(h, N_Z))
        self.dec = nn.Sequential(nn.Linear(N_Z, h), nn.SiLU(),
                                 nn.Linear(h, h), nn.SiLU(), nn.Linear(h, D_IN))
        # IDENTICAL readout family for every mode (audit fix #1):
        # ledger = gain * <r, z> + bias. For pinned, r is tied to C (the
        # method); elsewhere r is free. gain+bias learned everywhere.
        if mode != "pinned":
            self.r = nn.Parameter(torch.linalg.qr(torch.randn(N_Z, 1))[0])
        self.gain = nn.Parameter(torch.tensor(1.0))
        self.bias = nn.Parameter(torch.tensor(0.0))

    def readout_dir(self):
        C = self.gen.pinned_basis()
        return C if C is not None else torch.linalg.qr(self.r)[0]

    def ledger(self, z):
        return self.gain * (z @ self.readout_dir())[..., 0] + self.bias

    def rollout(self, x):
        B, K, _ = x.shape
        E = torch.matrix_exp(self.gen())
        g = self.enc(x)
        z = torch.zeros(B, N_Z, device=x.device)
        zs = [z]
        for k in range(K):
            z = z @ E.T + g[:, k]
            zs.append(z)
        return torch.stack(zs, 1)


def make_batch(B, K, gen):
    """Gated deposits: flux = flag * amount. Both channels are IN the block,
    mixed with distractor content; the encoder must compute the product."""
    p = RATE_LO + (RATE_HI - RATE_LO) * torch.rand(B, 1, generator=gen, device=DEV)
    flag = (torch.rand(B, K, generator=gen, device=DEV) < p).float()
    amount = torch.rand(B, K, generator=gen, device=DEV)
    a = flag * amount
    M = torch.cat([torch.zeros(B, 1, device=DEV), a.cumsum(1)], 1)
    content = 0.5 * torch.randn(B, K, D_IN, generator=gen, device=DEV)
    x = content.clone()
    x[..., 0] = amount                       # amount channel (always present)
    x[..., 1] = 2 * flag - 1                 # flag channel
    return x, M, content


def train_one(mode: str, seed: int):
    torch.manual_seed(seed)
    gen = torch.Generator(device=DEV).manual_seed(seed + 1)
    model = Model(mode).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.StepLR(opt, 1000, gamma=0.5)
    t0 = time.time()
    for _ in range(STEPS):
        x, M, content = make_batch(BATCH, K_TRAIN, gen)
        zs = model.rollout(x)
        pred = model.ledger(zs)
        l_led = F.mse_loss(pred, M)
        recon = model.dec(zs[:, 1:].reshape(-1, N_Z)).reshape(BATCH, K_TRAIN, D_IN)
        l_con = F.mse_loss(recon, content)
        loss = l_led + l_con
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
    return model, time.time() - t0


@torch.no_grad()
def evaluate(model, seed, B=512):
    gen = torch.Generator(device=DEV).manual_seed(seed + 7777)
    x, M, content = make_batch(B, K_TEST, gen)
    zs = model.rollout(x)
    pred = model.ledger(zs)
    out = {"ledger": {}}
    for t in HORIZONS:
        p, m = pred[:, t].double(), M[:, t].double()
        mse_c = ((m.mean() - m) ** 2).mean().item()
        pc, mc = p - p.mean(), m - m.mean()
        corr = (pc @ mc).item() / max((pc.norm() * mc.norm()).item(), 1e-12)
        out["ledger"][t] = {
            "corr": corr,
            "skill": 1 - ((p - m) ** 2).mean().item() / mse_c,
            "rel": ((p - m).abs() / m.clamp(min=1e-3)).mean().item(),
        }
    # content reconstruction, early vs late in the SAME long rollout
    for name, t in [("early", K_TRAIN), ("late", K_TEST)]:
        r = model.dec(zs[:, t])
        out[f"content_{name}"] = F.mse_loss(r, content[:, t - 1]).item()
    # learned decay spectrum: is anything being protected?
    W = model.gen().double()
    sym_eigs = torch.linalg.eigvalsh(0.5 * (W + W.T).cpu())
    out["decay_min"] = -sym_eigs.max().item()   # slowest decay rate (0 = protected)
    out["decay_med"] = -sym_eigs.median().item()
    # alignment: does the slowest-decaying direction match the readout?
    evec = torch.linalg.eigh(0.5 * (W + W.T).cpu())[1][:, -1]
    r_dir = model.readout_dir()[:, 0].double().cpu()
    out["align"] = abs((evec @ r_dir).item())
    return out


def main():
    print(f"L0-FRONTIER  n_z={N_Z} d_in={D_IN}  gated+jittered ledger, "
          f"capacity pressure\nmodes={MODES} seeds={SEEDS}\n")
    res = {m: [] for m in MODES}
    for seed in SEEDS:
        for mode in MODES:
            model, dt = train_one(mode, seed)
            e = evaluate(model, seed)
            res[mode].append(e)
            L = e["ledger"][K_TEST]
            print(f"  s{seed} {mode:>7}: led@1024 corr={L['corr']:+.3f} "
                  f"skill={L['skill']:+.3f} | con early={e['content_early']:.3f} "
                  f"late={e['content_late']:.3f} | decay[min={e['decay_min']:.2e} "
                  f"med={e['decay_med']:.2e}] align={e['align']:.2f} ({dt:.0f}s)",
                  flush=True)

    print("\n" + "=" * 96)
    print("SUMMARY (mean +- std over seeds)  --  a mode must win BOTH columns to matter")
    print("=" * 96)
    print(f"{'mode':>8} | {'ledger skill @1024':>20} | {'content MSE late':>17} | "
          f"{'slowest decay':>13} | {'align':>6}")
    print("-" * 96)
    for m in MODES:
        sk = np.array([r["ledger"][K_TEST]["skill"] for r in res[m]])
        cl = np.array([r["content_late"] for r in res[m]])
        dm = np.array([r["decay_min"] for r in res[m]])
        al = np.array([r["align"] for r in res[m]])
        print(f"{m:>8} | {sk.mean():+.3f} +- {sk.std():.3f}     | "
              f"{cl.mean():.3f} +- {cl.std():.3f} | {dm.mean():.2e}   | {al.mean():.2f}")
    print("-" * 96)
    print("""
READING
  scalar : cannot protect selectively. Expect ledger skill OR content to fail.
  diag/ks: can represent the pinned solution. decay_min ~ 0 AND align ~ 1 means
           SGD FOUND selective protection on its own -> constraint buys nothing.
  pinned : protection by construction; the question is only whether it also
           keeps content viable (the complement is free to decay).
""")


if __name__ == "__main__":
    main()

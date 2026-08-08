"""
L1 (RUNG 1): the ledger experiment over REAL language-model states.

Everything upstream is real text through a frozen LM: haystack `count` streams
(deposits like "A shipment of 27 units was logged at the Ziegler depot" buried
in filler with same-template distractors), encoded by Qwen2.5-0.5B in LOCAL
context mode into 64-token block-pooled hidden states (cache/l1_*.pt). The
memory module is the only component that spans blocks -- identical arms,
recipe, and statistics to the toy experiments; only the inputs changed.

    input   X_k in R^896   (block-pooled hidden state, standardized)
    label   M_k = running sum of TARGET-entity deposit amounts (/40) up to
            block k -- distractor deposits from other entities must be ignored,
            so the flux is a semantic function of the block, not a template count
    train   streams ~60 blocks (seed-0 corpus)
    eval    held-out streams at the SAME horizon (seed-1 corpus)
    test    held-out streams at ~16x the horizon (seed-2 corpus, ~64k tokens)

ARMS (identical to l0_forget hardening): scalar, scalar0, diag0, ks, pinned.
Encoder widened for 896-dim inputs (896->128->64->n_z=8); decay init 1e-3
(scalar0/diag0: frozen ~0); same optimizer/clip/schedule family.

PRE-REGISTERED READINGS (committed before any cache or result existed):
  TRANSFER      scalar-learned skill at 16x < 0.5 while pinned > 0.8 and
                pinned beats scalar on >= 3/3 seeds -> the toy mechanism
                transfers to real representations.
  NO-TRANSFER   all arms tie (no paired separation at 3 seeds) OR every arm
                fails (flux estimation from real states is the bottleneck:
                even short-horizon eval skill < 0.5 for all arms) -> the toy
                mechanism does NOT transfer as measured; report which.
  Secondary check: does learned scalar decay converge near ~1e-3 again
  (attractor location on real data) -- diagnostic, not a gate.

Statistics at n=3 are indicative only (two-sided sign floor 0.25); the n=10
extension runs only if the 3-seed direction is clear.
"""

from __future__ import annotations

import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEV = "cuda" if torch.cuda.is_available() else "cpu"
N_Z = 8
STEPS, BATCH, LR = 2000, 32, 3e-3
DECAY_INIT = 1e-3
AMOUNT_SCALE = 40.0
ARMS = ["scalar", "scalar0", "diag0", "ks", "pinned"]
SEEDS = [0, 1, 2]


def softplus_inv(y: float) -> float:
    return math.log(math.expm1(y))


class Gen(nn.Module):
    def __init__(self, mode: str, n_z: int = N_Z):
        super().__init__()
        self.mode, self.n_z = mode, n_z
        s = 0.03 / (2 * math.sqrt(n_z))
        self.A = nn.Parameter(torch.randn(n_z, n_z) * s)
        b = math.sqrt(0.03 / (4 * n_z))
        self.B = nn.Parameter(torch.randn(n_z, n_z) * b)
        self.s = nn.Parameter(torch.tensor(
            -20.0 if mode == "scalar0" else softplus_inv(DECAY_INIT)))
        self.d = nn.Parameter(torch.full(
            (n_z,), -20.0 if mode == "diag0" else softplus_inv(DECAY_INIT)))
        if mode == "pinned":
            self.C_raw = nn.Parameter(torch.linalg.qr(torch.randn(n_z, 1))[0])

    def pinned_basis(self):
        return (torch.linalg.qr(self.C_raw)[0]
                if self.mode == "pinned" else None)

    def forward(self):
        I = torch.eye(self.n_z, device=self.A.device)
        K = self.A - self.A.T
        if self.mode in ("scalar", "scalar0"):
            return K - F.softplus(self.s) * I
        if self.mode == "diag0":
            return K - torch.diag(F.softplus(self.d))
        if self.mode == "ks":
            return K - self.B @ self.B.T - F.softplus(self.s) * I
        C = self.pinned_basis()
        P = I - C @ C.T
        PB = P @ self.B
        return P @ K @ P - PB @ PB.T - F.softplus(self.s) * P


class Model(nn.Module):
    def __init__(self, mode: str, d_model: int):
        super().__init__()
        self.gen = Gen(mode)
        self.enc = nn.Sequential(nn.Linear(d_model, 128), nn.SiLU(),
                                 nn.Linear(128, 64), nn.SiLU(),
                                 nn.Linear(64, N_Z))
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


def load_split(path: str, K: int | None = None):
    """-> X (N, K, d) standardization-ready, M (N, K+1) running labels."""
    blob = torch.load(path, weights_only=False)
    recs = blob["records"]
    n_blocks = min(r["X"].shape[0] for r in recs)
    if K is not None:
        n_blocks = min(n_blocks, K)
    X = torch.stack([r["X"][:n_blocks] for r in recs]).float()
    M = torch.zeros(len(recs), n_blocks + 1)
    for i, r in enumerate(recs):
        db, da = r.get("deposit_blocks"), r.get("deposit_amounts")
        if db is None:
            raise ValueError(f"record {i} lacks deposit fields")
        for b, a in zip(db.tolist(), da.tolist()):
            if b < n_blocks:
                M[i, b + 1:] += a / AMOUNT_SCALE
    return X, M


def evaluate(model, X, M, mu, sd, horizons):
    model.eval()
    with torch.no_grad():
        Xn = ((X - mu) / sd).to(DEV)
        zs = model.rollout(Xn)
        pred = model.ledger(zs).cpu()
    out = {}
    for t in horizons:
        p, m = pred[:, t].double(), M[:, t].double()
        mse_c = ((m.mean() - m) ** 2).mean().item()
        out[t] = 1 - ((p - m) ** 2).mean().item() / max(mse_c, 1e-12)
    model.train()
    return out


def main():
    tr_X, tr_M = load_split("cache/l1_train.pt")
    ev_X, ev_M = load_split("cache/l1_evalshort.pt")
    te_X, te_M = load_split("cache/l1_testlong.pt")
    K_tr, K_te = tr_X.shape[1], te_X.shape[1]
    d_model = tr_X.shape[2]
    mu, sd = tr_X.mean((0, 1), keepdim=True), tr_X.std((0, 1), keepdim=True) + 1e-6
    print(f"train {tuple(tr_X.shape)}  evalshort {tuple(ev_X.shape)}  "
          f"testlong {tuple(te_X.shape)}  d_model={d_model}")
    print(f"K_train={K_tr}  K_test={K_te}  ratio={K_te/K_tr:.1f}x")
    print(f"label scale: train M_final mean {tr_M[:, -1].mean():.2f} "
          f"(sd {tr_M[:, -1].std():.2f}); "
          f"testlong M_final mean {te_M[:, -1].mean():.2f}\n")
    hor_te = sorted({min(K_te, h) for h in (64, 256, 512, K_te)})

    res = {a: [] for a in ARMS}
    for seed in SEEDS:
        for arm in ARMS:
            torch.manual_seed(seed)
            gen = torch.Generator().manual_seed(seed + 1)
            model = Model(arm, d_model).to(DEV)
            opt = torch.optim.Adam(model.parameters(), lr=LR)
            sched = torch.optim.lr_scheduler.StepLR(opt, STEPS // 3, gamma=0.5)
            t0 = time.time()
            Xn_all = (tr_X - mu) / sd
            for _ in range(STEPS):
                idx = torch.randint(0, tr_X.shape[0], (BATCH,), generator=gen)
                xb = Xn_all[idx].to(DEV)
                mb = tr_M[idx].to(DEV)
                zs = model.rollout(xb)
                loss = F.mse_loss(model.ledger(zs), mb)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
            ev = evaluate(model, ev_X, ev_M, mu, sd, [K_tr])
            te = evaluate(model, te_X, te_M, mu, sd, hor_te)
            dec = F.softplus(model.gen.s).item() if arm not in ("diag0",) \
                else F.softplus(model.gen.d).max().item()
            res[arm].append({"ev": ev[K_tr], "te": te, "decay": dec})
            print(f"  s{seed} {arm:>8}: eval@{K_tr}={ev[K_tr]:+.3f}  "
                  + "  ".join(f"te@{t}={te[t]:+.3f}" for t in hor_te)
                  + f"  decay={dec:.2e}  ({time.time()-t0:.0f}s)", flush=True)

    print(f"\n{'arm':>8} | {'eval@train-horizon med':>22} | "
          f"{'test@16x med':>13} | {'learned decay med':>17}")
    print("-" * 72)
    for a in ARMS:
        e = np.median([r["ev"] for r in res[a]])
        t = np.median([r["te"][K_te] for r in res[a]])
        d = np.median([r["decay"] for r in res[a]])
        print(f"{a:>8} | {e:>+22.3f} | {t:>+13.3f} | {d:>17.2e}")
    print("-" * 72)

    sc = [r["te"][K_te] for r in res["scalar"]]
    pn = [r["te"][K_te] for r in res["pinned"]]
    all_short = {a: np.median([r["ev"] for r in res[a]]) for a in ARMS}
    wins = sum(p > s for p, s in zip(pn, sc))
    print("\nPRE-REGISTERED READING")
    if max(all_short.values()) < 0.5:
        print("  NO-TRANSFER (flux bottleneck): every arm fails even at the "
              "training horizon -- real-state flux estimation dominates; "
              "retention differences unmeasurable in this setup.")
    elif np.median(sc) < 0.5 and np.median(pn) > 0.8 and wins == 3:
        print(f"  TRANSFER: scalar med {np.median(sc):+.3f} < 0.5, pinned med "
              f"{np.median(pn):+.3f} > 0.8, pinned>scalar {wins}/3. The toy "
              "mechanism transfers to real LM representations at 3 seeds "
              "(indicative; n=10 extension warranted).")
    else:
        print(f"  MIXED/NO-TRANSFER: scalar med {np.median(sc):+.3f}, pinned "
              f"med {np.median(pn):+.3f}, pinned>scalar {wins}/3 -- does not "
              "meet the pre-stated TRANSFER bar; report as-is.")


if __name__ == "__main__":
    try:
        from capture_env import banner
        banner("train_memory")
    except ImportError:
        pass
    main()

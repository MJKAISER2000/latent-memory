"""
L1 (RUNG 1, FIX 1): learned attention pooling over per-token LM states.

The first L1 run (results/L1_count.txt) hit its pre-registered NO-TRANSFER
verdict: with MEAN pooling over 64-token blocks, every arm failed even at the
training horizon -- one deposit sentence is diluted ~4.5:1 and target-vs-
distractor entity discrimination is washed out before the memory ever sees it.

THE FIX under test: a learned single-query attention pooler, trained jointly
with the memory from the ledger loss. Per block: keys k_t = W_k h_t (896->64),
learned query q; pooled = sum softmax(q.k_t/8) * h_t. The pooler can learn to
zero in on deposit sentences and target entities. Everything downstream
(arms, recipe, statistics) is IDENTICAL to train_memory.py -- the arms are
imported from it, not copied.

Caches: cache/l1_*_tok.pt (per-token fp16 states via --store-tokens). The
original mean-pool caches and results are untouched -- results/L1_count.txt
remains the committed record of the mean-pooling run and is the comparison
baseline for the pooling gate below.

PRE-REGISTERED READINGS (committed before the token caches or results existed):
  POOLING GATE  median eval@train-horizon across arms >= +0.5 (mean-pool run:
                -0.14..-0.36) -> the flux bottleneck was pooling, as
                hypothesized. If ALL arms stay < +0.5 at the train horizon,
                the bottleneck is elsewhere (candidates: 64-token blocks too
                coarse for entity binding, local 2048-token encoding, layer
                choice) -> report NO-POOLING-FIX and stop; do not fish.
  TRANSFER      (only meaningful if the gate passes) scalar-learned skill at
                16x < 0.5 while pinned > 0.8, pinned > scalar on 3/3 seeds ->
                the toy mechanism transfers to real representations at usable
                accuracy.
  Secondary diagnostics: learned decay location (attractor scale ~1e-3?);
  whether frozen-decay arms again fail through rotation at 16x.
"""

from __future__ import annotations

import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from train_memory import Gen, N_Z, DECAY_INIT, AMOUNT_SCALE  # arms unchanged

DEV = "cuda" if torch.cuda.is_available() else "cpu"
STEPS, BATCH, LR = 2000, 32, 3e-3
D_ATT = 64
ARMS = ["scalar", "scalar0", "diag0", "ks", "pinned"]
SEEDS = [0, 1, 2]


class AttnPool(nn.Module):
    """Single-query attention over the tokens of one block."""

    def __init__(self, d_model: int, d_att: int = D_ATT):
        super().__init__()
        self.Wk = nn.Linear(d_model, d_att, bias=False)
        self.q = nn.Parameter(torch.randn(d_att) / math.sqrt(d_att))

    def forward(self, T):                    # T: (..., n_tok, d_model)
        k = self.Wk(T)                       # (..., n_tok, d_att)
        a = torch.softmax(k @ self.q / math.sqrt(k.shape[-1]), dim=-1)
        return (a.unsqueeze(-1) * T).sum(-2)  # (..., d_model)


class PooledModel(nn.Module):
    """AttnPool + the unchanged memory arms from train_memory."""

    def __init__(self, mode: str, d_model: int):
        super().__init__()
        self.pool = AttnPool(d_model)
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

    def rollout_pooled(self, xp):            # xp: (B, K, d_model) pooled
        B, K, _ = xp.shape
        E = torch.matrix_exp(self.gen())
        g = self.enc(xp)
        z = torch.zeros(B, N_Z, device=xp.device)
        zs = [z]
        for k in range(K):
            z = z @ E.T + g[:, k]
            zs.append(z)
        return torch.stack(zs, 1)

    def forward(self, T, mu, sd):            # T: (B, K, n_tok, d) fp32 on GPU
        xp = self.pool((T - mu) / sd)
        return self.rollout_pooled(xp)


def load_tok_split(path):
    blob = torch.load(path, weights_only=False)
    recs = blob["records"]
    n_blocks = min(r["T"].shape[0] for r in recs)
    T = torch.stack([r["T"][:n_blocks] for r in recs])        # cpu fp16
    M = torch.zeros(len(recs), n_blocks + 1)
    for i, r in enumerate(recs):
        for b, a in zip(r["deposit_blocks"].tolist(),
                        r["deposit_amounts"].tolist()):
            if b < n_blocks:
                M[i, b + 1:] += a / AMOUNT_SCALE
    return T, M


@torch.no_grad()
def evaluate(model, T, M, mu, sd, horizons, stream_chunk=6):
    model.eval()
    preds = []
    for i in range(0, T.shape[0], stream_chunk):
        Tc = T[i:i + stream_chunk].to(DEV).float()
        zs = model(Tc, mu, sd)
        preds.append(model.ledger(zs).cpu())
        del Tc, zs
    pred = torch.cat(preds)
    out = {}
    for t in horizons:
        p, m = pred[:, t].double(), M[:, t].double()
        mse_c = ((m.mean() - m) ** 2).mean().item()
        out[t] = 1 - ((p - m) ** 2).mean().item() / max(mse_c, 1e-12)
    model.train()
    return out


def main():
    tr_T, tr_M = load_tok_split("cache/l1_train_tok.pt")
    d_model = tr_T.shape[-1]
    # token-level standardizer from a fixed subsample of training tokens
    samp = tr_T[:, :, ::8].float().reshape(-1, d_model)
    mu = samp.mean(0).to(DEV)
    sd = (samp.std(0) + 1e-6).to(DEV)
    K_tr = tr_T.shape[1]
    print(f"train T {tuple(tr_T.shape)} (fp16 cpu)  K_train={K_tr}")

    ev_T, ev_M = load_tok_split("cache/l1_evalshort_tok.pt")

    res = {a: [] for a in ARMS}
    for seed in SEEDS:
        for arm in ARMS:
            torch.manual_seed(seed)
            gen = torch.Generator().manual_seed(seed + 1)
            model = PooledModel(arm, d_model).to(DEV)
            opt = torch.optim.Adam(model.parameters(), lr=LR)
            sched = torch.optim.lr_scheduler.StepLR(opt, STEPS // 3, gamma=0.5)
            t0 = time.time()
            for _ in range(STEPS):
                idx = torch.randint(0, tr_T.shape[0], (BATCH,), generator=gen)
                Tb = tr_T[idx].to(DEV).float()
                mb = tr_M[idx].to(DEV)
                zs = model(Tb, mu, sd)
                loss = F.mse_loss(model.ledger(zs), mb)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                del Tb, zs
            ev = evaluate(model, ev_T, ev_M, mu, sd, [K_tr])
            dec = (F.softplus(model.gen.s).item() if arm != "diag0"
                   else F.softplus(model.gen.d).max().item())
            res[arm].append({"ev": ev[K_tr], "decay": dec,
                             "model": model.state_dict()})
            print(f"  s{seed} {arm:>8}: eval@{K_tr}={ev[K_tr]:+.3f} "
                  f"decay={dec:.2e} ({time.time()-t0:.0f}s)", flush=True)

    # POOLING GATE decided on eval@train-horizon before any long-horizon look
    med_ev = {a: float(np.median([r["ev"] for r in res[a]])) for a in ARMS}
    print(f"\neval@{K_tr} medians: " +
          "  ".join(f"{a}={v:+.3f}" for a, v in med_ev.items()))
    if max(med_ev.values()) < 0.5:
        print("\nPRE-REGISTERED READING\n  NO-POOLING-FIX: attention pooling did "
              "not lift short-horizon skill above +0.5 for any arm. The flux "
              "bottleneck is NOT (only) mean-pooling dilution; candidates: "
              "block granularity, local-context encoding, layer choice. "
              "Long-horizon comparison skipped per protocol -- do not fish.")
        return

    del tr_T
    te_T, te_M = load_tok_split("cache/l1_testlong_tok.pt")
    K_te = te_T.shape[1]
    hor = sorted({min(K_te, h) for h in (64, 256, 512, K_te)})
    print(f"\ngate PASSED -> long-horizon eval at K_test={K_te} "
          f"({K_te/K_tr:.1f}x)")
    for seed_i, seed in enumerate(SEEDS):
        for arm in ARMS:
            model = PooledModel(arm, d_model).to(DEV)
            model.load_state_dict(res[arm][seed_i]["model"])
            te = evaluate(model, te_T, te_M, mu, sd, hor)
            res[arm][seed_i]["te"] = te
            print(f"  s{seed} {arm:>8}: "
                  + "  ".join(f"te@{t}={te[t]:+.3f}" for t in hor), flush=True)

    print(f"\n{'arm':>8} | {'eval@train med':>14} | {'test@16x med':>13} | "
          f"{'decay med':>10}")
    print("-" * 58)
    for a in ARMS:
        tmed = float(np.median([r["te"][K_te] for r in res[a]]))
        dmed = float(np.median([r["decay"] for r in res[a]]))
        print(f"{a:>8} | {med_ev[a]:>+14.3f} | {tmed:>+13.3f} | {dmed:>10.2e}")

    sc = [r["te"][K_te] for r in res["scalar"]]
    pn = [r["te"][K_te] for r in res["pinned"]]
    wins = sum(p > s for p, s in zip(pn, sc))
    print("\nPRE-REGISTERED READING")
    if np.median(sc) < 0.5 and np.median(pn) > 0.8 and wins == 3:
        print(f"  TRANSFER: scalar med {np.median(sc):+.3f} < 0.5, pinned med "
              f"{np.median(pn):+.3f} > 0.8, pinned>scalar {wins}/3. The toy "
              "mechanism transfers to real LM representations at usable "
              "accuracy (3 seeds, indicative; n=10 extension warranted).")
    else:
        print(f"  MIXED: scalar med {np.median(sc):+.3f}, pinned med "
              f"{np.median(pn):+.3f}, pinned>scalar {wins}/3 -- report as-is.")


if __name__ == "__main__":
    try:
        from capture_env import banner
        banner("train_memory_attn")
    except ImportError:
        pass
    main()

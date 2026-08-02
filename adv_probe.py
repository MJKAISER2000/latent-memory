"""
ADVERSARIAL CHECK 2 -- is the L0 win structural, or is it the eps term?
Plus: does `pinned` track the SAMPLE-SPECIFIC total or just the population mean?

THE ASYMMETRY (latent_twin_memory.py:127 vs :139)
    dissipative:  W = -(B B^T) - eps*I     <-- eps floor on EVERY direction
    pinned:       W = K - (PB)(PB)^T - eps*P  <-- eps*P is ZERO on the pinned dir
So `dissipative` is forbidden from having any conserved direction, by a
hyperparameter. Note -(B B^T) alone has a kernel and CAN conserve a direction
exactly; only the -eps*I term destroys it. Ablate eps and the two modes become
capability-equivalent.

DIAGNOSTICS (all at fixed t, across a batch with DIFFERENT deposit patterns)
  corr(pred_t, M_t)      ~0 => clock;  ~1 => tracks the sample
  slope of M on pred     1.0 => calibrated
  skill vs blind         1 - E|pred-M| / E|0.075t - M|;  <=0 => worse than no input
  increment regression   d_k = pred_{k+1} - pred_k  regressed on a_k.
                         slope~1,R2 high => it is integrating deposits
                         slope~0, d_k ~ 0.075 => it is a CLOCK
  ablations              a:=0 (content only) and a:=fresh independent draw.
                         A clock is unmoved by both.
"""

from __future__ import annotations

import math
import sys
import time

import torch
import torch.nn.functional as F

import toy_ledger as T
from latent_twin_memory import LatentTwinMemory, MemoryConfig

DEV = T.DEV
HOR = [64, 128, 256, 512, 1024]
B_EVAL = 1024
MU = T.DEPOSIT_P / 2


def train(mode, seed=0, eps=1e-3):
    """Byte-for-byte toy_ledger.train_one (dense) with eps exposed."""
    torch.manual_seed(seed)
    gen = torch.Generator(device=DEV).manual_seed(seed + 1)
    v = torch.randn(T.D_IN, generator=gen, device=DEV)
    cfg = MemoryConfig(d_in=T.D_IN, n_z=T.N_Z, mode=mode, n_pinned=T.N_PIN, eps=eps)
    model = LatentTwinMemory(cfg).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=T.LR)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=1000, gamma=0.5)
    t0 = time.time()
    for _ in range(T.STEPS):
        x, a, M, content = T.make_batch(T.BATCH, T.K_TRAIN, gen, v)
        zs, g = model.rollout(x)
        pred = model.ledger(zs.reshape(-1, T.N_Z)).reshape(T.BATCH, T.K_TRAIN + 1, -1)[..., 0]
        recon = model.dec(zs[:, 1:].reshape(-1, T.N_Z)).reshape(T.BATCH, T.K_TRAIN, T.D_IN)
        loss = F.mse_loss(pred, M) + F.mse_loss(recon, content)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
    return model, v, time.time() - t0


@torch.no_grad()
def ledger_of(model, x):
    zs, _ = model.rollout(x)
    B, Kp1, _ = zs.shape
    return model.ledger(zs.reshape(-1, T.N_Z)).reshape(B, Kp1, -1)[..., 0]


@torch.no_grad()
def probe(model, v, seed):
    g = torch.Generator(device=DEV).manual_seed(seed + 777)
    model.eval()
    x, a, M, content = T.make_batch(B_EVAL, T.K_TEST, g, v)
    pred = ledger_of(model, x).double()
    Md = M.double()

    r = {}
    for t in HOR:
        p, m = pred[:, t], Md[:, t]
        pc, mc = p - p.mean(), m - m.mean()
        den = (pc.norm() * mc.norm()).item()
        corr = (pc @ mc).item() / den if den > 1e-12 else float("nan")
        slope = ((pc @ mc) / pc.pow(2).sum()).item() if pc.pow(2).sum() > 1e-12 else float("nan")
        e_model = (p - m).abs().mean().item()
        e_blind = (MU * t - m).abs().mean().item()
        r[t] = dict(
            corr=corr, slope=slope,
            sd_pred=p.std().item(), sd_M=m.std().item(),
            mean_pred=p.mean().item(), mean_M=m.mean().item(),
            abs_err=e_model, abs_blind=e_blind, skill=1 - e_model / e_blind,
            rel=((p - m).abs() / m.clamp(min=1e-3)).mean().item(),
        )

    # increment regression: does the ledger move by a_k, or by a constant?
    d = (pred[:, 1:] - pred[:, :-1]).reshape(-1)
    av = a.double().reshape(-1)
    ac, dc = av - av.mean(), d - d.mean()
    inc = dict(
        slope=((ac @ dc) / ac.pow(2).sum()).item(),
        r2=((ac @ dc) ** 2 / (ac.pow(2).sum() * dc.pow(2).sum())).item(),
        mean_d=d.mean().item(), sd_d=d.std().item(),
        mean_a=av.mean().item(),
        # residual increment after removing the a_k-driven part
        resid_sd=(dc - ((ac @ dc) / ac.pow(2).sum()) * ac).std().item(),
    )

    # ablations
    x0 = 0.0 * a[..., None] * v + content                       # deposits removed
    a2 = (torch.rand(B_EVAL, T.K_TEST, generator=g, device=DEV) < T.DEPOSIT_P).float() \
        * torch.rand(B_EVAL, T.K_TEST, generator=g, device=DEV)
    x2 = a2[..., None] * v + content                            # fresh deposits
    p0 = ledger_of(model, x0).double()[:, T.K_TEST]
    p2 = ledger_of(model, x2).double()[:, T.K_TEST]
    M2 = a2.double().cumsum(1)[:, -1]
    abl = dict(
        pred_full=pred[:, T.K_TEST].mean().item(),
        pred_zero=p0.mean().item(),
        pred_fresh=p2.mean().item(),
        corr_fresh=torch.corrcoef(torch.stack([p2, M2]))[0, 1].item(),
        true_full=Md[:, T.K_TEST].mean().item(),
    )
    model.train()
    return r, inc, abl


def rep_line(tag, r):
    return (f"{tag:>26} | " + " ".join(
        f"{r[t]['corr']:>6.3f}" for t in HOR) + "  ||  " + " ".join(
        f"{r[t]['skill']:>+6.2f}" for t in HOR) + "  ||  " + " ".join(
        f"{r[t]['rel']:>6.3f}" for t in HOR))


def main():
    seeds = [int(s) for s in (sys.argv[1].split(",") if len(sys.argv) > 1 else ["0", "1"])]
    runs = [("pinned", 1e-3), ("dissipative", 1e-3), ("dissipative", 1e-6),
            ("conservative", 1e-3)]
    store = {}
    for seed in seeds:
        for mode, eps in runs:
            key = (mode, eps, seed)
            model, v, dt = train(mode, seed=seed, eps=eps)
            store[key] = probe(model, v, seed)
            r, inc, abl = store[key]
            print(f"[seed {seed}] {mode}(eps={eps:g}) {dt:.0f}s  "
                  f"rel@1024={r[1024]['rel']:.4f}  corr@1024={r[1024]['corr']:.3f}  "
                  f"skill@1024={r[1024]['skill']:+.3f}  "
                  f"inc_slope={inc['slope']:.3f} inc_r2={inc['r2']:.3f}", flush=True)
            del model
            torch.cuda.empty_cache()

    print("\n" + "=" * 118)
    print("BLIND FLOOR (input-independent, exact metric): "
          "t=64 0.5445 | 128 0.2222 | 256 0.1479 | 512 0.1017 | 1024 0.0710")
    print("BAYES FLOOR (best per-block encoder + exact accumulator):  "
          "t=64 0.0477 | 128 0.0337 | 256 0.0238 | 512 0.0168 | 1024 0.0119")
    print("=" * 118)
    hdr = " ".join(f"{('t' + str(t)):>6}" for t in HOR)
    print(f"{'run':>26} | {hdr}  ||  {hdr}  ||  {hdr}")
    print(f"{'':>26} | {'corr(pred,M)':^41}||{'skill vs blind':^41}||{'rel err':^41}")
    print("-" * 118)
    for (mode, eps, seed), (r, inc, abl) in store.items():
        print(rep_line(f"s{seed} {mode}(e={eps:g})", r))
    print("-" * 118)

    print("\nINCREMENT REGRESSION  d_k = pred_{k+1}-pred_k  ~  a_k     "
          "(true: slope 1, mean_d 0.075)")
    print(f"{'run':>26} | {'slope':>8} {'R^2':>8} {'mean_d':>8} {'sd_d':>8} {'resid_sd':>9}")
    for (mode, eps, seed), (r, inc, abl) in store.items():
        print(f"{f's{seed} {mode}(e={eps:g})':>26} | {inc['slope']:>8.4f} {inc['r2']:>8.4f} "
              f"{inc['mean_d']:>8.4f} {inc['sd_d']:>8.4f} {inc['resid_sd']:>9.4f}")

    print("\nABLATIONS at t=1024   (a clock is unmoved by removing the deposits)")
    print(f"{'run':>26} | {'true M':>8} {'pred':>8} {'pred|a=0':>9} "
          f"{'pred|fresh a':>13} {'corr(fresh)':>12}")
    for (mode, eps, seed), (r, inc, abl) in store.items():
        print(f"{f's{seed} {mode}(e={eps:g})':>26} | {abl['true_full']:>8.2f} "
              f"{abl['pred_full']:>8.2f} {abl['pred_zero']:>9.2f} "
              f"{abl['pred_fresh']:>13.2f} {abl['corr_fresh']:>12.3f}")


if __name__ == "__main__":
    main()

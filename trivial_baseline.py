"""
Is the L0 result real, or is the task degenerate?

THE WORRY
  Deposits are iid: a_k = Bernoulli(0.15) * Uniform(0,1), so
      E[a] = 0.075,  Var[a] = 0.15/3 - 0.075^2 = 0.04437
  Therefore M_t = sum_{k<t} a_k concentrates hard:
      E[M_t] = 0.075 t,   SD[M_t] = sqrt(0.04437 t)
  Relative spread SD/E = sqrt(0.04437/t)/0.075 -> shrinks as 1/sqrt(t).

  So a CONSTANT predictor that ignores every input and emits 0.075t gets
  relative error  E|M_t - 0.075t| / E[M_t]  =  sqrt(2*0.04437/(pi t))/0.075,
  which is ~7% at t=1024. Our `dissipative` (34%) and `conservative` (47%)
  numbers are WORSE than that. If so they are not merely suboptimal -- they are
  actively worse than ignoring the input, and "pinned beats dissipative 12.7x"
  is a much weaker statement than it sounds.

THE DECISIVE DIAGNOSTIC
  MSE cannot distinguish "tracks this sample's total" from "emits the population
  mean". Correlation can. At a fixed horizon t, across a batch of samples with
  DIFFERENT random deposit patterns, compute corr(pred, M_t).
     corr ~ 0  -> the model learned the constant rate. Task is degenerate.
     corr ~ 1  -> the model tracks the sample-specific total. Real.
  Also report skill score vs the constant predictor:
     skill = 1 - MSE_model / MSE_constant     (>0 means it beat the constant)
"""

from __future__ import annotations

import math

import numpy as np
import torch

import toy_ledger as T

P, HI = T.DEPOSIT_P, 1.0
E_A = P * HI / 2
VAR_A = P * (HI ** 2) / 3 - E_A ** 2
HORIZONS = [64, 128, 256, 512, 1024]
MODES = ["dissipative", "conservative", "pinned"]
B_EVAL = 1024


def analytic():
    print("analytic properties of the L0 task")
    print(f"  E[a]={E_A:.5f}  Var[a]={VAR_A:.5f}")
    print(f"{'t':>6} {'E[M_t]':>9} {'SD[M_t]':>9} {'SD/E':>8} {'const-pred rel err':>20}")
    for t in HORIZONS:
        m, sd = E_A * t, math.sqrt(VAR_A * t)
        mae = math.sqrt(2 * VAR_A * t / math.pi)      # E|N(0,sd)| = sd*sqrt(2/pi)
        print(f"{t:>6} {m:>9.3f} {sd:>9.3f} {sd/m:>8.4f} {mae/m:>19.4f}")


@torch.no_grad()
def diagnose(model, gen, v, K):
    model.eval()
    x, a, M, _ = T.make_batch(B_EVAL, K, gen, v)
    zs, _ = model.rollout(x)
    pred = model.ledger(zs.reshape(-1, T.N_Z)).reshape(B_EVAL, K + 1, -1)[..., 0]
    model.train()

    out = {}
    for t in HORIZONS:
        p, m = pred[:, t].double(), M[:, t].double()
        const = m.mean()                                  # oracle constant predictor
        mse_m = ((p - m) ** 2).mean().item()
        mse_c = ((const - m) ** 2).mean().item()
        pc, mc = p - p.mean(), m - m.mean()
        denom = (pc.norm() * mc.norm()).item()
        corr = (pc @ mc).item() / denom if denom > 1e-12 else float("nan")

        # ORACLE AFFINE RECALIBRATION: least-squares fit of a*pred + b to the
        # TRUE M_t, fitted on the eval set itself. This is the most generous
        # possible calibration -- it cannot be done at inference time, since it
        # needs the answers. Its purpose is diagnostic: it separates
        #   "the generator destroyed the information"   (ceiling is low)
        # from
        #   "the information survived but the gain drifted"  (ceiling is high,
        #    raw skill is not).
        # Because correlation is scale-invariant, the ceiling equals corr^2.
        var_p = pc.var(unbiased=False).item()
        if var_p > 1e-12:
            a = (pc @ mc).item() / (len(pc) * var_p)
            b = (m.mean() - a * p.mean()).item()
            mse_cal = ((a * p + b - m) ** 2).mean().item()
        else:
            a, b, mse_cal = 0.0, m.mean().item(), mse_c
        out[t] = {
            "corr": corr,
            "skill": 1 - mse_m / mse_c if mse_c > 0 else float("nan"),
            "skill_cal": 1 - mse_cal / mse_c if mse_c > 0 else float("nan"),
            "gain": a,          # 1.0 => already calibrated; far from 1 => drift
            "rel": ((p - m).abs() / m.clamp(min=1e-3)).mean().item(),
            "rel_const": ((const - m).abs() / m.clamp(min=1e-3)).mean().item(),
        }
    return out


def main():
    analytic()

    res = {}
    for mode in MODES:
        model, _, _, _, dt = T.train_one(mode, seed=0)
        g = torch.Generator(device=T.DEV).manual_seed(1)
        v = torch.randn(T.D_IN, generator=g, device=T.DEV)
        res[mode] = diagnose(model, g, v, T.K_TEST)
        print(f"  trained {mode} ({dt:.0f}s)", flush=True)

    hdr = " ".join(f"{('t=' + str(t)):>10}" for t in HORIZONS)
    for key, title, note in [
        ("corr", "CORRELATION corr(pred, M_t) across samples",
         "~0 => learned the constant rate (task degenerate);  ~1 => tracks the sample"),
        ("skill", "SKILL vs the oracle constant predictor  (1 - MSE_model/MSE_const)",
         "<=0 => no better than emitting the population mean"),
        ("rel", "relative error (the metric reported in README)", ""),
    ]:
        print("\n" + "=" * 72)
        print(title)
        if note:
            print(note)
        print("=" * 72)
        print(f"{'mode':>13} | {hdr}")
        print("-" * 72)
        for mode in MODES:
            print(f"{mode:>13} | " + " ".join(f"{res[mode][t][key]:>10.4f}" for t in HORIZONS))
        if key == "rel":
            print(f"{'CONSTANT':>13} | " +
                  " ".join(f"{res[MODES[0]][t]['rel_const']:>10.4f}" for t in HORIZONS))
        print("-" * 72)

    print("\nVERDICT")
    for mode in MODES:
        c, s = res[mode][1024]["corr"], res[mode][1024]["skill"]
        tag = ("tracks the sample" if c > 0.7 and s > 0 else
               "DEGENERATE - no better than the constant predictor" if s <= 0 else
               "partial")
        print(f"  {mode:>13} @t=1024: corr={c:+.3f} skill={s:+.3f}  -> {tag}")


if __name__ == "__main__":
    main()

"""
ADVERSARIAL CHECK 3 -- is the dissipative baseline's loss INFORMATION or GAIN?

adv_probe shows dissipative has corr(pred, M_1024) ~ 0.95 and per-block
increment slope ~1.00, yet rel err 0.34 (4.8x worse than ignoring the input).
That is the signature of a systematic scale error, not of forgetting.

Test: fit ONE affine recalibration  pred -> alpha*pred + beta  per horizon,
using a HELD-OUT batch, and re-score. If the gap collapses, the L0 headline
number is measuring a calibration artifact that a single learned scalar fixes.
"""
import math, sys, torch
import toy_ledger as T
from adv_probe import train, ledger_of

HOR = [64, 128, 256, 512, 1024]
MU = T.DEPOSIT_P / 2


def run(mode, seed, eps):
    model, v, dt = train(mode, seed=seed, eps=eps)
    model.eval()
    with torch.no_grad():
        return _score(model, v, seed) + (dt,)


@torch.no_grad()
def _score(model, v, seed):
    gA = torch.Generator(device=T.DEV).manual_seed(seed + 4242)   # fit split
    gB = torch.Generator(device=T.DEV).manual_seed(seed + 8484)   # score split
    out = {}
    xA, _, MA, _ = T.make_batch(512, T.K_TEST, gA, v)
    xB, _, MB, _ = T.make_batch(512, T.K_TEST, gB, v)
    pA, pB = ledger_of(model, xA).double(), ledger_of(model, xB).double()
    for t in HOR:
        a_, b_ = pA[:, t], MA[:, t].double()
        va = a_.var()
        al = (((a_ - a_.mean()) * (b_ - b_.mean())).mean() / va) if va > 1e-12 else torch.tensor(1.0)
        be = b_.mean() - al * a_.mean()
        p, m = pB[:, t], MB[:, t].double()
        pr = al * p + be
        rel = lambda q: ((q - m).abs() / m.clamp(min=1e-3)).mean().item()
        out[t] = (rel(p), rel(pr), al.item(), be.item(),
                  rel(torch.full_like(m, MU * t)))
    del model
    torch.cuda.empty_cache()
    return (out,)


if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    runs = [("dissipative", 1e-3), ("dissipative", 1e-6), ("pinned", 1e-3)]
    print(f"seed={seed}  affine recalibration fitted on a held-out batch, scored on another\n")
    print(f"{'run':>22} {'t':>6} {'raw rel':>9} {'recal rel':>10} {'alpha':>8} "
          f"{'beta':>9} {'blind':>8}")
    for mode, eps in runs:
        o, dt = run(mode, seed, eps)
        for t in HOR:
            r, rc, al, be, bl = o[t]
            print(f"{f'{mode}(e={eps:g})':>22} {t:>6} {r:>9.4f} {rc:>10.4f} "
                  f"{al:>8.4f} {be:>9.3f} {bl:>8.4f}", flush=True)
        print()

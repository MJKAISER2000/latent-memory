"""
ADVERSARIAL CHECK 1 -- what are the FLOORS of the L0 task? (no training needed)

Three numbers every L0 result must be compared against:

  (A) BLIND FLOOR. Best predictor that ignores every input. Because deposits are
      iid, M_t concentrates and a constant c*t is already good.
      NOTE: toy_ledger.py:70 reports  mean_b |pred - M| / max(M,1e-3),
      an expectation of a RATIO. results/trivial_baseline.txt reports
      E|M-mu|/E[M] instead, which UNDERSTATES the blind floor at small t
      (Jensen: the ratio blows up when M_t happens to be small). Both given.

  (B) BAYES FLOOR. x_k = a_k*v + 0.5*randn (toy_ledger.py:56-57) with v FIXED.
      The sufficient statistic is y = <x,v>/||v||^2 = a + N(0, 0.25/||v||^2).
      Even a perfect per-block encoder cannot recover a_k exactly. The best any
      shared per-block encoder + exact accumulator can do is sum_k E[a_k|y_k],
      whose error random-walks at sqrt(t * MMSE). This is the floor `pinned`
      is allowed to approach -- if pinned sits ON it, pinned is optimal, not
      degenerate.

  (C) DISSIPATIVE CEILING. latent_twin_memory.py:127 hardcodes
          W = -B B^T - eps*I,  eps = 1e-3
      so EVERY direction decays at rate >= eps. An accumulator built on it sums
      a geometric series, not an arithmetic one:
          reachable(t) = (1 - e^{-eps t}) / (1 - e^{-eps})   instead of t.
      Meanwhile latent_twin_memory.py:139 uses  S = -(PB)(PB)^T - eps*P,
      which EXEMPTS the pinned direction from the eps floor.
      If the measured dissipative error matches this curve, the L0 comparison
      is measuring a hyperparameter, not a structural property.
"""

import math

import torch

DEV = "cuda" if torch.cuda.is_available() else "cpu"
P, D_IN, EPS = 0.15, 32, 1e-3
HOR = [8, 16, 32, 64, 128, 256, 512, 1024]
N = 400_000


def blind_floor(g):
    mu, var = P / 2, P / 3 - (P / 2) ** 2
    print(f"E[a]={mu:.6f}  Var[a]={var:.6f}   (matches the stated 0.075 / 0.0444)")
    print(f"\n(A) BLIND FLOOR -- predictor uses NO input\n")
    print(f"{'t':>6} {'E[M]':>8} {'sd(M)':>7} {'E|M-mu|':>8} | "
          f"{'E|M-mu|/E[M]':>13} {'E[|M-mu|/M]':>12} {'best const':>11} {'c*/t':>8}"
          f" {'SE(B=64)':>9}")
    print("-" * 100)
    out = {}
    Ms = torch.empty(N, len(HOR), device=DEV)
    a_cum = torch.zeros(N, device=DEV)
    prev = 0
    for j, t in enumerate(HOR):
        for _ in range(t - prev):
            m = (torch.rand(N, generator=g, device=DEV) < P).float()
            a_cum += m * torch.rand(N, generator=g, device=DEV)
        Ms[:, j] = a_cum
        prev = t
    for j, t in enumerate(HOR):
        M = Ms[:, j].double()
        m_t, s_t = mu * t, math.sqrt(var * t)
        mae = math.sqrt(2 * var * t / math.pi)
        naive = mae / m_t
        actual = ((M - m_t).abs() / M.clamp(min=1e-3)).mean().item()
        cs = torch.linspace(0.4 * m_t, 1.1 * m_t, 176, device=DEV).double()
        vals = ((cs[:, None] - M[None, :]).abs() / M.clamp(min=1e-3)[None, :]).mean(1)
        k = int(vals.argmin())
        per = (M - m_t).abs() / M.clamp(min=1e-3)
        se = (per.std() / math.sqrt(64)).item()
        out[t] = (actual, vals[k].item(), (M - m_t).abs().mean().item())
        print(f"{t:>6} {m_t:>8.3f} {s_t:>7.3f} {mae:>8.3f} | {naive:>13.4f} "
              f"{actual:>12.4f} {vals[k].item():>11.4f} {cs[k].item()/t:>8.5f} {se:>9.4f}")
    return out


def bayes_floor(g):
    """MMSE of a_k given the single sufficient statistic in x_k."""
    v = torch.randn(D_IN, generator=g, device=DEV)
    nv2 = (v @ v).item()
    sig = 0.5 / math.sqrt(nv2)            # sd of <content,v>/||v||^2
    print(f"\n(B) BAYES FLOOR -- ||v||^2={nv2:.2f}, obs noise sd on a_k = {sig:.5f}")

    # E[(a - E[a|y])^2] by MC + numeric posterior over a grid.
    n = 200_000
    m = (torch.rand(n, generator=g, device=DEV) < P).float()
    a = m * torch.rand(n, generator=g, device=DEV)
    y = a + sig * torch.randn(n, generator=g, device=DEV)
    # posterior mean under the true prior: mixture of atom at 0 and U(0,1)
    grid = torch.linspace(0, 1, 2001, device=DEV)
    dg = grid[1] - grid[0]
    def post_mean(yv):
        lw0 = (1 - P) * torch.exp(-0.5 * (yv / sig) ** 2)                      # atom
        d = torch.exp(-0.5 * ((yv[:, None] - grid[None, :]) / sig) ** 2)
        w = P * d * dg                                                          # U(0,1)
        num = (w * grid[None, :]).sum(1)
        den = lw0 + w.sum(1)
        return num / den
    pm = torch.cat([post_mean(y[i:i + 20000]) for i in range(0, n, 20000)])
    mmse = ((a - pm) ** 2).mean().item()
    bias = (pm - a).mean().item()
    print(f"    MMSE per block = {mmse:.6f}  (sd {math.sqrt(mmse):.5f});  "
          f"Var(a)={P/3-(P/2)**2:.5f}   -> {1-mmse/(P/3-(P/2)**2):.4f} of variance recoverable")
    print(f"    residual mean bias = {bias:+.2e} (should be ~0: cond. expectation)")
    print(f"\n{'t':>6} {'E|err|':>9} {'E[|err|/M]':>11}   (best achievable by per-block enc + exact accumulator)")
    mu = P / 2
    for t in HOR:
        sd = math.sqrt(t * mmse)
        e = math.sqrt(2 / math.pi) * sd
        print(f"{t:>6} {e:>9.4f} {e/(mu*t):>11.4f}")
    return mmse


def dissipative_ceiling():
    print(f"\n(C) DISSIPATIVE CEILING from the hardcoded eps={EPS} decay floor")
    print("    reachable(t)/t = [(1-e^{-eps t})/(1-e^{-eps})]/t ;  gain G fit on t<=64")
    S = lambda t: (1 - math.exp(-EPS * t)) / (1 - math.exp(-EPS))
    G = 64 / S(64)     # best gain if trained to match at the training horizon
    print(f"    fitted gain G={G:.4f}\n")
    print(f"{'t':>6} {'S(t)':>9} {'S(t)/t':>8} {'pred rel err':>13} {'MEASURED dissip':>16}")
    meas = {64: 0.0720, 128: 0.0528, 256: 0.0838, 512: 0.1814, 1024: 0.3447}
    for t in HOR:
        s = S(t)
        err = abs(G * s / t - 1.0)
        mm = f"{meas[t]:.4f}" if t in meas else "-"
        print(f"{t:>6} {s:>9.2f} {s/t:>8.4f} {err:>13.4f} {mm:>16}")
    print("\n    If the last two columns track, the L0 gap is the eps term, not structure.")


if __name__ == "__main__":
    g = torch.Generator(device=DEV).manual_seed(20260802)
    blind_floor(g)
    bayes_floor(g)
    dissipative_ceiling()

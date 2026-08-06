"""
FREEZE PROBE: measure, don't assert, why init0 does not move under Adam.

Audit finding: the dashboard explained the init0 freeze via "softplus gradients
(~2e-9) fall below Adam's eps=1e-8 floor" -- a mechanism sentence that was
consistent with the outcome but never measured. This logs the actual per-step
quantities for the decay parameters of a diag model initialized at s=d=-20:

    |dL/d(raw)|        raw gradient reaching the parameter
    Adam m-hat, v-hat  first/second-moment estimates
    |step|             the actual parameter change applied per step
    cumulative drift   total movement in raw space over the run

Prediction under the eps-floor mechanism: sqrt(v-hat) << adam_eps=1e-8, so
|step| ~ lr * |m-hat| / adam_eps  << lr, and cumulative drift stays far below
the ~13 raw units needed to reach the attractor (softplus_inv(8e-4) ~ -7.1).
If instead |step| ~ lr (Adam sign-following), the freeze story is wrong and
the observed non-movement needs another explanation.

Short run (600 steps) is sufficient: the question is per-step step size, not
convergence. Uses the exact l0_frontier recipe otherwise.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

import l0_frontier as L


def main(steps: int = 600, seed: int = 0):
    torch.manual_seed(seed)
    gen = torch.Generator(device=L.DEV).manual_seed(seed + 1)
    model = L.Model("diag").to(L.DEV)
    with torch.no_grad():
        model.gen.s.fill_(-20.0)
        model.gen.d.fill_(-20.0)
    opt = torch.optim.Adam(model.parameters(), lr=L.LR)

    g_hist, step_hist = [], []
    d0 = model.gen.d.detach().clone()
    for k in range(steps):
        x, M, content = L.make_batch(L.BATCH, L.K_TRAIN, gen)
        zs = model.rollout(x)
        loss = F.mse_loss(model.ledger(zs), M)
        recon = model.dec(zs[:, 1:].reshape(-1, L.N_Z)).reshape(
            L.BATCH, L.K_TRAIN, L.D_IN)
        loss = loss + F.mse_loss(recon, content)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        before = model.gen.d.detach().clone()
        g_hist.append(model.gen.d.grad.abs().max().item())
        opt.step()
        step_hist.append((model.gen.d.detach() - before).abs().max().item())

    st = opt.state[model.gen.d]
    mhat = (st["exp_avg"] / (1 - 0.9 ** st["step"])).abs().max().item()
    vhat = (st["exp_avg_sq"] / (1 - 0.999 ** st["step"])).abs().max().item()
    drift = (model.gen.d.detach() - d0).abs().max().item()
    need = -7.1 - (-20.0)          # raw distance to the attractor's raw value

    g = np.array(g_hist); s = np.array(step_hist)
    print(f"FREEZE PROBE  diag init0, {steps} steps, seed {seed}, device {L.DEV}")
    print(f"  |dL/d(raw d)|   median {np.median(g):.3e}   p95 {np.percentile(g,95):.3e}")
    print(f"  Adam m-hat max  {mhat:.3e}")
    print(f"  Adam v-hat max  {vhat:.3e}   sqrt(v-hat) {vhat**0.5:.3e}   "
          f"adam_eps 1e-08  -> ratio sqrt(v)/eps = {vhat**0.5/1e-8:.3f}")
    print(f"  |step| median   {np.median(s):.3e}   (lr = {L.LR:.0e}; "
          f"sign-following Adam would give ~{L.LR:.0e})")
    print(f"  cumulative drift over {steps} steps: {drift:.3e} raw units")
    print(f"  raw distance to attractor: ~{need:.1f}; extrapolated steps to "
          f"reach it: {need/max(np.median(s),1e-300)/1000:.2e}k")
    verdict = (
        "EPS-FLOOR CONFIRMED: sqrt(v-hat) << adam_eps, steps collapse to ~lr*g/eps"
        if vhat**0.5 < 1e-8 and np.median(s) < 0.1 * L.LR
        else "EPS-FLOOR NOT the mechanism at measured scales -- revise wording")
    print(f"  VERDICT: {verdict}")


if __name__ == "__main__":
    main()

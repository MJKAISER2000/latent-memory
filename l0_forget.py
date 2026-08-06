"""
L0-FORGET: a benchmark where forgetting is PROVABLY necessary -- validation first.

WHY THIS EXISTS (audit finding, Aug 2026)
  Every prior comparison in this project shares one validity gap: the frontier
  task never actually forced forgetting. With d_in=32 >> n_z=8, content was
  never substantially reconstructable (all arms sat near the zero-predictor
  floor), so "content late" measured decoder noise, not retention -- and a
  never-decay memory (scalar0) matched or beat selective protection (pinned)
  on both axes. Consequently the data cannot yet distinguish the central
  claim's substance ("protect the ledger WHILE forgetting the rest") from the
  trivial alternative ("never forget anything").

DESIGN CHANGES vs l0_frontier
  * d_in = 8 = n_z, content variance 1.0: the current block IS reconstructable,
    so interference from undecayed history has something to destroy.
  * content target = dims 2..7 (the non-deposit channels) of the CURRENT block.
  * Everything else mirrors the frontier recipe (same optimizer, clipping,
    schedule shape, gated+jittered deposits, K_train=64, K_test=1024).

VALIDATION-FIRST PROTOCOL
  Phase A ("necessity"): train models whose scalar decay is FIXED (not learned)
  at d in {0, 1e-3, 1e-2, 1e-1, 1.0}, rotation learned. If content-late at the
  best d>0 is not materially better than at d=0, THE TASK STILL DOES NOT FORCE
  FORGETTING and Phase B is meaningless -- report that outcome honestly and
  stop. Criterion (pre-stated): median content-late(best d>0) at least 20%
  below content-late(d=0), 3 seeds.

  Phase B (only if A passes): learned-scalar vs scalar0-frozen vs pinned,
  3 seeds. The question the whole project needs answered: does hard selective
  protection uniquely hold BOTH axes when retention and forgetting genuinely
  conflict?
    - scalar-learned : expected to sit at the attractor -> ledger dies
    - scalar0-frozen : no forgetting anywhere -> content should now die
    - pinned         : ledger exact + complement free to decay -> both live?
  Any other outcome falsifies the corresponding part of the story.

Self-contained by design (no imports from l0_frontier): this experiment must
not be able to perturb, or be perturbed by, the established files.
"""

from __future__ import annotations

import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEV = "cuda" if torch.cuda.is_available() else "cpu"
D_IN, N_Z = 8, 8
CONTENT_DIMS = slice(2, 8)          # reconstruction target: 6 channels
K_TRAIN, K_TEST = 64, 1024
STEPS, BATCH, LR = 2000, 64, 3e-3
RATE_LO, RATE_HI = 0.10, 0.50
DECAY_INIT = 1e-3
FIXED_GRID = [0.0, 1e-3, 1e-2, 1e-1, 1.0]
SEEDS = [0, 1, 2]
NECESSITY_MARGIN = 0.20             # pre-stated: >=20% relative improvement


def softplus_inv(y: float) -> float:
    return math.log(math.expm1(y))


class Gen(nn.Module):
    """Modes: 'fixed' (scalar decay frozen at a given value, rotation learned),
    'scalar' (learned decay), 'scalar0' (softplus init -20 ~ frozen zero),
    'pinned' (hard C^T W = 0, learned decay on the complement)."""

    def __init__(self, mode: str, n_z: int = N_Z, fixed_d: float = 0.0):
        super().__init__()
        self.mode, self.n_z = mode, n_z
        s = 0.03 / (2 * math.sqrt(n_z))
        self.A = nn.Parameter(torch.randn(n_z, n_z) * s)
        b = math.sqrt(0.03 / (4 * n_z))
        self.B = nn.Parameter(torch.randn(n_z, n_z) * b)
        raw = softplus_inv(DECAY_INIT) if mode != "scalar0" else -20.0
        self.s = nn.Parameter(torch.tensor(raw))
        if mode == "fixed":
            self.register_buffer("fixed_d", torch.tensor(float(fixed_d)))
        if mode == "pinned":
            self.C_raw = nn.Parameter(torch.linalg.qr(torch.randn(n_z, 1))[0])

    def pinned_basis(self):
        return (torch.linalg.qr(self.C_raw)[0] if self.mode == "pinned" else None)

    def forward(self):
        I = torch.eye(self.n_z, device=self.A.device)
        K = self.A - self.A.T
        if self.mode == "fixed":
            return K - self.fixed_d * I
        if self.mode in ("scalar", "scalar0"):
            return K - F.softplus(self.s) * I
        C = self.pinned_basis()
        P = I - C @ C.T
        PB = P @ self.B
        return P @ K @ P - PB @ PB.T - F.softplus(self.s) * P


class Model(nn.Module):
    def __init__(self, mode: str, fixed_d: float = 0.0):
        super().__init__()
        self.gen = Gen(mode, fixed_d=fixed_d)
        h = 64
        self.enc = nn.Sequential(nn.Linear(D_IN, h), nn.SiLU(),
                                 nn.Linear(h, h), nn.SiLU(), nn.Linear(h, N_Z))
        n_c = CONTENT_DIMS.stop - CONTENT_DIMS.start
        self.dec = nn.Sequential(nn.Linear(N_Z, h), nn.SiLU(),
                                 nn.Linear(h, h), nn.SiLU(), nn.Linear(h, n_c))
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
    p = RATE_LO + (RATE_HI - RATE_LO) * torch.rand(B, 1, generator=gen, device=DEV)
    flag = (torch.rand(B, K, generator=gen, device=DEV) < p).float()
    amount = torch.rand(B, K, generator=gen, device=DEV)
    M = torch.cat([torch.zeros(B, 1, device=DEV), (flag * amount).cumsum(1)], 1)
    x = torch.randn(B, K, D_IN, generator=gen, device=DEV)   # content var 1.0
    x[..., 0] = amount
    x[..., 1] = 2 * flag - 1
    return x, M


def train_one(mode: str, seed: int, fixed_d: float = 0.0):
    torch.manual_seed(seed)
    gen = torch.Generator(device=DEV).manual_seed(seed + 1)
    model = Model(mode, fixed_d=fixed_d).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.StepLR(opt, max(STEPS // 3, 1), gamma=0.5)
    t0 = time.time()
    for _ in range(STEPS):
        x, M = make_batch(BATCH, K_TRAIN, gen)
        zs = model.rollout(x)
        l_led = F.mse_loss(model.ledger(zs), M)
        recon = model.dec(zs[:, 1:].reshape(-1, N_Z)).reshape(
            BATCH, K_TRAIN, -1)
        l_con = F.mse_loss(recon, x[..., CONTENT_DIMS])
        (l_led + l_con).backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        sched.step()
    return model, time.time() - t0


@torch.no_grad()
def evaluate(model, seed, B=512):
    gen = torch.Generator(device=DEV).manual_seed(seed + 7777)
    x, M = make_batch(B, K_TEST, gen)
    zs = model.rollout(x)
    pred = model.ledger(zs)
    out = {}
    for t in (K_TRAIN, K_TEST):
        p, m = pred[:, t].double(), M[:, t].double()
        mse_c = ((m.mean() - m) ** 2).mean().item()
        out[f"skill_{t}"] = 1 - ((p - m) ** 2).mean().item() / mse_c
    for name, t in (("early", K_TRAIN), ("late", K_TEST)):
        r = model.dec(zs[:, t])
        out[f"con_{name}"] = F.mse_loss(r, x[:, t - 1, CONTENT_DIMS]).item()
    W = model.gen().double()
    out["decay_min"] = -torch.linalg.eigvalsh(
        0.5 * (W + W.T).cpu()).max().item()
    return out


def phase_a():
    print(f"PHASE A -- necessity validation. Fixed decay grid {FIXED_GRID}, "
          f"{len(SEEDS)} seeds.\nPre-stated criterion: median content-late at "
          f"best d>0 must be >={NECESSITY_MARGIN:.0%} below d=0.\n", flush=True)
    res = {d: [] for d in FIXED_GRID}
    for seed in SEEDS:
        for d in FIXED_GRID:
            model, dt = train_one("fixed", seed, fixed_d=d)
            e = evaluate(model, seed)
            res[d].append(e)
            print(f"  s{seed} d={d:<7g} con_late={e['con_late']:.4f} "
                  f"con_early={e['con_early']:.4f} skill@1024={e['skill_1024']:+.3f} "
                  f"({dt:.0f}s)", flush=True)
    print(f"\n{'d':>8} {'con_late med':>13} {'con_early med':>14} {'skill@1024 med':>15}")
    meds = {}
    for d in FIXED_GRID:
        cl = float(np.median([r["con_late"] for r in res[d]]))
        ce = float(np.median([r["con_early"] for r in res[d]]))
        sk = float(np.median([r["skill_1024"] for r in res[d]]))
        meds[d] = cl
        print(f"{d:>8g} {cl:>13.4f} {ce:>14.4f} {sk:>+15.3f}")
    base = meds[0.0]
    best_d = min((d for d in FIXED_GRID if d > 0), key=lambda d: meds[d])
    improve = (base - meds[best_d]) / base if base > 0 else 0.0
    passed = improve >= NECESSITY_MARGIN
    print(f"\nNECESSITY: content-late d=0: {base:.4f}  best d>0 ({best_d:g}): "
          f"{meds[best_d]:.4f}  improvement {improve:.1%} "
          f"(criterion >={NECESSITY_MARGIN:.0%}) -> "
          f"{'PASS - forgetting is necessary on this task' if passed else 'FAIL - task still does not force forgetting; Phase B not run'}")
    return passed


def phase_b():
    print(f"\nPHASE B -- the decisive comparison. arms=scalar/scalar0/pinned, "
          f"seeds={SEEDS}\n", flush=True)
    arms = ["scalar", "scalar0", "pinned"]
    res = {a: [] for a in arms}
    for seed in SEEDS:
        for a in arms:
            model, dt = train_one(a, seed)
            e = evaluate(model, seed)
            res[a].append(e)
            print(f"  s{seed} {a:>8}: skill@1024={e['skill_1024']:+.3f} "
                  f"con_late={e['con_late']:.4f} decay_min={e['decay_min']:.2e} "
                  f"({dt:.0f}s)", flush=True)
    print(f"\n{'arm':>9} | {'ledger skill@1024 med':>22} | {'content late med':>17}")
    for a in arms:
        sk = float(np.median([r["skill_1024"] for r in res[a]]))
        cl = float(np.median([r["con_late"] for r in res[a]]))
        print(f"{a:>9} | {sk:>+22.3f} | {cl:>17.4f}")
    print("""
READING (pre-stated predictions; any other outcome falsifies that row)
  scalar : attractor -> ledger dies, content ok
  scalar0: no forgetting -> ledger holds, content dies (if Phase A was honest)
  pinned : ledger holds AND content survives -> the first data actually
           supporting selective protection over never-forgetting""")


if __name__ == "__main__":
    if phase_a():
        phase_b()

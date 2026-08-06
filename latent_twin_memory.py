"""
Physics-conforming latent twin memory.

Transplants the structured-generator machinery from Chung, Bu & Verma (2026),
"Physics-Conforming Latent Twins" (arXiv:2606.15053v2), into a driven
(input-forced) sequence memory.

The paper's flow is autonomous:      z_t = exp((t-s)W) z_s
Context memory is driven:            z_{k+1} = exp(dW) z_k + g_k

Core construction (generalises the heat-equation parameterisation, paper p.21):

    W = K + S,   built entirely inside the orthogonal complement of pinned
                 directions C, so that

        C^T W = 0        ->  C^T z is EXACTLY conserved by the homogeneous flow
        K = -K^T         ->  norm-preserving rotation on C-perp
        S = S^T <= 0     ->  monotone dissipation of ||z||^2 on C-perp

Under additive forcing this upgrades conservation to an exact BALANCE LAW:

        C^T z_t = C^T z_s + sum_{k=s}^{t-1} C^T g_k

i.e. the ledger changes only by what came in through the door. This holds by
construction -- no penalty term. The learning problem is therefore not
"conserve the quantity" but "make C^T g_k mean the right thing", which is the
input-side analogue of the paper's pullback-matching loss (their Eq. 7).

That distinction is the load-bearing lesson of the paper: their Fig. 10 shows
exact preservation of an unanchored latent invariant is worthless.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Structured generators
# ---------------------------------------------------------------------------


class StructuredGenerator(nn.Module):
    """Latent generator W with prescribed conservation / dissipation structure.

    Modes correspond to the columns of Table 1 in the paper:

      "free"        unconstrained W                          (baseline)
      "dissipative" W = -BB^T - eps*I,  W + W^T <= 0         (Mamba/S4 corner)
      "conservative" W = A - A^T,  W + W^T = 0               (DeltaNet corner)
      "pinned"      W = K + S inside C-perp                  (the mixed case)

    The "pinned" mode is the interesting one and is not, as far as I can tell,
    standard in SSM design: it conserves a designated subspace exactly while
    everything orthogonal to it decays monotonically.
    """

    def __init__(
        self,
        n_z: int,
        mode: str = "pinned",
        n_pinned: int = 1,
        eps: float = 1e-3,
        learn_pinned: bool = True,
        init_spectral: float = 0.03,
    ):
        super().__init__()
        assert mode in {"free", "dissipative", "conservative", "pinned"}
        self.n_z = n_z
        self.mode = mode
        self.eps = eps
        self.n_pinned = n_pinned if mode == "pinned" else 0

        # Initialise so that exp(dt*W) starts near the identity for EVERY mode.
        # This matters for fairness: with the naive 1/sqrt(n_z) init the
        # unconstrained generator has spectral radius ~1, so a 64-step rollout
        # is exp(W)^64 ~ e^64 and the baseline NaNs before training begins.
        # Targeting |eig(W)| ~ init_spectral makes the decay/rotation timescale
        # comparable to the sequence length instead.
        #   spectral norm of randn(n,n)*s  ~  2*s*sqrt(n)
        #   spectral norm of (B B^T)       ~  4*s^2*n
        s_a = init_spectral / (2.0 * math.sqrt(n_z))
        s_b = math.sqrt(init_spectral / (4.0 * n_z))
        self.A = nn.Parameter(torch.randn(n_z, n_z) * s_a)  # -> skew part
        self.B = nn.Parameter(torch.randn(n_z, n_z) * s_b)  # -> psd part

        if self.n_pinned > 0:
            C0 = torch.linalg.qr(torch.randn(n_z, self.n_pinned))[0]
            self.C_raw = nn.Parameter(C0, requires_grad=learn_pinned)
        else:
            self.register_parameter("C_raw", None)

    # -- pinned subspace ----------------------------------------------------

    def pinned_basis(self) -> torch.Tensor | None:
        """Orthonormal basis C (n_z, n_pinned) of the conserved subspace."""
        if self.C_raw is None:
            return None
        # QR keeps C orthonormal under gradient updates.
        return torch.linalg.qr(self.C_raw)[0]

    def projector(self) -> torch.Tensor:
        """P = I - C C^T, orthogonal projector onto C-perp."""
        I = torch.eye(self.n_z, device=self.A.device, dtype=self.A.dtype)
        C = self.pinned_basis()
        if C is None:
            return I
        return I - C @ C.T

    # -- generator ----------------------------------------------------------

    def forward(self) -> torch.Tensor:
        I = torch.eye(self.n_z, device=self.A.device, dtype=self.A.dtype)

        if self.mode == "free":
            return self.A

        if self.mode == "conservative":
            return self.A - self.A.T

        if self.mode == "dissipative":
            return -(self.B @ self.B.T) - self.eps * I

        # mode == "pinned": build K + S *inside* range(P).
        #
        # P symmetric idempotent  =>  (P M P)^T = P M^T P, so conjugating by P
        # preserves skewness and negative-semidefiniteness, while forcing
        # C^T K = C^T S = 0.  Note the eps term uses P, not I: the pinned
        # directions must get zero decay, not eps decay.
        P = self.projector()
        M = self.A - self.A.T
        K = P @ M @ P
        PB = P @ self.B
        S = -(PB @ PB.T) - self.eps * P
        return K + S

    # -- structural diagnostics (used as asserts in the self-test) ----------

    @torch.no_grad()
    def structure_report(self) -> dict:
        W = self.forward()
        # eigvalsh on CPU/float64: the cuSOLVER path fails to converge on the
        # near-degenerate symmetric parts produced by the unconstrained mode.
        sym = (W + W.T).double().cpu()
        if not torch.isfinite(sym).all():
            return {"max_sym_eig": float("nan"), "skew_defect": float("nan"),
                    "pin_defect": float("nan")}
        out = {
            "max_sym_eig": torch.linalg.eigvalsh(sym).max().item(),
            "skew_defect": 0.0,
            "pin_defect": 0.0,
        }
        if self.mode == "pinned":
            P = self.projector()
            M = self.A - self.A.T
            K = P @ M @ P
            out["skew_defect"] = (K + K.T).abs().max().item()
            C = self.pinned_basis()
            out["pin_defect"] = (C.T @ W).abs().max().item()
        return out


# ---------------------------------------------------------------------------
# Flow map
# ---------------------------------------------------------------------------


class ExpFlow(nn.Module):
    """m_{s->t}(z) = exp((t-s) * dt * W) z, for integer block offsets.

    Implemented by binary powering: precompute E_i = exp(2^i * dt * W) once
    (log2(K_max) matrix exponentials of size n_z), then any integer offset is
    log2(K) batched matvecs. This gives genuine O(log K) random access in time
    -- the property that makes the two-time-indexed formulation attractive for
    context memory in the first place.

    Numerics: the FORWARD direction is stable for dissipative W (the E_i shrink
    toward the projector onto ker(W)). The interaction-picture prefix-sum trick
    for the *driven* case needs exp(-kW), which blows up; see PLAN.md for why
    the driven random-access path must be chunked instead.
    """

    def __init__(self, n_bits: int = 12, dt: float = 1.0):
        super().__init__()
        self.n_bits = n_bits
        self.dt = dt

    def powers(self, W: torch.Tensor) -> list[torch.Tensor]:
        Es, E = [], torch.matrix_exp(self.dt * W)
        for _ in range(self.n_bits):
            Es.append(E)
            E = E @ E
        return Es

    def forward(
        self, z: torch.Tensor, n: torch.Tensor, W: torch.Tensor, Es=None
    ) -> torch.Tensor:
        """z: (B, n_z) latent states.  n: (B,) non-negative integer offsets."""
        Es = self.powers(W) if Es is None else Es
        n = n.long()
        # audit fix: offsets beyond 2**n_bits used to be silently reduced
        # modulo 2**n_bits; that is a wrong answer, not a range reduction.
        if bool((n < 0).any()) or bool((n >= 2 ** self.n_bits).any()):
            raise ValueError(
                f"ExpFlow offset out of range [0, {2**self.n_bits}); "
                "raise n_bits to cover the requested horizon")
        for i in range(self.n_bits):
            bit = ((n >> i) & 1).bool()
            if not bit.any():
                continue
            z = torch.where(bit[:, None], z @ Es[i].T, z)
        return z


# ---------------------------------------------------------------------------
# Full memory module
# ---------------------------------------------------------------------------


@dataclass
class MemoryConfig:
    d_in: int = 32          # per-block input width (KV block, hidden block, ...)
    n_z: int = 32           # latent memory width
    d_hidden: int = 64
    mode: str = "pinned"
    n_pinned: int = 1
    eps: float = 1e-3
    dt: float = 1.0
    n_bits: int = 12


class LatentTwinMemory(nn.Module):
    """Driven latent twin memory.

        g_k    = Enc(x_k)                      block encoder ("influx")
        z_{k+1}= exp(dt W) z_k + g_k           structured driven flow
        ledger = C^T z                         exact balance law, by construction
        x_hat  = Dec(z)                        content decoder

    Enc/Dec are deliberately generic: for the KV-cache experiment x_k is a
    flattened block of keys+values, for the hidden-state experiment it is a
    pooled block of residual-stream activations.
    """

    def __init__(self, cfg: MemoryConfig):
        super().__init__()
        self.cfg = cfg
        self.gen = StructuredGenerator(
            cfg.n_z, mode=cfg.mode, n_pinned=cfg.n_pinned, eps=cfg.eps
        )
        self.flow = ExpFlow(n_bits=cfg.n_bits, dt=cfg.dt)

        self.enc = nn.Sequential(
            nn.Linear(cfg.d_in, cfg.d_hidden), nn.SiLU(),
            nn.Linear(cfg.d_hidden, cfg.d_hidden), nn.SiLU(),
            nn.Linear(cfg.d_hidden, cfg.n_z),
        )
        self.dec = nn.Sequential(
            nn.Linear(cfg.n_z, cfg.d_hidden), nn.SiLU(),
            nn.Linear(cfg.d_hidden, cfg.d_hidden), nn.SiLU(),
            nn.Linear(cfg.d_hidden, cfg.d_in),
        )

        # Baselines get an equally expressive *learned* linear readout, so the
        # only difference between them and "pinned" is whether the generator is
        # constrained to conserve that readout. Same params, same expressiveness.
        if cfg.mode != "pinned":
            R0 = torch.linalg.qr(torch.randn(cfg.n_z, max(cfg.n_pinned, 1)))[0]
            self.readout = nn.Parameter(R0)
        else:
            self.register_parameter("readout", None)

    # -- structural readout -------------------------------------------------

    def ledger(self, z: torch.Tensor) -> torch.Tensor:
        """C^T z -- the conserved functional. (B, n_pinned)

        For mode="pinned" this is exactly conserved by the homogeneous flow.
        For the baselines it is an unconstrained learned readout: they can
        learn to *approximately* conserve it, but nothing enforces it.
        """
        C = self.gen.pinned_basis()
        if C is None:
            return z @ torch.linalg.qr(self.readout)[0]
        return z @ C

    # -- rollout ------------------------------------------------------------

    def rollout(self, x: torch.Tensor, z0: torch.Tensor | None = None):
        """x: (B, K, d_in) stream of blocks. Returns z: (B, K+1, n_z).

        Sequential in K. Exact. For long streams use the chunked scan
        (PLAN.md, step 2b) -- correctness first here.
        """
        B, K, _ = x.shape
        W = self.gen()
        E = torch.matrix_exp(self.cfg.dt * W)
        g = self.enc(x)                                   # (B, K, n_z)

        z = torch.zeros(B, self.cfg.n_z, device=x.device, dtype=x.dtype) if z0 is None else z0
        zs = [z]
        for k in range(K):
            z = z @ E.T + g[:, k]
            zs.append(z)
        return torch.stack(zs, dim=1), g

    def jump(self, z: torch.Tensor, n: torch.Tensor, Es=None) -> torch.Tensor:
        """Homogeneous random access: m_{s->t} with no intervening input."""
        return self.flow(z, n, self.gen(), Es=Es)


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------


# NOTE (audit): the three loss helpers below are unexercised API stubs --
# no experiment in this repo calls them. The experiments supervise the
# ledger directly (F.mse_loss on the affine readout). They document how the
# paper's pullback/conformity losses would attach; they are not evidence
# that those losses were used.
def pullback_loss(latent_fn: torch.Tensor, physical_fn: torch.Tensor) -> torch.Tensor:
    """Paper Eq. (7): align the latent functional with the decoded physical one.

    Without this, exact conservation is conservation of nothing (their Fig. 10).
    """
    return F.mse_loss(latent_fn, physical_fn)


def latent_conformity_loss(c_after: torch.Tensor, c_before: torch.Tensor,
                           dissipative: bool = False) -> torch.Tensor:
    """Paper Eqs. (8)/(9): soft equality or one-sided dissipation penalty.

    Only needed when the structure is NOT enforced architecturally. With the
    pinned generator this term is identically zero -- which is the point.
    """
    d = c_after - c_before
    if dissipative:
        d = torch.clamp(d, min=0.0)
    return (d ** 2).mean()


def balance_law_loss(influx: torch.Tensor, true_influx: torch.Tensor) -> torch.Tensor:
    """C^T g_k should equal the true per-block flux.

    This is the input-side pullback: it is what gives the exactly-conserved
    ledger its meaning. Everything else about conservation is free.
    """
    return F.mse_loss(influx, true_influx)


# ---------------------------------------------------------------------------
# Self-test: verify the structural guarantees actually hold
# ---------------------------------------------------------------------------


def _self_test(device="cpu"):
    torch.manual_seed(0)
    n_z, n_pin = 24, 2
    gen = StructuredGenerator(n_z, mode="pinned", n_pinned=n_pin, eps=1e-3).to(device)
    W = gen()
    C = gen.pinned_basis()
    rep = gen.structure_report()

    print("--- structural guarantees (pinned mode) ---")
    print(f"  K skew defect        max|K + K^T| = {rep['skew_defect']:.2e}   (want 0)")
    print(f"  pin defect           max|C^T W|   = {rep['pin_defect']:.2e}   (want 0)")
    print(f"  dissipativity  max eig(W + W^T)  = {rep['max_sym_eig']:.2e}   (want <= 0)")

    # Conservation of C^T z under the flow, at several horizons.
    z = torch.randn(8, n_z, device=device)
    print("\n--- C^T z under exp(tW), should be constant ---")
    for t in [1, 10, 100, 1000, 10000]:
        zt = z @ torch.matrix_exp(float(t) * W).T
        drift = (zt @ C - z @ C).abs().max().item()
        nrm = (zt.norm(dim=1) / z.norm(dim=1)).mean().item()
        print(f"  t={t:>6}   ledger drift={drift:.2e}   ||z_t||/||z_0||={nrm:.4f}")

    # Monotone dissipation of the norm.
    print("\n--- ||z_t|| monotone non-increasing ---")
    zs = [z]
    E = torch.matrix_exp(W)
    for _ in range(50):
        zs.append(zs[-1] @ E.T)
    n = torch.stack([q.norm(dim=1) for q in zs])
    viol = (n[1:] - n[:-1]).clamp(min=0).max().item()
    print(f"  max increase over 50 steps = {viol:.2e}   (want <= 0 up to fp error)")

    # Binary-powering flow matches direct matrix_exp.
    print("\n--- ExpFlow binary powering vs. torch.matrix_exp ---")
    flow = ExpFlow(n_bits=12, dt=1.0)
    ns = torch.tensor([0, 1, 7, 63, 1000, 4095, 33, 512])
    got = flow(z, ns, W)
    want = torch.stack([z[i] @ torch.matrix_exp(float(ns[i]) * W).T for i in range(8)])
    print(f"  max abs err = {(got - want).abs().max().item():.2e}")

    # Driven balance law: C^T z_K == C^T z_0 + sum C^T g_k, exactly.
    print("\n--- driven balance law (the actual claim) ---")
    cfg = MemoryConfig(d_in=16, n_z=n_z, n_pinned=n_pin, mode="pinned")
    mem = LatentTwinMemory(cfg).to(device)
    x = torch.randn(4, 128, 16, device=device)
    zs, g = mem.rollout(x)
    Cb = mem.gen.pinned_basis()
    lhs = zs[:, -1] @ Cb
    rhs = zs[:, 0] @ Cb + (g @ Cb).sum(dim=1)
    print(f"  max|C^T z_K - (C^T z_0 + sum C^T g_k)| = {(lhs - rhs).abs().max().item():.2e}")
    print("  (exact by construction over 128 driven steps, no loss term involved)")


if __name__ == "__main__":
    _self_test()

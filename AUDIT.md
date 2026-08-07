# Reproducibility audit — latent-memory (Aug 2026)

Conducted as an independent audit-and-improve pass: 8 read-only audit agents
(code map, claims inventory, results forensics, math/implementation, statistics,
confounds, reproducibility/environment) plus one sequential GPU reproduction
lane, followed by verified implementation of the justified fixes. Every finding
below cites its evidence; every implemented change has a verification result.
Commit at audit start: `11e3bd3`. Commit at audit end: see `git log` (audit
batch + this report).

The full report (all seven sections, tables, and the claims audit) is the
canonical record delivered with the audit; this file is the committed copy.
Key artifacts produced by the audit:

- `results/adv_control.txt` — the retraction-control evidence, regenerated with
  a committed log (it previously existed in no log anywhere)
- `results/freeze_probe.txt` — the init0 freeze mechanism, measured
- `results/L0_forget.txt` — the forced-forgetting benchmark: Phase A necessity
  PASS (92.4% ≥ 20% pre-stated), Phase B first supporting evidence that hard
  selective protection beats a frozen parameter (pinned holds ledger 3/3 seeds
  and matches/beats the freeze on content; the freeze drops the ledger 1/3)
- `results/VERIFICATION.md` — extended from 46 to 58 recomputed claims, 0
  mismatches, now covering the dashboard's RDATA figure arrays, the regenerated
  control, the freeze probe, and the forget benchmark
- `results/ENV.md`, `requirements.lock.txt` (full pip freeze), `capture_env.py`,
  `freeze_probe.py`, `l0_forget.py`

Headline corrections applied to the package (each verified against logs):

1. "SNR 8–11 everywhere" (5 sites) → SNR is 1.3–11.2; the 1.3 sits at d*
   itself where the mean gradient crosses zero (results/L0_snr.txt row d=1e-3).
2. F1 quoted smoke-test ratios (6,365× / 237×) → committed-log values
   (89,222× / ~3,000×), with the correction noted in place.
3. "No initialization can rescue (attractor recaptures init0)" → rewritten to
   the measured outcome: init0 was NOT recaptured; under Adam the softplus(-20)
   gradients fall below the ε=1e-8 denominator floor (measured: √v̂=2.7e-9,
   median step 60× below sign-following, ~260k steps to reach the attractor vs
   a 3,000-step budget). Optimizer-dependence caveat stated.
4. Retraction-control numbers now have committed-log provenance
   (ε≈0 recal 0.0115 vs pinned 0.0116, ratio 0.99×; fixed-ε raw 0.3386;
   no-input floor 0.0705 vs analytic 0.0700).
5. Pre-registration scoped precisely: 4 tests committed before any stats data
   (9d4cccc); 2 scalar0 tests committed alongside only the first scalar row;
   stats seeds 0–2 replicate already-known frontier runs — fresh-seed-only
   statistic disclosed (7/7, p=0.0156).
6. Parity claims reworded ("no detectable difference at n=10", Wilcoxon
   p=0.064/0.131 disclosed, no equivalence test, metric saturation noted).
7. Parameter accounting disclosed and probe-verified (scalar 75 / diag 82 /
   ks 139 / pinned 139 live non-enc/dec params; ks exactly matched to pinned;
   frozen variants match their parents — capacity does not explain the gap).
8. Stale artifacts removed/annotated: truncated `results/L0_hard.txt` deleted,
   `l0_hard.py` marked SUPERSEDED; "one seed so far" and RUNNING/REPLICATING
   pills corrected; one-sided vs two-sided sign-floor fixed (0.25 at n=3);
   hero "every seed" → "median +1.000, min +0.999 (n=10)"; Fig. 2 marker
   relabeled (grid point 9.68e-4 vs converged 7.75e-4); literal BEL/TAB
   control-character corruption in the dashboard repaired.
9. Code hygiene: seed_check.py gained a __main__ guard (previously ran its
   15-minute sweep on import); l0_hard.py no longer mutates toy_ledger globals
   at import; ExpFlow now raises on offsets ≥ 2^n_bits (previously silently
   wrong modulo 2^n_bits); the three unused loss helpers are labeled as
   unexercised API stubs; scipy declared; sbatch gained the live-claim stages.

Reproduction status: frontier seed-0 scalar and pinned reproduce bit-exact at
printed precision on the recorded environment (torch 2.5.1+cu121, RTX 3060
Laptop, driver 546.33); the self-test reproduces at order-of-magnitude with
device-dependent rounding (README now says so); the committed verifier
reproduces its report byte-identically. Determinism is device-class-scoped
(per-device torch.Generator streams); CPU-only machines will not reproduce the
logged numbers, now documented in ENV.md.

Known limitations that remain open (in priority order):

- L0-Forget Phase B is n=3 (two-sided sign floor 0.25): run n=10 before any
  claim strengthening; also add diag0 and ks arms to it.
- The rescue suite's "3× budget does not rescue" rests on n=2 with one
  partially-rescued seed; "soft fixes fail" needs ≥5 seeds and a λ sweep.
- The mechanism probes (landscape, SNR) are single-mode single-seed conditional
  slices; generality rests on the 12-run convergence, stated as such.
- trivial_baseline.py / l05_anchor.py eval generators overlap the training
  stream (retracted-track diagnostics only; frontier/stats eval streams are
  disjoint, seed+7777).
- QR sign discontinuity in pinned_basis is real but unfixed by design: fixing
  it would break bit-exact reproduction of committed logs. Fix at the next
  clean experimental break.
- The d*·K_train scaling law, L1-count on real LM states, and any claim beyond
  the synthetic task family remain untested (compute-bounded).


## Hardening outcomes (post-audit runs; drivers committed before data)

1. `results/L0_forget_n10.txt` — guard reproduced committed scalar s0 bit-exact;
   5 arms x 10 seeds. Pinned: +1.000 [1.000,1.000], holds 10/10, wins the
   pre-registered paired ledger tests vs every alternative (10/10, p=0.002,
   vs scalar/diag0/ks; 9/10, p=0.021, vs scalar0). Freezes hold 9/10 at
   0.97-0.99. Content differences not significant. The n=3 "first evidence"
   upgrades to statistically supported on the ledger metric; the n=3
   freeze-failure impression (1/3) moderates to ~1/10.
2. `results/rescue_stats.txt` — REVISES two verdicts. "3x budget does not
   rescue" -> unreliable (2/5). "Soft fixes fail" -> refuted as stated:
   lambda=30 rescues 3/3 (skill +0.95, content 0.39); lambda<=3 fails 0/6.
   Stable conclusion: the retained direction's effective decay must reach ~0;
   the mechanism (freeze / projection / strong penalty) is secondary, with
   projection uniquely exact and the most reliable at n=10.
3. `results/scaling_dstar.txt` — REFUTES the d* ~ 1/K_train prediction at the
   pre-stated threshold (d*.K spread 5.6x > 4x over K in [32,256]); d* stays
   at 0.75-1.1e-3 (init/task-anchored; init=1e-3 confound acknowledged;
   STEPS fixed across K). Horizon-robust fact: under-retention at 16xK at
   every K tested. "Horizon-myopic" wording narrowed accordingly.

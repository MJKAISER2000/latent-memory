# latent-memory: what this project is, what it found, what's wrong with it

*One page, no spin. Every number below is recomputed from committed logs
(`results/VERIFICATION.md`, 64/64 checks). All experiments: one RTX 3060 laptop.*

## The idea

Physics simulators can be built so conservation laws hold *by construction* —
violating them is impossible, not just discouraged. We applied that to AI
memory: a small recurrent state whose transition matrix is constructed so one
designated subspace is exactly conserved (`CᵀW = 0`) while everything else is
free to decay. Under a stream of inputs this gives a **balance law** — the
protected "ledger" changes only by explicit, logged deposits (exact to ~1e-13).
The question: does a memory with hard guarantees beat one that merely *learns*
to remember?

## Main findings

**1. Standard learned-decay memories fail long-horizon retention because
training itself selects the failure.** A memory with a learnable forgetting
rate, trained on short sequences (64 steps) at standard settings, converges to
a decay rate (~8e-4) that is the *optimum of the training objective* — we
measured the loss landscape's interior minimum and the gradient pushing toward
it from both sides. That rate is invisible during training (gain ≈ 0.95 over
64 steps) and catastrophic at test length (gain ≈ 0.45 at 1024 steps). Result:
retention skill −6 vs a trivial baseline, 10/10 seeds, while the constrained
memory scores +1.000 (p = 0.002, pre-registered paired tests).

**2. The fix is making the retained direction's decay unreachable — the exact
mechanism matters less than we first claimed.** Three things work: freezing
the decay parameter at zero (works via Adam's ε floor — measured), the hard
projection, and *sufficiently strong* regularization (λ=30 rescues 3/3; our
earlier "soft fixes fail" claim was an artifact of testing only λ=3, which
fails, as does 3× training budget — 2/5). What reliably fails is the
practitioner-default configuration.

**3. When forgetting is genuinely necessary, the hard projection is the only
fully reliable option.** On our first benchmark, "never forget anything"
matched the projection — because that task never punished hoarding. We built a
benchmark where it provably does (never-forgetting costs 13× on current-content
accuracy), then compared. The projection holds the ledger at +1.000 on 10/10
seeds and wins pre-registered paired tests against every alternative,
including both freeze variants (p = 0.002–0.021), which drop the ledger ~1
seed in 10 and sit at 0.97–0.99 otherwise. Content accuracy differences are
not statistically significant. So the projection's edge is *exactness and
reliability*, not raw performance.

**4. One of our own predictions failed: the attractor does not track the
training horizon.** We predicted the learned decay would scale as ~1/K_train.
It doesn't — over an 8× range of training lengths it stays near ~1e-3
(pre-stated threshold refuted the law). The robust fact is the consequence,
not the mechanism's location: under-retention at 16× the training length
occurred at every training length we tested.

Also load-bearing: our first headline result ("29× better!") was an artifact
of a rigged baseline — caught by our own adversarial audit, retracted, and
kept in the repo as a case study, along with the audit protocol that caught it.

## Exact experimental process

**The memory.** State `z ∈ R^8`. Per input block: `z_{k+1} = exp(W)·z_k + Enc(x_k)`,
where `Enc` and `Dec` are 2-hidden-layer MLPs (width 64, SiLU). Ledger readout:
`M̂ = gain·⟨r, z⟩ + bias` (direction `r` learned; for the pinned arm it is tied to the
conserved direction `C`). The **only** thing that differs between arms is how the
transition generator `W` is parameterized (`A`, `B` are learned 8×8 matrices,
`s`, `d` learned decay parameters, softplus-positive, all initialized so decay = 1e-3):

| arm | W | can it protect one direction? |
|---|---|---|
| scalar | `(A−Aᵀ) − softplus(s)·I` | no — uniform decay |
| diag | `(A−Aᵀ) − diag(softplus(d))` | representable, must be learned |
| ks | `(A−Aᵀ) − BBᵀ − softplus(s)·I` | representable, must be learned |
| scalar0 / diag0 | same as scalar/diag, decay init at −20 (≈0, de-facto frozen under Adam) | protects everything |
| pinned | `P(A−Aᵀ)P − (PB)(PB)ᵀ − softplus(s)·P`, `P = I − CCᵀ` | yes, exactly, by construction |

Live parameter counts were probe-verified (scalar 75 / diag 82 / ks 139 / pinned 139
non-encoder params; encoder+decoder ≈13.6k identical everywhere; ks exactly matches pinned).

**The tasks.** Synthetic streams of K blocks. Each block carries a deposit
`a_k = flag_k · amount_k` (flag ~ Bernoulli(p), p drawn per-sequence from U(0.1, 0.5) so a
constant predictor can't win; amount ~ U(0,1); both are channels *inside* the block, so the
label `M_t = Σ a_k` is a nonlinear function of the input that no arm gets for free) plus
Gaussian distractor content. Two variants: the *frontier* task (32-dim blocks, 8-dim memory)
and the *forced-forgetting* task (8-dim blocks, content variance 1.0, reconstruction target =
the 6 non-deposit channels of the current block). For the latter we first **validated the
tension**: models with fixed decay swept over d ∈ {0, 1e-3, 1e-2, 0.1, 1} show
never-forgetting costs 13× on content (median 3.53 vs 0.27; pre-stated criterion ≥20%
improvement — passed at 92.4%), so retention and forgetting genuinely conflict.

**Training.** Every arm, identical recipe: Adam (lr 3e-3, default betas/ε), batch 64,
3000 steps (2000 on the forgetting task), LR halved every third of training, gradient-norm
clip 1.0, K_train = 64. Loss = MSE(ledger prediction, true running sum, at every prefix)
+ MSE(content reconstruction). Weights seeded by `torch.manual_seed(seed)`; data drawn from
a per-device `torch.Generator(seed+1)`.

**Evaluation.** K_test = 1024 (16× the training horizon), fresh generator (`seed+7777`,
disjoint from training; all arms at a given seed share the identical eval batch, making
paired tests valid), 512 sequences. Ledger metric: *skill* = 1 − MSE/MSE_const, where the
constant is the eval batch's own mean (an oracle baseline; 0 = no better than ignoring the
input, 1 = perfect). Content metric: reconstruction MSE at t = 64 and t = 1024. Diagnostics
per run: minimum decay eigenvalue and alignment of the slowest direction with the readout.

**Statistics.** 10 seeds per arm, paired by seed, exact two-sided sign tests. Comparisons
committed to git before the data existed (verifiable in history); where that was only
partially true for one earlier run, the report says so and gives the fresh-seed-only
statistic.

**Mechanism probes.** (a) *Landscape*: take a trained model, sweep its decay across 22
values, record train-horizon loss, test-horizon loss, and the gradient — locating the
interior train-loss minimum at d\* ≈ 9.7e-4, where training converged. (b) *SNR*: at each
decay value, 64 minibatch gradients — mean, spread, and sign; the mean gradient pushes decay
*up* below d\* and *down* above it (SNR 1.3–11.2, the dip exactly at d\*'s zero crossing).
(c) *Freeze probe*: per-step gradient magnitudes and Adam moment estimates for a frozen-zero
run — showing updates collapse because √v̂ ≈ 2.7e-9 sits below Adam's ε = 1e-8.
(d) *Scaling*: repeat training at K_train ∈ {32, 64, 128, 256} with K_test = 16×K_train.

**Process controls.** One-variable controls (the original headline died when changing a
single constant erased it); a no-input floor and an information floor bracketing every
metric; oracle affine recalibration on held-out batches to separate information loss from
miscalibration; parameter accounting by autograd probe; a bit-exact reproduction guard
before extending any experiment; environment capture (torch/CUDA/GPU/commit) per run; and
an independent verifier that re-parses every committed log and recomputes all 64 quantitative
claims in the documentation (0 mismatches). Results reproduce bit-exactly on the recorded
GPU; data streams are device-class-specific (CPU runs will differ — documented).

## Drawbacks

- **All evidence is synthetic.** Small toy tasks (8-dim inputs, 8-dim memory)
  built to isolate the effect. Nothing yet shows this matters inside a real
  language model; that test (assets built, not run) could come back empty.
- **The guarantee only covers what you can name.** The ledger protects
  quantities you designate in advance and route deposits to. It is no help for
  "remember whatever turns out to matter later."
- **Cheap alternatives are often good enough.** A frozen parameter (one line)
  or strong regularization gets ~97–99% of the benefit when forgetting
  pressure is mild. The projection earns its keep only when retention must be
  exact, auditable, or reliable under real forgetting pressure.
- **The mechanism story is narrower than it sounds.** "Training trades
  retention away" is measured only on one task family, mostly via 1-D probes
  of one model; what *sets* the attractor's location is unresolved (init vs
  task constants — our sweep can't separate them).
- **Effect sizes at the margin are small.** Against the freezes, the
  projection's ledger advantage is +1.000 vs ~0.97–0.99 and 10/10 vs 9/10
  reliability — statistically significant, but a practitioner may reasonably
  not care.

## Status

Repo: github.com/mjkaiser2000/latent-memory. Interactive dashboard with the
full research log, per-claim provenance links, figures, and the retraction:
`dashboard.html`. Next real test: the same comparison over a frozen language
model's hidden states.

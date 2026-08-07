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

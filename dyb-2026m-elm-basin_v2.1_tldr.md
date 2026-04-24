# TL;DR Summaries (5 Perspectives)

Companion to: *Valid and False Snapping in EML Expression Trees: The Basin Selection Problem*
Daniyel Yaacov Bilar, Chokmah LLC
v2.1, April 24 2026

---

## For the Expert Researcher (ML / Symbolic Regression)

Three-phase temperature annealing guarantees vertex commitment in EML expression trees, but the real bottleneck is basin selection, not commitment. Valid snap rates peak at representational depth for ln(x) (18/20 at d=4) but improve with over-depth for exp(x) (5/20 at d=2 to 17/20 at d=4) because extra gates provide escape routes from the dominant competing basin eml(x,x). The paper's key methodological contribution is distinguishing valid snaps (correct form plus post-snap MAE < 0.01) from false snaps (vertex commitment but wrong form), revealing a failure mode invisible to prior work that only checked selector entropy. Odrzywolek's own SI warm-start evidence (Table S7: 100% recovery at d=5 and d=6 from noised correct initializations) independently confirms that the correct solutions are stable attractors, isolating the loss surface as the bottleneck. The continuous relaxation can interpolate between vertex configurations in ways no snapped tree can match, so pre-snap loss is a misleading proxy for correctness.

## For the Practitioner (Engineer / Data Scientist)

If you are using soft-to-hard snapping for neural symbolic regression, stop reporting pre-snap loss as your success metric. This paper shows that even when every selector snaps cleanly to a vertex (100% commitment), the resulting symbolic expression can be completely wrong. For the exp(x) target, 75% of seeds at minimal depth collapse into eml(x,x), a valid discrete form with MAE 0.688, not the target. The fix: always evaluate post-snap loss on your actual symbolic expression. For your own architectures, consider over-parameterizing depth to escape dominant local minima, but beware that extra depth eventually hurts once the escape mechanism is no longer needed.

## For the General Public

Scientists trained AI models to discover exact math formulas (like "what is the formula for e^x?") by letting the AI softly mix possibilities, then "snapping" to definite choices. They found a hidden problem: the AI always snaps to a definite answer, but that answer is often wrong, even when the AI looks like it is doing great during training. It is like a student who confidently circles an answer on a multiple-choice test, but picked the wrong letter. The study shows that for one formula (natural log), the AI gets it right 90% of the time at the right complexity level. But for another (exponential), it only gets it right 25% of the time unless you give the AI extra room to work with, then it jumps to 85%. The big lesson: checking that the AI "made a decision" is not enough; you must check if the decision was correct.

## For the Skeptic

Is this just another "neural nets are brittle" paper with a toy problem? Not quite. The contribution is sharper: prior work conflated two distinct failure modes and only reported the easy one. The authors run 238 training runs with deterministic seeding, report exact counts (not vague "trends"), and falsify their own criterion by showing sqrt(x) fails everywhere as predicted. The sample size (20 seeds per cell) is admitted as coarse, and they do not tune their three-phase schedule. The real weakness: one optimizer (Adam), one loss (MAE), one architecture variant. The v2.1 revision adds a cross-check against Odrzywolek's independent pipeline (different targets, different schedule, same basin-selection story), which reduces but does not eliminate the one-optimizer concern. The "over-depth helps exp(x)" finding is intriguing but may not generalize to non-tree structures. The paper is honest about these limits and correctly frames its scope as methodological (defining valid vs false snaps) rather than algorithmic (solving basin selection).

## For the Decision-Maker (Funding / Strategy)

Investment signal: The EML architecture for symbolic regression has a critical gap between "looks like it works" and "actually works." Current training protocols guarantee discrete outputs but not correct ones. This creates both risk and opportunity: any group that solves basin selection (via better initialization, curriculum training, or warm-starting) will have a genuine technical moat. Short-term: do not deploy EML-based symbolic regression without post-snap verification. Medium-term: fund research on phase-1 basin selection methods; the annealing protocol itself is not the bottleneck. Watch for: extensions to non-tree architectures, multi-function curriculum training, and hybrid neuro-symbolic approaches that use EML as one component among many.

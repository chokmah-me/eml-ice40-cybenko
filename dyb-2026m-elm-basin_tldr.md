# TL;DR Summaries (5 Perspectives)

Companion to: *Valid and False Snapping in EML Expression Trees: The Basin Selection Problem*
Daniyel Yaacov Bilar, Chokmah LLC
v2.3, June 12 2026

---

## For the Expert Researcher (ML / Symbolic Regression)

Three-phase temperature annealing guarantees vertex commitment in EML expression trees, but the real bottleneck is basin selection, not commitment. Valid snap rates peak at representational depth for ln(x) (18/20 at d=4) but improve with over-depth for exp(x) (5/20 at d=2 to 17/20 at d=4), where the extra gates correlate with escape from the dominant competing basin eml(x,x); the mechanism behind that improvement is observed but not established. The paper's key methodological contribution is distinguishing valid snaps (correct form plus post-snap MAE < 0.01) from false snaps (vertex commitment but wrong form), revealing a failure mode invisible to commitment-only success criteria. Odrzywolek's own SI warm-start evidence (Table S7: 100% recovery at d=5 and d=6 from noised correct initializations) independently confirms that the correct solutions are stable attractors, isolating the loss surface as the bottleneck. The continuous relaxation can interpolate between vertex configurations in ways no snapped tree can match, so pre-snap loss is a misleading proxy for correctness. A companion technical note in the same repository shows that targeted warm-start and top-aligned curriculum initialization recover the difficult cells (exp d=3, ln d=5) at 20/20.

## For the Practitioner (Engineer / Data Scientist)

If you are using soft-to-hard snapping for neural symbolic regression, stop reporting pre-snap loss as your success metric. This paper shows that even when every selector snaps cleanly to a vertex (100% commitment), the resulting symbolic expression can be completely wrong. For the exp(x) target, 75% of seeds at minimal depth collapse into eml(x,x), a valid discrete form with MAE 0.688, not the target. The fix: always evaluate post-snap loss on your actual symbolic expression. For your own architectures, over-parameterizing depth was associated with escaping dominant local minima in these experiments, but beware that extra depth eventually hurts once the dominant competitor is no longer the binding constraint. If you can guess or pre-train the target structure, the companion warm-start/curriculum note shows initialization alone takes the hard cells from 25-35% to 100%.

## For the General Public

Scientists trained AI models to discover exact math formulas (like "what is the formula for e^x?") by letting the AI softly mix possibilities, then "snapping" to definite choices. They found a hidden problem: the AI always snaps to a definite answer, but that answer is often wrong, even when the AI looks like it is doing great during training. It is like a student who confidently circles an answer on a multiple-choice test, but picked the wrong letter. The study shows that for one formula (natural log), the AI gets it right 90% of the time at the right complexity level. But for another (exponential), it only gets it right 25% of the time, while versions of the AI given extra room to work with got it right 85% of the time. The big lesson: checking that the AI "made a decision" is not enough; you must check if the decision was correct.

## For the Skeptic

Is this just another "neural nets are brittle" paper with a toy problem? Not quite. The contribution is sharper: prior work reported commitment-based success criteria; this paper adds a symbolic-correctness criterion and shows the two come apart systematically. The authors run 238 training runs with deterministic seeding, report exact counts (not vague "trends"), and falsify their own criterion by showing sqrt(x) fails everywhere as predicted. The sample size (20 seeds per cell) is admitted as coarse, and they do not tune their three-phase schedule. The real weakness: one optimizer (Adam), one loss (MAE), one architecture variant. The v2.1 revision adds a cross-check against Odrzywolek's independent pipeline (different targets, different schedule, same basin-selection story), which reduces but does not eliminate the one-optimizer concern. The v2.2-v2.3 revisions, responding to external review, removed a mechanism claim ("extra gates provide escape routes") that the experiments did not establish; the over-depth effect for exp(x) is now reported as an observation, and may not generalize to non-tree structures. The paper is honest about these limits and correctly frames its scope as methodological (defining valid vs false snaps) rather than algorithmic (solving basin selection).

## For the Decision-Maker (Funding / Strategy)

Investment signal: The EML architecture for symbolic regression has a critical gap between "looks like it works" and "actually works." Current training protocols guarantee discrete outputs but not correct ones. This creates both risk and opportunity: basin selection is addressable, and the same group's follow-up note demonstrates that cheap initialization-level interventions (warm-start, top-aligned curriculum) already recover the tested hard cells at 100%, suggesting the moat lies in generalizing such methods beyond known targets. Short-term: do not deploy EML-based symbolic regression without post-snap verification. Medium-term: fund research on phase-1 basin selection methods for targets whose structure is not known in advance; the annealing protocol itself is not the bottleneck. Watch for: extensions to non-tree (subtree-sharing/DAG) architectures, multi-function curriculum training, and hybrid neuro-symbolic approaches that use EML as one component among many.

---

## AI Utilization Statement

This statement describes the use of AI systems in producing this work, in keeping with emerging norms for AI disclosure in scientific publishing (cf. ACM, Nature, Science 2024-2025 author guidelines).

The author originated the thesis, supplied source materials, and made all editorial decisions. Claude (Anthropic; Haiku 4.5, Opus 4.5 and 4.7 via the claude.ai web interface) was used across approximately three dozen conversational sessions during the research and writing of this work. Several proprietary research prompts, as well as the commercial NovaKit Utilities v3 "Cognition-as-Utility" toolkit, were used in a series of GAN-like adversarial critical reviews discrediting, sharpening, and advising on revision of this work. TL;DR summaries were generated by Kimi 2.6. Feedback was provided by DeepSeek 3 and 4.

The released code, CSV, and figures are fully reproducible from the released artifacts. No AI system is needed to re-run the experiments or regenerate the figures. The paper's empirical claims stand or fall on the released data alone. AI contribution was to the research process and writing, not to the underlying scientific result.

The human author (Daniyel Yaacov Bilar) takes full responsibility for the scientific content, factual accuracy, and framing of this work. Errors, if any, are the author's.

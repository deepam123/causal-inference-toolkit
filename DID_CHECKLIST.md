# Diff-in-Differences (DiD) — Pre-Analysis Checklist

A set of fundamental questions to ask **before** applying DiD to any study.

---

## 1. What is my unit of treatment?
Who or what actually received the intervention — a person, location, queue, team?
This defines your granularity. Everything flows from this.

> In this study: **location** received Lean Ops, so we work at location-month level.

---

## 2. Do I have a clear before/after?
Is there a well-defined moment the intervention started?
Fuzzy rollouts (some locations in month 4, others in month 7) break standard DiD.

> In this study: **Intervention Month = 7**. Clean cutoff.

---

## 3. Do I have a clean control group?
Are my control units truly untreated?
If control locations were also quietly doing process improvements, your "control" is contaminated.

---

## 4. Parallel trends — would treated and control have moved together without the intervention?
This is the **big one**. You can never prove it, but you can check:
- Did they trend similarly *before* the intervention?
- Are they fundamentally different types of units?

> Violation of parallel trends = biased DiD estimate. No workaround exists within DiD itself.

---

## 5. Is there spillover?
Did treated units affect control units?
Example: if agents moved between locations, Lean Ops knowledge spreads to control groups — contaminating your comparison.

---

## 6. Are there confounding events?
Did anything else happen simultaneously with the intervention that only affected treated locations?
Examples: new product launch, hiring wave, system change.

---

## 7. How long is my post period?
- Too short → you might miss delayed effects
- Too long → other confounding events start accumulating

---

## 8. Do I have enough treated units?
DiD needs enough treated units to distinguish signal from noise.
- 1 treated unit → unreliable, consider **Synthetic Control** instead
- General rule of thumb: at least 10–20 treated units for DiD to be trustworthy

---

## Key Reminder: Granularity vs. Noise Tradeoff
Going more granular (e.g. queue × location × month) gives richer insights but:
- Increases noise per cell (fewer observations)
- Harder to satisfy parallel trends
- Higher risk of false positives

**Best practice:** Run aggregate DiD first, then subgroup analysis by queue/product to check for heterogeneous effects.

---

## Further Reading
- Angrist & Pischke — *Mostly Harmless Econometrics*
- Callaway & Sant'Anna (2021) — DiD with multiple treatment periods
- Roth et al. (2023) — What's trending in DiD research


# causal-inference-toolkit

A working toolkit for causal inference and experimentation, built around one running case study (a BPO call center piloting a "Lean Floor Management" program) so every method can be checked against real outcomes.

Full scenario, hypotheses, and the "why simple averages won't work" walkthrough live in **STUDY_GUIDE.md**. This README is the map: what's built, what's next, and how to pick the right method for a given question.

## Method Landscape

Not every causal question calls for the same tool. **Work through this decision tree top to bottom** — the first row that matches your situation is your method:

![Decision Tree](decision_tree.svg)

## Methods Status & Details

| Method | Status | Best For | File |
|--------|--------|----------|------|
| Difference-in-Differences | ✅ Built | A comparable group didn't get treated, and you have before/after data | `methods/diff_in_diff.py`, `DID_CHECKLIST.md` |
| Propensity Score Matching | 🔜 Next up | Non-random treatment, but you can measure what drove selection | `methods/propensity_matching.py` |
| Bayesian Hierarchical Model | 🔜 Planned | Small sample of treated units, need honest uncertainty | `methods/bayesian_experiment.py` |
| Synthetic Control | 🔜 Planned | Only one (or very few) treated units, no comparable group | `methods/synthetic_control.py` |
| Regression Discontinuity | 🔜 Planned | Treatment assigned by a hard cutoff (score, date, threshold) | `methods/regression_discontinuity.py` |
| Instrumental Variables | 🔜 Planned | Confounded by something unmeasured, but a valid "nudge" variable exists | `methods/instrumental_variables.py` |
| A/B Testing Foundations | ✅ Built | You control randomization — the RCT case | `experimentation/01_ab_testing_foundations.py` |
| CUPED / Variance Reduction | ✅ Built | Tightening confidence intervals on a randomized test you're already running | `experimentation/02_cuped_variance_reduction.py` |

Each ✅ script runs against the same synthetic dataset and reports its estimate next to the true simulated effect, so the accuracy is checkable rather than asserted. Each 🔜 method gets its own validation suite.

## Quick Start

```bash
pip install -r requirements.txt

python data/generate_data.py          # builds the synthetic call-center dataset

python methods/diff_in_diff.py        # run any method script the same way
```

## Repo Structure

```
causal-inference-toolkit/
├── data/
│   └── generate_data.py              — synthetic dataset with known ground-truth effects
├── methods/
│   └── diff_in_diff.py               — DiD: 2x2, regression, log, parallel-trends check, placebo test
├── experimentation/
│   ├── 01_ab_testing_foundations.py   — sample size, peeking, SRM, multiple testing
│   └── 02_cuped_variance_reduction.py — CUPED variance reduction on a randomized test
├── STUDY_GUIDE.md                    — the business scenario, hypotheses, and why each method is needed
├── DID_CHECKLIST.md                  — pre-analysis checklist before applying DiD
└── requirements.txt
```

## Why the Ground-Truth Approach

Every dataset here is generated with a known true effect baked in (see `data/generate_data.py`). Each method's script prints its estimate right next to that true value. This makes the toolkit doubly useful:
- **For learning**: You see what each method *actually recovers*, not just theory
- **For validation**: Before you deploy a method on real data, you can trust it here

causal-inference-toolkit
A working toolkit for causal inference and experimentation, built around one running case study (a BPO call center piloting a "Lean Floor Management" program) so every method can be checked against a known ground-truth effect instead of taken on faith.

Full scenario, hypotheses, and the "why simple averages won't work" walkthrough live in STUDY_GUIDE.md. This README is the map: what's built, what's next, and how to pick the right method for a given situation.
Method landscape
Not every causal question calls for the same tool. This is the logic used to pick one:

flowchart TD

    A[Do you have a randomized<br/>treatment/control split?] -->|Yes| A1[Standard A/B test analysis<br/>— see experimentation/]

    A -->|No| B{Is treatment assigned by a<br/>known cutoff on a continuous variable?}

    B -->|Yes| RDD[Regression Discontinuity Design]

    B -->|No| C{Do you have the same units<br/>before AND after, with an<br/>untreated comparison group?}

    C -->|Yes| C1{Do treated & control look similar<br/>in level/trend before treatment?}

    C1 -->|Yes, roughly parallel| DID[Difference-in-Differences]

    C1 -->|No, or only one treated unit| SCM[Synthetic Control]

    C -->|No control group at all| ITS[Interrupted Time Series]

    C -->|No time dimension| D{Rich pre-treatment covariates<br/>that explain who got treated?}

    D -->|Yes| PSM[Propensity Score Matching]

    D -->|No, but something nudges<br/>treatment for an unrelated reason| IV[Instrumental Variables]

Method
Status
Best for
File
Difference-in-Differences
✅ Built
A comparable group didn't get treated, and you have before/after data
methods/diff_in_diff.py, DID_CHECKLIST.md
Propensity Score Matching
🔜 Next up
Non-random treatment, but you can measure what drove selection
methods/propensity_matching.py
Bayesian Hierarchical Model
🔜 Planned
Small sample of treated units, need honest uncertainty
methods/bayesian_experiment.py
Synthetic Control
🔜 Planned
Only one (or very few) treated units, no comparable group
methods/synthetic_control.py
Regression Discontinuity
🔜 Planned
Treatment assigned by a hard cutoff (score, date, threshold)
methods/regression_discontinuity.py
Instrumental Variables
🔜 Planned
Confounded by something unmeasured, but a valid "nudge" variable exists
methods/instrumental_variables.py
A/B Testing Foundations
✅ Built
You control randomization — the RCT case
experimentation/01_ab_testing_foundations.py
CUPED / Variance Reduction
✅ Built
Tightening confidence intervals on a randomized test you're already running
experimentation/02_cuped_variance_reduction.py


Each ✅ script runs against the same synthetic dataset and reports its estimate next to the true simulated effect, so the accuracy is checkable rather than asserted. Each 🔜 method gets its own PR when it's built, following the same pattern.
Quick start
pip install -r requirements.txt

python data/generate_data.py          # builds the synthetic call-center dataset

python methods/diff_in_diff.py        # run any method script the same way
Repo structure
causal-inference-toolkit/

├── data/

│   └── generate_data.py          — synthetic dataset with known ground-truth effects

├── methods/

│   └── diff_in_diff.py           — DiD: 2x2, regression, log, parallel-trends check, placebo test

├── experimentation/

│   ├── 01_ab_testing_foundations.py   — sample size, peeking, SRM, multiple testing

│   └── 02_cuped_variance_reduction.py — CUPED variance reduction on a randomized test

├── STUDY_GUIDE.md                — the business scenario, hypotheses, and why each method is needed

├── DID_CHECKLIST.md              — pre-analysis checklist before applying DiD

└── requirements.txt
Why the ground-truth approach
Every dataset here is generated with a known true effect baked in (see data/generate_data.py). Each method's script prints its estimate right next to that true value. This makes the toolkit double as a sanity-check harness: if a method can't recover a known answer on clean synthetic data, it has no business being trusted on messy real data.


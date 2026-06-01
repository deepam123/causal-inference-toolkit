  # Lean Operations Study — Complete Guide
## Causal Inference Toolkit | deepam123/causal-inference-toolkit

---

## 🏭 The Business Problem

A BPO call center network (40 locations, ~50K experts) is piloting a
**Lean Floor Management Model** at 15 locations. The model introduces:
- Structured daily huddles
- Real-time floor coaching
- Visual management boards
- Tiered escalation protocols

**The executive question:** Does lean floor management actually improve
customer outcomes — or are we just seeing noise?

---

## 📊 Outcome Variables

| Variable | Type | Baseline | What It Measures |
|---|---|---|---|
| `aht` | Continuous (minutes) | ~13 min | How long each call takes |
| `tnps` | Continuous (0–10) | ~7.3 | Customer satisfaction on the call |
| `issue_resolution` | Binary (0/1) | ~72% | Was the issue resolved first contact? |
| `escalation_rate` | Binary (0/1) | ~18% | Was the call escalated to a supervisor? |
| `repeat_contact_7d` | Binary (0/1) | ~22% | Did the customer call back within 7 days? |

### True Effects (Ground Truth — hidden in real life, known here for validation)

| Metric | True Lean Effect |
|---|---|
| AHT | **−2.5 minutes** |
| tNPS | **+0.8 points** |
| Issue Resolution | **+7 percentage points** |
| Escalation Rate | **−5 percentage points** |
| Repeat Contact | **−6 percentage points** |

> 💡 Because we generated the data, we know the truth. The goal of each
> causal method is to **recover these numbers from noisy data.**
> If our methods work, estimates should be close to these values.

---

## 🔬 Hypotheses

### Primary Hypothesis
> **H1:** Locations that adopted lean floor management will show
> statistically significant improvements across all 5 outcome metrics
> compared to control locations, after controlling for confounders.

### Secondary Hypotheses

| # | Hypothesis | Why It Matters |
|---|---|---|
| H2 | Lean effect on AHT will be larger at locations with higher % of complex queues (tech support, billing) | Tests if lean coaching matters more where calls are harder |
| H3 | SMB-heavy locations will show greater tNPS improvement under lean | SMB customers have higher expectations; floor coaching may close the gap |
| H4 | Locations with higher expert tenure will have smaller lean effect on AHT | Experienced experts already handle calls efficiently; less room to improve |
| H5 | Repeat contact reduction will be mediated by issue resolution improvement | If lean improves resolution, repeat contacts should fall as a downstream effect |

---

## 🚧 Challenges (Why Simple Averages Won't Work)

### Challenge 1: Queue Mix Confounding
**Problem:** Treated locations randomly ended up with different queue
mixes than control locations. A treated location heavy on tech support
(AHT = 18 min) will look worse than a control location heavy on account
setup (AHT = 8 min) — even if lean is working perfectly.

**Solution:** Control for queue mix proportions in all models.

```
Naive comparison:          Lean AHT = 14.2 min, Control = 11.8 min
                           → "Lean made things WORSE!" (wrong)

After controlling queues:  Lean AHT = 11.5 min, Control = 13.8 min
                           → "Lean improved AHT by 2.3 min" (closer to truth)
```

### Challenge 2: Non-Random Treatment Assignment
**Problem:** Larger locations were slightly more likely to be selected
for lean. Larger locations also tend to have more resources, better
infrastructure, and higher baseline performance. This creates selection
bias — treated locations were already on a better trajectory.

**Solution:** Propensity Score Matching — match each treated location
to a control location with similar size, tenure, queue mix, and segment.

### Challenge 3: Time Trends
**Problem:** The entire industry was improving over time (gradual
efficiency gains, product improvements). If we just compare pre vs post
for lean locations, we'll attribute industry-wide gains to lean.

**Solution:** Diff-in-Diff — subtract out the trend seen in control
locations. Only the *extra* improvement in treated locations counts.

```
Treated improvement (pre→post):   −3.1 min AHT
Control improvement (pre→post):   −0.6 min AHT (industry trend)
DiD estimate:                      −2.5 min AHT ← the true lean effect
```

### Challenge 4: Uncertainty & Small Location Counts
**Problem:** We only have 40 locations. With small N, frequentist
estimates can be noisy and confidence intervals wide. A single outlier
location can swing the result.

**Solution:** Bayesian Hierarchical Model — borrows strength across
locations, shrinks outlier estimates toward the group mean, and gives
us a full probability distribution over the effect size instead of just
a point estimate + p-value.

### Challenge 5: No Perfect Counterfactual
**Problem:** We can't observe what a lean location *would have looked
like* without lean. We need to construct a synthetic version.

**Solution:** Synthetic Control — build a weighted combination of
control locations that closely matches each treated location's
pre-intervention trajectory. Then compare post-intervention divergence.

---

## 🛠️ The Four Methods — When & Why

### Method 1: Difference-in-Differences (DiD)
**Best for:** Estimating average treatment effect across all locations

**Logic:**
```
Effect = (Treated_Post − Treated_Pre) − (Control_Post − Control_Pre)
```
**Key assumption:** Parallel trends — treated and control locations
were moving in the same direction before the intervention.

**What it tells you:** "On average, lean reduced AHT by X minutes,
controlling for time trends."

---

### Method 2: Propensity Score Matching (PSM)
**Best for:** Removing selection bias from non-random treatment assignment

**Logic:**
1. Build a model predicting P(treated) from location characteristics
   (size, tenure, queue mix, segment mix)
2. Match each treated location to its most similar control location
3. Compare outcomes only within matched pairs

**What it tells you:** "Comparing apples to apples — lean locations
vs identical-profile non-lean locations."

---

### Method 3: Bayesian Hierarchical Model
**Best for:** Estimating effect with uncertainty, especially with small N

**Logic:**
- Each location gets its own effect estimate
- A shared prior pulls all estimates toward a group mean
- Output is a full posterior distribution: "We are 95% confident the
  true AHT reduction is between −1.8 and −3.2 minutes"

**What it tells you:** "Not just the effect, but how confident we
should be — and whether variation across locations is real or noise."

---

### Method 4: Synthetic Control
**Best for:** When you want to visualize the counterfactual trajectory

**Logic:**
1. Find a weighted combination of control locations that matches the
   treated location's pre-intervention trend perfectly
2. Project that synthetic trend forward into the post period
3. The gap between actual and synthetic = the treatment effect

**What it tells you:** "Here's what Location X would have looked like
without lean — and here's the divergence after rollout."

---

## 📁 Project File Structure

```
causal-inference-toolkit/
├── data/
│   ├── generate_data.py          ← YOU ARE HERE (Step 1 ✓)
│   ├── call_center_data.csv      ← generated output
│   └── location_profiles.csv     ← generated output
│
├── methods/
│   ├── diff_in_diff.py           ← Step 2 (next)
│   ├── propensity_matching.py    ← Step 3
│   ├── bayesian_experiment.py    ← Step 4
│   └── synthetic_control.py      ← Step 5
│
├── notebooks/
│   └── causal_inference_walkthrough.ipynb   ← Step 6 (ties it all together)
│
├── requirements.txt              ← Step 2 (alongside DiD)
├── STUDY_GUIDE.md                ← THIS FILE
└── README.md                     ← Step 6 (update at end)
```

---

## 🗓️ Build Order & What You'll Learn at Each Step

| Step | File | Method | Key Learning |
|---|---|---|---|
| ✅ 1 | `data/generate_data.py` | Data Generation | Confounders, synthetic data design |
| ▶️ 2 | `methods/diff_in_diff.py` | DiD | Parallel trends, time fixed effects |
| 3 | `methods/propensity_matching.py` | PSM | Selection bias, matching estimators |
| 4 | `methods/bayesian_experiment.py` | Bayesian | Priors, posteriors, credible intervals |
| 5 | `methods/synthetic_control.py` | Synth Control | Counterfactual construction |
| 6 | `notebooks/walkthrough.ipynb` | All methods | End-to-end narrative + visualization |

---

## ✅ How You'll Know It's Working

At the end of each method, you'll see an output like:

```
── DiD Results ──────────────────────────────────────
Metric              Estimate    True Effect   Δ Error
aht                 -2.43       -2.50         0.07  ✓
tnps                +0.76       +0.80         0.04  ✓
issue_resolution    +0.068      +0.070        0.002 ✓
escalation_rate     -0.048      -0.050        0.002 ✓
repeat_contact_7d   -0.057      -0.060        0.003 ✓
```

If estimates are close to true effects → method is working.
If they're far off → we'll diagnose why (usually a missing confounder).

---

*Built with Claude Code · github.com/deepam123/causal-inference-toolkit*

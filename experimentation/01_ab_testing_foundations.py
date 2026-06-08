# =============================================================================
# DAY 1 LAB: A/B Testing Foundations
# Experimentation & Causal Inference Interview Prep
# =============================================================================
# Industry Context: Netflix runs 250+ A/B tests simultaneously. Airbnb's
# experimentation platform (ERF) runs thousands of experiments per year.
# Getting the fundamentals right is non-negotiable at these companies.
# =============================================================================

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from statsmodels.stats.power import TTestIndPower
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

print("=" * 70)
print("DAY 1: A/B TESTING FOUNDATIONS")
print("=" * 70)

# =============================================================================
# SECTION 1: THE BASICS — Simulating a Clean A/B Test
# =============================================================================
print("\n📘 SECTION 1: Simulating a Clean A/B Test")
print("-" * 50)
print("""
Industry Context:
  Netflix tests everything — thumbnails, recommendation algorithms, UI layouts.
  A typical test: does a new homepage layout improve play rate?
  
  Our simulation: Does a new checkout flow improve conversion rate?
  - Control: 12% baseline conversion
  - Treatment: 14% conversion (true lift of ~2pp)
""")

n_control = 5000
n_treatment = 5000
p_control = 0.12
p_treatment = 0.14

control = np.random.binomial(1, p_control, n_control)
treatment = np.random.binomial(1, p_treatment, n_treatment)

conv_control = control.mean()
conv_treatment = treatment.mean()
observed_lift = (conv_treatment - conv_control) / conv_control * 100

t_stat, p_value = stats.ttest_ind(control, treatment)
ci_low, ci_high = stats.t.interval(
    0.95,
    df=n_control + n_treatment - 2,
    loc=conv_treatment - conv_control,
    scale=stats.sem(np.concatenate([treatment - control]))
)

print(f"  Control conversion:   {conv_control:.3f} ({conv_control*100:.1f}%)")
print(f"  Treatment conversion: {conv_treatment:.3f} ({conv_treatment*100:.1f}%)")
print(f"  Observed lift:        {observed_lift:.1f}%")
print(f"  p-value:              {p_value:.4f}")
print(f"  95% CI on difference: ({ci_low:.4f}, {ci_high:.4f})")
print(f"  Statistically significant: {'✅ YES' if p_value < 0.05 else '❌ NO'}")

# =============================================================================
# SECTION 2: SAMPLE SIZE CALCULATION
# =============================================================================
print("\n📘 SECTION 2: Sample Size Calculation")
print("-" * 50)
print("""
Interview Q: "How do you decide how long to run an experiment?"
Answer: You calculate required sample size BEFORE starting, based on:
  - Baseline conversion rate
  - Minimum Detectable Effect (MDE) — smallest lift worth caring about
  - Statistical power (1 - β), typically 0.80
  - Significance level (α), typically 0.05

Airbnb lesson: Many teams run underpowered tests and call them too early.
This leads to false negatives (missing real effects).
""")

analysis = TTestIndPower()

scenarios = [
    {"mde": 0.01, "label": "1pp lift (small)"},
    {"mde": 0.02, "label": "2pp lift (medium)"},
    {"mde": 0.05, "label": "5pp lift (large)"},
]

print(f"  {'MDE':<20} {'Required n/group':<20} {'Total n':<15} {'Days @ 1K/day'}")
print(f"  {'-'*65}")
for s in scenarios:
    effect_size = s["mde"] / np.sqrt(p_control * (1 - p_control))
    n = analysis.solve_power(effect_size=effect_size, alpha=0.05, power=0.80)
    print(f"  {s['label']:<20} {int(n):<20,} {int(n*2):<15,} {int(n*2/1000)} days")

print("""
  ⚠️  PITFALL: Teams often say "let's run it for 2 weeks" without
  checking if 2 weeks gives enough power. Always calculate first!
""")

# =============================================================================
# SECTION 3: PITFALL #1 — PEEKING (Early Stopping)
# =============================================================================
print("\n📘 SECTION 3: PITFALL #1 — Peeking at Results")
print("-" * 50)
print("""
Industry Context:
  This is one of the most common mistakes at tech companies.
  A PM checks results daily and stops the test when p < 0.05.
  This inflates false positive rate dramatically.
  
  Netflix/Airbnb solution: Sequential testing (always-valid p-values)
  or pre-committing to a fixed sample size.
""")

# Simulate peeking: null hypothesis is TRUE (no real effect)
# Both groups have same conversion rate — any "significance" is false positive
n_simulations = 1000
n_total = 2000
checks = list(range(100, n_total + 1, 100))

false_positive_rates_peeking = []
false_positive_rates_fixed = []

for sim in range(n_simulations):
    ctrl = np.random.binomial(1, 0.12, n_total)
    trt = np.random.binomial(1, 0.12, n_total)  # same rate — null is TRUE
    
    # Peeking: stop as soon as p < 0.05
    peeked_significant = False
    for check in checks:
        _, p = stats.ttest_ind(ctrl[:check], trt[:check])
        if p < 0.05:
            peeked_significant = True
            break
    false_positive_rates_peeking.append(peeked_significant)
    
    # Fixed: only check at the end
    _, p_final = stats.ttest_ind(ctrl, trt)
    false_positive_rates_fixed.append(p_final < 0.05)

fpr_peek = np.mean(false_positive_rates_peeking)
fpr_fixed = np.mean(false_positive_rates_fixed)

print(f"  Under null hypothesis (no true effect):")
print(f"  False positive rate — fixed sample:  {fpr_fixed:.1%} (expected: ~5%)")
print(f"  False positive rate — peeking:        {fpr_peek:.1%} (should be ~5%, but isn't!)")
print(f"""
  ⚠️  PITFALL: Peeking inflates false positive rate to ~{fpr_peek:.0%}!
  You'd ship a broken feature 1 in {int(1/fpr_peek)} times instead of 1 in 20.
  
  ✅ FIX OPTIONS:
     1. Pre-commit to sample size, don't peek
     2. Use Bonferroni correction for multiple looks
     3. Sequential testing (e.g. mSPRT) — used by Airbnb & Netflix
     4. Bayesian testing with explicit stopping rules
""")

# =============================================================================
# SECTION 4: PITFALL #2 — SAMPLE RATIO MISMATCH (SRM)
# =============================================================================
print("\n📘 SECTION 4: PITFALL #2 — Sample Ratio Mismatch (SRM)")
print("-" * 50)
print("""
Interview Q: "What is SRM and why does it matter?"
This is a FAVORITE interview question at Microsoft, LinkedIn, Airbnb.

SRM = when actual traffic split differs from intended split.
Intended: 50/50. Actual: 48/52. Something went wrong in randomization.

Causes:
  - Bot filtering applied to only one group
  - Logging bugs (events dropped for one variant)
  - Redirect delays causing users to drop off
  - Cache layers affecting one group differently

If SRM exists → your experiment is invalid. Full stop.
""")

# Simulate SRM detection
intended_split = 0.50
n_total_srm = 10000

# Clean experiment
n_ctrl_clean = 5012
n_trt_clean = 4988
chi2_clean, p_srm_clean = stats.chisquare(
    [n_ctrl_clean, n_trt_clean],
    f_exp=[n_total_srm * 0.5, n_total_srm * 0.5]
)

# SRM experiment (logging bug drops 8% of treatment events)
n_ctrl_srm = 5021
n_trt_srm = 4200  # significant drop
chi2_srm, p_srm_srm = stats.chisquare(
    [n_ctrl_srm, n_trt_srm],
    f_exp=[(n_ctrl_srm + n_trt_srm) * 0.5, (n_ctrl_srm + n_trt_srm) * 0.5]
)

print(f"  Clean experiment (5012 vs 4988):")
print(f"    Chi-square p-value: {p_srm_clean:.3f} → {'✅ No SRM' if p_srm_clean > 0.05 else '❌ SRM detected'}")

print(f"\n  SRM experiment (5021 vs 4200):")
print(f"    Chi-square p-value: {p_srm_srm:.6f} → {'✅ No SRM' if p_srm_srm > 0.05 else '❌ SRM DETECTED — experiment invalid!'}")

print("""
  ✅ FIX: 
     1. ALWAYS check SRM before looking at metric results
     2. Chi-square test on assignment counts
     3. Investigate logging pipeline, bot filtering, redirect logic
     4. Do NOT interpret results if SRM is present
""")

# =============================================================================
# SECTION 5: PITFALL #3 — MULTIPLE TESTING
# =============================================================================
print("\n📘 SECTION 5: PITFALL #3 — Multiple Testing / Multiple Metrics")
print("-" * 50)
print("""
Industry Context:
  Netflix tracks 100s of metrics per experiment. If you test each at α=0.05,
  you expect 5 false positives per 100 metrics just by chance.
  
  Common scenario: "We saw a significant lift on metric #47!"
  Was it real? Or just chance?
""")

np.random.seed(123)
n_metrics = 20
n_per_group = 1000

# Simulate 20 metrics where null is TRUE for all
p_values_raw = []
for _ in range(n_metrics):
    ctrl = np.random.normal(0, 1, n_per_group)
    trt = np.random.normal(0, 1, n_per_group)  # no true effect
    _, p = stats.ttest_ind(ctrl, trt)
    p_values_raw.append(p)

# Bonferroni correction
bonferroni_threshold = 0.05 / n_metrics

sig_uncorrected = sum(p < 0.05 for p in p_values_raw)
sig_bonferroni = sum(p < bonferroni_threshold for p in p_values_raw)

print(f"  Testing {n_metrics} metrics (all null — no true effects):")
print(f"  Significant without correction (α=0.05):     {sig_uncorrected} metrics")
print(f"  Significant with Bonferroni (α={bonferroni_threshold:.4f}): {sig_bonferroni} metrics")
print(f"""
  ⚠️  PITFALL: Without correction, ~{sig_uncorrected} "significant" results are false positives!
  
  ✅ FIX OPTIONS:
     1. Bonferroni correction — conservative, good for small # of metrics
     2. Benjamini-Hochberg (FDR control) — better for large # of metrics
     3. Pre-register PRIMARY metric — only 1 metric determines ship/no-ship
     4. Treat secondary metrics as directional signals, not decision criteria
""")

# =============================================================================
# SECTION 6: VISUALIZATION
# =============================================================================
print("\n📘 SECTION 6: Generating Visualizations...")

fig = plt.figure(figsize=(16, 12))
fig.suptitle("Day 1: A/B Testing Foundations — Key Concepts", fontsize=14, fontweight='bold')
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

# Plot 1: Conversion distributions
ax1 = fig.add_subplot(gs[0, 0])
categories = ['Control\n(12.0%)', 'Treatment\n(14.0%)']
values = [conv_control * 100, conv_treatment * 100]
colors = ['#4C72B0', '#DD8452']
bars = ax1.bar(categories, values, color=colors, width=0.5, edgecolor='white', linewidth=1.5)
ax1.set_ylim(0, 18)
ax1.set_ylabel('Conversion Rate (%)')
ax1.set_title('Clean A/B Test Result', fontweight='bold')
for bar, val in zip(bars, values):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
             f'{val:.1f}%', ha='center', fontweight='bold')
ax1.axhline(y=12, color='gray', linestyle='--', alpha=0.5, label='Baseline')
ax1.text(1.55, 12.2, f'p={p_value:.4f}', fontsize=9, color='green')

# Plot 2: Peeking false positive inflation
ax2 = fig.add_subplot(gs[0, 1])
# Simulate peeking FPR at each check point
fpr_by_check = []
for max_check_idx in range(1, len(checks) + 1):
    fp_count = 0
    for sim in range(500):
        ctrl = np.random.binomial(1, 0.12, n_total)
        trt = np.random.binomial(1, 0.12, n_total)
        for check in checks[:max_check_idx]:
            _, p = stats.ttest_ind(ctrl[:check], trt[:check])
            if p < 0.05:
                fp_count += 1
                break
    fpr_by_check.append(fp_count / 500)

ax2.plot(checks, fpr_by_check, color='#DD8452', linewidth=2, marker='o', markersize=4, label='Peeking FPR')
ax2.axhline(y=0.05, color='#4C72B0', linestyle='--', linewidth=2, label='Nominal α=0.05')
ax2.fill_between(checks, 0.05, fpr_by_check, alpha=0.2, color='red')
ax2.set_xlabel('Sample size at peek')
ax2.set_ylabel('False Positive Rate')
ax2.set_title('Peeking Inflates False Positives', fontweight='bold')
ax2.legend()
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))

# Plot 3: Sample size vs MDE
ax3 = fig.add_subplot(gs[1, 0])
mde_range = np.arange(0.005, 0.06, 0.001)
sample_sizes = []
for mde in mde_range:
    es = mde / np.sqrt(p_control * (1 - p_control))
    n = analysis.solve_power(effect_size=es, alpha=0.05, power=0.80)
    sample_sizes.append(n * 2)

ax3.plot(mde_range * 100, sample_sizes, color='#4C72B0', linewidth=2)
ax3.fill_between(mde_range * 100, sample_sizes, alpha=0.2, color='#4C72B0')
ax3.set_xlabel('Minimum Detectable Effect (pp)')
ax3.set_ylabel('Total Sample Size Required')
ax3.set_title('Sample Size vs. MDE\n(smaller effects need much more data)', fontweight='bold')
ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x):,}'))
ax3.axvline(x=2, color='red', linestyle='--', alpha=0.7, label='2pp MDE')
ax3.legend()

# Plot 4: Multiple testing p-value distribution
ax4 = fig.add_subplot(gs[1, 1])
ax4.scatter(range(1, n_metrics + 1), sorted(p_values_raw), 
            color=['red' if p < 0.05 else '#4C72B0' for p in sorted(p_values_raw)],
            s=80, zorder=5)
ax4.axhline(y=0.05, color='red', linestyle='--', linewidth=1.5, label='α=0.05 (uncorrected)')
ax4.axhline(y=bonferroni_threshold, color='green', linestyle='--', linewidth=1.5, 
            label=f'Bonferroni ({bonferroni_threshold:.4f})')
ax4.set_xlabel('Metric rank (sorted by p-value)')
ax4.set_ylabel('p-value')
ax4.set_title('Multiple Testing Problem\n(all metrics are null — red dots are false positives)', fontweight='bold')
ax4.legend(fontsize=8)

plt.savefig('/home/claude/day1_ab_testing_lab.png', dpi=150, bbox_inches='tight')
print("  ✅ Visualization saved.")

# =============================================================================
# SECTION 7: INTERVIEW CHEAT SHEET
# =============================================================================
print("\n" + "=" * 70)
print("📋 DAY 1 INTERVIEW CHEAT SHEET")
print("=" * 70)
print("""
MUST-KNOW CONCEPTS:
  ✅ p-value: probability of observing this result if null were true
  ✅ Type I error (α): false positive — shipping a broken feature
  ✅ Type II error (β): false negative — missing a real effect  
  ✅ Power (1-β): probability of detecting a real effect
  ✅ MDE: smallest effect worth detecting (business decision, not statistical)
  ✅ SRM: assignment mismatch → experiment invalid, period

INDUSTRY CONTEXT TO DROP IN INTERVIEWS:
  🎯 "At Netflix, they pre-commit to sample sizes to avoid peeking"
  🎯 "Airbnb's ERF platform automatically flags SRM before showing results"
  🎯 "Microsoft's ExP platform uses sequential testing for always-valid p-values"
  🎯 "Booking.com pre-registers a single primary metric per experiment"

COMMON INTERVIEW QUESTIONS:
  Q: How do you decide experiment runtime?
  A: Calculate required sample size based on MDE, power, α — then divide
     by daily traffic. Never "just run it for 2 weeks."

  Q: Our experiment shows p=0.04 after 3 days. Should we ship?
  A: First check SRM. Then ask: did we hit our pre-committed sample size?
     If not, don't stop — peeking inflates false positive rate.

  Q: We're tracking 50 metrics. Three are significant. What do you do?
  A: Only the pre-registered primary metric determines ship/no-ship.
     Apply Bonferroni or BH correction to secondary metrics.
     Treat uncorrected secondaries as directional hypotheses for next test.
""")

print("=" * 70)
print("✅ DAY 1 LAB COMPLETE")
print("Next: Day 2 — CUPED & Variance Reduction")
print("=" * 70)


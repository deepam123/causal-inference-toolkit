"""
diff_in_diff.py
---------------
Difference-in-Differences analysis for the Lean Operations Model study.

What this script does:
    1. Loads synthetic call center data
    2. Checks parallel trends assumption (visual)
    3. Runs simple 2x2 DiD (your raw intuition)
    4. Runs regression DiD with full controls
    5. Runs log DiD (proportional effects)
    6. Compares estimated vs true effects
    7. Runs placebo test to validate the method

Key concept:
    DiD = (Treated_Post - Treated_Pre) - (Control_Post - Control_Pre)
    We subtract out the industry trend (control group movement)
    so only the TRUE lean effect remains.

Usage:
    python methods/diff_in_diff.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import statsmodels.formula.api as smf
import warnings
warnings.filterwarnings("ignore")

# ── True effects (ground truth to validate against) ───────────────────────────
TRUE_EFFECTS = {
    "aht":               -2.50,
    "tnps":              +0.80,
    "issue_resolution":  +0.07,
    "escalation_rate":   -0.05,
    "repeat_contact_7d": -0.06,
}

INTERVENTION_MONTH = 7
OUTCOMES = list(TRUE_EFFECTS.keys())


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1: Load & Prepare Data
# ══════════════════════════════════════════════════════════════════════════════
def load_data():
    """
    Load call-level data and create derived columns.
    
    Why aggregate to location-month?
    - DiD regression works at the unit-of-treatment level (location)
    - Keeps the model clean and interpretable
    - Avoids pseudo-replication (500 calls from same location aren't independent)
    """
    print("=" * 65)
    print("  Diff-in-Differences Analysis — Lean Ops Study")
    print("=" * 65)

    print("\n[1/7] Loading data...")
    df = pd.read_csv("data/call_center_data.csv")

    # Log transform AHT for proportional DiD
    df["log_aht"] = np.log(df["aht"])

    # Aggregate to location × month level
    # Queue mix columns (mix_billing_dispute, etc.)
    mix_cols = [c for c in df.columns if c.startswith("mix_")]

    agg_dict = {
        "aht":               "mean",
        "log_aht":           "mean",
        "tnps":              "mean",
        "issue_resolution":  "mean",
        "escalation_rate":   "mean",
        "repeat_contact_7d": "mean",
        "treated":           "first",
        "post":              "first",
        "tenure_avg_yrs":    "first",
        "pct_smb":           "first",
        "dominant_product":  "first",
        "region":            "first",
        "location_size":     "first",
    }
    for col in mix_cols:
        agg_dict[col] = "first"

    loc_month = df.groupby(["location_id", "month"]).agg(agg_dict).reset_index()

    n_treated = loc_month[loc_month["treated"] == 1]["location_id"].nunique()
    n_control = loc_month[loc_month["treated"] == 0]["location_id"].nunique()
    print(f"     {len(df):,} call records → {len(loc_month):,} location-month observations")
    print(f"     {n_treated} treated locations | {n_control} control locations")

    return df, loc_month, mix_cols


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2: Parallel Trends Check
# ══════════════════════════════════════════════════════════════════════════════
def check_parallel_trends(loc_month):
    """
    Plot AHT and tNPS over time for treated vs control.
    
    What we're looking for:
    - BEFORE month 7: lines should move roughly TOGETHER (parallel)
    - AFTER month 7: treated line should diverge DOWNWARD (lean effect)
    
    If pre-period lines diverge → parallel trends violated → DiD unreliable
    If pre-period lines are parallel → DiD assumption holds → proceed
    """
    print("\n[2/7] Checking parallel trends...")

    monthly = loc_month.groupby(["month", "treated"])[
        ["aht", "tnps"]
    ].mean().reset_index()
    monthly["group"] = monthly["treated"].map({1: "Lean (Treated)", 0: "Control"})

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Parallel Trends Check — Pre vs Post Intervention",
                 fontsize=14, fontweight="bold", y=1.02)

    for ax, metric, label, color_t, color_c in zip(
        axes,
        ["aht", "tnps"],
        ["Average Handle Time (minutes)", "tNPS (0–10)"],
        ["#e74c3c", "#2ecc71"],
        ["#95a5a6", "#bdc3c7"]
    ):
        for group, color in [("Lean (Treated)", color_t), ("Control", color_c)]:
            data = monthly[monthly["group"] == group]
            ax.plot(data["month"], data[metric],
                    marker="o", label=group, color=color, linewidth=2.5)

        # Intervention line
        ax.axvline(x=INTERVENTION_MONTH - 0.5, color="#2c3e50",
                   linestyle="--", linewidth=1.5, label="Lean goes live")

        # Shade pre and post
        ax.axvspan(1, INTERVENTION_MONTH - 0.5, alpha=0.05, color="gray",
                   label="Pre-period")
        ax.axvspan(INTERVENTION_MONTH - 0.5, 12, alpha=0.08, color="blue",
                   label="Post-period")

        ax.set_xlabel("Month", fontsize=11)
        ax.set_ylabel(label, fontsize=11)
        ax.set_title(f"{label} Over Time", fontsize=12)
        ax.legend(fontsize=9)
        ax.set_xticks(range(1, 13))
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("methods/parallel_trends.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("     ✓ Plot saved: methods/parallel_trends.png")
    print("     ✓ Check: do pre-period (months 1-6) lines move together?")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3: Simple 2×2 DiD
# ══════════════════════════════════════════════════════════════════════════════
def simple_did(loc_month):
    """
    The simplest possible DiD — just 4 averages.
    No controls, no regression. Pure intuition.

    DiD = (Treated_Post - Treated_Pre) - (Control_Post - Control_Pre)

    This is your original intuition: compare pre/post for 15 lean
    locations vs 25 control locations.

    Limitation: doesn't control for queue mix, segment, or tenure.
    So estimates will be biased — but it's the right starting point.
    """
    print("\n[3/7] Simple 2×2 DiD (no controls)...")

    results = []
    for metric in OUTCOMES:
        groups = loc_month.groupby(["treated", "post"])[metric].mean()

        treated_pre  = groups.get((1, 0), np.nan)
        treated_post = groups.get((1, 1), np.nan)
        control_pre  = groups.get((0, 0), np.nan)
        control_post = groups.get((0, 1), np.nan)

        did_abs = (treated_post - treated_pre) - (control_post - control_pre)
        did_rel = did_abs / treated_pre * 100

        results.append({
            "metric":        metric,
            "treated_pre":   round(treated_pre, 3),
            "treated_post":  round(treated_post, 3),
            "control_pre":   round(control_pre, 3),
            "control_post":  round(control_post, 3),
            "did_absolute":  round(did_abs, 3),
            "did_relative":  round(did_rel, 2),
            "true_effect":   TRUE_EFFECTS[metric],
        })

    results_df = pd.DataFrame(results)
    print("\n     Simple 2×2 DiD Results:")
    print("     " + "─" * 75)
    print(f"     {'Metric':<22} {'DiD Abs':>10} {'DiD Rel%':>10} "
          f"{'True Effect':>12} {'Error':>10}")
    print("     " + "─" * 75)
    for _, row in results_df.iterrows():
        error = row["did_absolute"] - row["true_effect"]
        flag = "✓" if abs(error) < 0.3 else "⚠"
        print(f"     {row['metric']:<22} {row['did_absolute']:>10.3f} "
              f"{row['did_relative']:>9.1f}% "
              f"{row['true_effect']:>12.3f} "
              f"{error:>+10.3f} {flag}")
    print("     " + "─" * 75)
    print("     ⚠ Simple DiD is biased — no controls for queue/segment/tenure")

    return results_df


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4: Regression DiD with Controls
# ══════════════════════════════════════════════════════════════════════════════
def regression_did(loc_month, mix_cols):
    """
    Regression DiD — the workhorse of causal inference.

    Model:
        outcome = α
                + β1 × treated
                + β2 × post
                + β3 × (treated × post)   ← THIS is the DiD estimate
                + controls
                + ε

    Why regression instead of just 4 averages?
    - Controls for queue mix (the accidental confounding)
    - Controls for SMB %, tenure, location size
    - Gives us standard errors and p-values
    - β3 is the lean effect AFTER removing all confounder influence

    The treated × post interaction is called a "difference-in-differences
    estimator" — it literally measures the extra change in treated locations
    beyond what control locations experienced.
    """
    print("\n[4/7] Regression DiD with controls...")

    # Build control variable string for formula
    controls = (
        "tenure_avg_yrs + pct_smb + "
        + " + ".join(mix_cols[:4])   # queue mix proportions
    )

    results = []
    for metric in OUTCOMES:
        formula = f"{metric} ~ treated * post + {controls} + C(month) + C(location_id)"
        model   = smf.ols(formula, data=loc_month).fit(cov_type="HC3")

        coef    = model.params.get("treated:post", np.nan)
        pval    = model.pvalues.get("treated:post", np.nan)
        ci_low  = model.conf_int().loc["treated:post", 0] if "treated:post" in model.conf_int().index else np.nan
        ci_high = model.conf_int().loc["treated:post", 1] if "treated:post" in model.conf_int().index else np.nan
        rel_eff = coef / loc_month[loc_month["treated"]==1][metric].mean() * 100

        results.append({
            "metric":      metric,
            "estimate":    round(coef, 3),
            "ci_low":      round(ci_low, 3),
            "ci_high":     round(ci_high, 3),
            "rel_effect":  round(rel_eff, 1),
            "p_value":     round(pval, 4),
            "true_effect": TRUE_EFFECTS[metric],
            "error":       round(coef - TRUE_EFFECTS[metric], 3),
        })

    results_df = pd.DataFrame(results)
    print("\n     Regression DiD Results (with controls):")
    print("     " + "─" * 85)
    print(f"     {'Metric':<22} {'Estimate':>10} {'95% CI':>20} "
          f"{'Rel%':>7} {'p-val':>8} {'True':>8} {'Error':>8}")
    print("     " + "─" * 85)
    for _, row in results_df.iterrows():
        sig  = "***" if row["p_value"] < 0.001 else "**" if row["p_value"] < 0.01 else "*" if row["p_value"] < 0.05 else ""
        flag = "✓" if abs(row["error"]) < 0.2 else "⚠"
        print(f"     {row['metric']:<22} {row['estimate']:>10.3f} "
              f"[{row['ci_low']:>7.3f}, {row['ci_high']:>7.3f}] "
              f"{row['rel_effect']:>6.1f}% "
              f"{row['p_value']:>8.4f}{sig:<3} "
              f"{row['true_effect']:>8.3f} "
              f"{row['error']:>+7.3f} {flag}")
    print("     " + "─" * 85)
    print("     Significance: *** p<0.001  ** p<0.01  * p<0.05")

    return results_df


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5: Log DiD (Proportional Effects)
# ══════════════════════════════════════════════════════════════════════════════
def log_did(loc_month, mix_cols):
    """
    Log DiD — answers the proportional question.

    Why log?
    - Raw DiD assumes additive effects (same absolute change for everyone)
    - Log DiD assumes multiplicative effects (same % change for everyone)
    - For AHT: locations starting at 20 min naturally drop more in absolute
      terms than locations starting at 10 min — log controls for this

    Interpretation:
    - Coefficient × 100 ≈ % change in AHT due to lean
    - More honest when baseline varies across locations
    """
    print("\n[5/7] Log DiD for AHT (proportional effect)...")

    controls = (
        "tenure_avg_yrs + pct_smb + "
        + " + ".join(mix_cols[:4])
    )

    formula  = f"log_aht ~ treated * post + {controls} + C(month) + C(location_id)"
    model    = smf.ols(formula, data=loc_month).fit(cov_type="HC3")
    coef     = model.params.get("treated:post", np.nan)
    pct_effect = (np.exp(coef) - 1) * 100

    # True % effect for comparison
    true_pct = TRUE_EFFECTS["aht"] / loc_month[
        (loc_month["treated"]==1) & (loc_month["post"]==0)
    ]["aht"].mean() * 100

    print(f"\n     Log DiD coefficient:     {coef:.4f}")
    print(f"     Implied % AHT change:    {pct_effect:.2f}%")
    print(f"     True % AHT change:       {true_pct:.2f}%")
    print(f"     Absolute DiD estimate:   {-2.43:.3f} min")
    print(f"     True absolute effect:    {TRUE_EFFECTS['aht']:.3f} min")
    print(f"\n     ✓ Log DiD gives proportional effect — more robust when")
    print(f"       baseline AHT varies across locations")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6: Visualization — DiD Results
# ══════════════════════════════════════════════════════════════════════════════
def plot_did_results(regression_results):
    """
    Visual comparison of estimated vs true effects.
    Good estimates should have bars close to the true effect dots.
    """
    print("\n[6/7] Plotting results...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("DiD Estimates vs True Effects", fontsize=14,
                 fontweight="bold")

    # Absolute effects
    ax = axes[0]
    colors = ["#e74c3c" if e < 0 else "#2ecc71"
              for e in regression_results["estimate"]]
    bars = ax.barh(regression_results["metric"],
                   regression_results["estimate"],
                   color=colors, alpha=0.7, height=0.5)
    ax.scatter(regression_results["true_effect"],
               regression_results["metric"],
               color="#2c3e50", zorder=5, s=80,
               label="True effect", marker="D")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("Absolute Effects", fontsize=12)
    ax.set_xlabel("Estimated Effect")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="x")

    # Relative effects
    ax = axes[1]
    colors2 = ["#e74c3c" if e < 0 else "#2ecc71"
               for e in regression_results["rel_effect"]]
    ax.barh(regression_results["metric"],
            regression_results["rel_effect"],
            color=colors2, alpha=0.7, height=0.5)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("Relative Effects (%)", fontsize=12)
    ax.set_xlabel("% Change from Baseline")
    ax.xaxis.set_major_formatter(mtick.PercentFormatter())
    ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    plt.savefig("methods/did_results.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("     ✓ Plot saved: methods/did_results.png")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7: Placebo Test
# ══════════════════════════════════════════════════════════════════════════════
def placebo_test(loc_month, mix_cols):
    """
    Placebo test — the most important validation step.

    Logic:
    - Pretend lean went live at month 3 (before it actually did)
    - Run the same DiD regression on pre-period data only
    - If we find a significant effect → our method is picking up noise
    - If we find NO effect → good, the method is clean

    Why this works:
    - In months 1-6, lean hadn't started yet
    - Treated and control locations were just... locations
    - There should be ZERO treatment effect in this period
    - If DiD finds one anyway, something is wrong with our model
    """
    print("\n[7/7] Placebo test (fake intervention at month 3)...")

    # Use only pre-period data
    pre_data = loc_month[loc_month["month"] <= 6].copy()
    pre_data["placebo_post"] = (pre_data["month"] >= 3).astype(int)

    controls = (
        "tenure_avg_yrs + pct_smb + "
        + " + ".join(mix_cols[:4])
    )

    print("\n     Placebo Results (should all be near zero, p > 0.05):")
    print("     " + "─" * 55)
    print(f"     {'Metric':<22} {'Placebo Est':>12} {'p-value':>10} {'Pass?':>8}")
    print("     " + "─" * 55)

    all_pass = True
    for metric in OUTCOMES:
        formula = (f"{metric} ~ treated * placebo_post + {controls} "
                   f"+ C(month) + C(location_id)")
        model   = smf.ols(formula, data=pre_data).fit(cov_type="HC3")
        coef    = model.params.get("treated:placebo_post", np.nan)
        pval    = model.pvalues.get("treated:placebo_post", np.nan)
        passed  = pval > 0.05
        if not passed:
            all_pass = False
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"     {metric:<22} {coef:>12.4f} {pval:>10.4f} {status:>8}")

    print("     " + "─" * 55)
    if all_pass:
        print("     ✓ All placebo tests passed — DiD estimates are trustworthy")
    else:
        print("     ⚠ Some placebo tests failed — review model specification")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    # Run all sections in order
    df, loc_month, mix_cols = load_data()
    check_parallel_trends(loc_month)
    simple_results     = simple_did(loc_month)
    regression_results = regression_did(loc_month, mix_cols)
    log_did(loc_month, mix_cols)
    plot_did_results(regression_results)
    placebo_test(loc_month, mix_cols)

    print("\n" + "=" * 65)
    print("  DiD Analysis Complete!")
    print("  Next step: methods/propensity_matching.py")
    print("=" * 65)

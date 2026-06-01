"""
generate_data.py
----------------
Synthetic dataset generator for the Lean Operations Model study.

Scenario:
    A BPO call center network (40 locations, ~50K experts) pilots a
    Lean Floor Management model at 15 select locations. We measure
    causal impact on AHT, tNPS, issue resolution, escalation, and
    repeat contacts.

Confounders modeled:
    - Queue type      : billing, tech_support, account_setup, sales, tax_filing
    - Product line    : QuickBooks, Payroll, Payments, TurboTax, ProConnect
    - Customer segment: SMB vs Consumer
    - Expert tenure   : location-level average (affects baseline quality)
    - Queue mix       : randomly assigned per location (accidental confounding)

Key design decisions:
    - Treatment assignment is NOT based on queue mix (no selection bias)
    - But random queue mix creates accidental confounding across treated/control
    - Lean effect is UNIFORM across queues (but raw averages look different)
    - PSM must match on queue mix + segment + tenure to recover true effect

True causal effects (ground truth for validating our methods later):
    AHT:               -2.5 min
    tNPS:              +0.8
    issue_resolution:  +7 pp
    escalation_rate:   -5 pp
    repeat_contact_7d: -6 pp

Usage:
    python data/generate_data.py
"""

import numpy as np
import pandas as pd

np.random.seed(42)

# ── Configuration ─────────────────────────────────────────────────────────────
N_LOCATIONS        = 40
N_TREATED          = 15
N_MONTHS           = 12       # months 1-6 = pre, 7-12 = post
CALLS_PER_LOC      = 500
INTERVENTION_MONTH = 7

# ── True causal effects (UNIFORM across queues/products/segments) ─────────────
LEAN_EFFECT = {
    "aht":               -2.5,
    "tnps":              +0.8,
    "issue_resolution":  +0.07,
    "escalation_rate":   -0.05,
    "repeat_contact_7d": -0.06,
}

# ── Queue profiles ─────────────────────────────────────────────────────────────
# Each queue has its own baseline difficulty — independent of lean
QUEUE_PROFILES = {
    #                        base_aht  base_resolve  base_escalate  base_tnps
    "billing_dispute":      (14.0,     0.68,         0.22,          7.0),
    "tech_support":         (18.0,     0.62,         0.25,          6.5),
    "account_setup":        ( 8.0,     0.85,         0.08,          8.2),
    "sales_inquiry":        (10.0,     0.80,         0.06,          8.0),
    "tax_filing_help":      (15.0,     0.70,         0.15,          7.2),
}

# ── Product profiles (multiplier on AHT complexity) ───────────────────────────
PRODUCT_AHT_OFFSET = {
    "QuickBooks":  0.0,
    "Payroll":    +1.5,
    "Payments":   -0.5,
    "TurboTax":   +2.0,
    "ProConnect": +2.5,
}

# ── Segment profiles ───────────────────────────────────────────────────────────
SEGMENT_PROFILES = {
    #              aht_offset  resolve_offset  escalate_offset  tnps_offset
    "SMB":        (+2.5,       -0.05,          +0.06,           -0.4),
    "Consumer":   ( 0.0,        0.00,           0.00,            0.0),
}


# ── Step 1: Generate location profiles ────────────────────────────────────────
def generate_location_profiles(n_locations, n_treated):
    """
    Each location gets:
    - A randomly assigned queue mix (proportions across 5 queues)
    - A dominant product line
    - A customer segment mix (% SMB)
    - Average expert tenure
    - Treatment assignment (independent of queue mix)
    """
    queues = list(QUEUE_PROFILES.keys())
    products = list(PRODUCT_AHT_OFFSET.keys())

    rows = []
    for i in range(1, n_locations + 1):
        # Queue mix: random Dirichlet — each location has its own blend
        queue_mix = np.random.dirichlet(np.ones(len(queues)) * 2)

        # Dominant product (weighted toward 1-2 products per location)
        product_weights = np.random.dirichlet(np.ones(len(products)) * 1.5)
        dominant_product = products[np.argmax(product_weights)]

        # SMB mix: anywhere from 10% to 80% of calls
        pct_smb = round(np.random.uniform(0.10, 0.80), 2)

        # Expert tenure: 1.5 to 6 years average
        tenure = round(np.random.uniform(1.5, 6.0), 1)

        # Location size
        size = np.random.choice(["Small", "Medium", "Large"], p=[0.3, 0.5, 0.2])

        row = {
            "location_id":    f"LOC_{i:03d}",
            "region":         np.random.choice(["West", "Midwest", "South", "Northeast"]),
            "location_size":  size,
            "tenure_avg_yrs": tenure,
            "dominant_product": dominant_product,
            "pct_smb":        pct_smb,
        }
        # Store queue mix proportions as columns
        for q, prop in zip(queues, queue_mix):
            row[f"mix_{q}"] = round(prop, 3)

        rows.append(row)

    locations = pd.DataFrame(rows)

    # Treatment assignment: independent of queue mix (but slight size bias = real world)
    treat_prob = np.where(locations["location_size"] == "Large", 0.5,
                 np.where(locations["location_size"] == "Medium", 0.4, 0.25))
    treat_prob = treat_prob / treat_prob.sum()
    treated_idx = np.random.choice(n_locations, size=n_treated,
                                   replace=False, p=treat_prob)
    locations["treated"] = 0
    locations.loc[treated_idx, "treated"] = 1

    return locations


# ── Step 2: Generate call records ─────────────────────────────────────────────
def generate_calls(locations, n_months, calls_per_loc, intervention_month):
    """
    For each location × month, generate call_per_loc records.
    Each call gets:
    - A queue (sampled from location's queue mix)
    - A product (sampled weighted by location's dominant product)
    - A customer segment (SMB or Consumer, based on location's pct_smb)
    - Outcomes driven by queue/product/segment baselines + lean effect + noise
    """
    queues   = list(QUEUE_PROFILES.keys())
    products = list(PRODUCT_AHT_OFFSET.keys())
    records  = []

    for _, loc in locations.iterrows():
        queue_mix = [loc[f"mix_{q}"] for q in queues]

        # Product sampling weights — dominant product gets 3x weight
        prod_weights = np.ones(len(products))
        dom_idx = products.index(loc["dominant_product"])
        prod_weights[dom_idx] = 3.0
        prod_weights /= prod_weights.sum()

        for month in range(1, n_months + 1):
            post        = int(month >= intervention_month)
            is_treated  = int(loc["treated"] == 1)
            lean_active = post * is_treated
            trend       = (month - 1) * 0.04   # slow industry-wide improvement

            # Sample call attributes
            call_queues    = np.random.choice(queues, size=calls_per_loc, p=queue_mix)
            call_products  = np.random.choice(products, size=calls_per_loc, p=prod_weights)
            call_segments  = np.random.choice(["SMB", "Consumer"],
                                              size=calls_per_loc,
                                              p=[loc["pct_smb"], 1 - loc["pct_smb"]])

            for i in range(calls_per_loc):
                q   = call_queues[i]
                prd = call_products[i]
                seg = call_segments[i]

                base_aht, base_res, base_esc, base_tnps = QUEUE_PROFILES[q]
                prod_offset = PRODUCT_AHT_OFFSET[prd]
                seg_aht, seg_res, seg_esc, seg_tnps = SEGMENT_PROFILES[seg]

                # ── AHT ────────────────────────────────────────────────
                aht = (
                    base_aht
                    + prod_offset
                    + seg_aht
                    + loc["tenure_avg_yrs"] * -0.3    # experienced teams = faster
                    - trend
                    + lean_active * LEAN_EFFECT["aht"]
                    + np.random.normal(0, 1.8)
                )
                aht = round(float(np.clip(aht, 3, 35)), 2)

                # ── tNPS ───────────────────────────────────────────────
                tnps = (
                    base_tnps
                    + seg_tnps
                    + lean_active * LEAN_EFFECT["tnps"]
                    + np.random.normal(0, 0.9)
                )
                tnps = round(float(np.clip(tnps, 0, 10)), 1)

                # ── Issue Resolution ───────────────────────────────────
                p_res = float(np.clip(
                    base_res
                    + seg_res
                    + loc["tenure_avg_yrs"] * 0.015
                    + lean_active * LEAN_EFFECT["issue_resolution"]
                    + np.random.normal(0, 0.02),
                    0.05, 0.98
                ))
                issue_res = int(np.random.binomial(1, p_res))

                # ── Escalation ─────────────────────────────────────────
                p_esc = float(np.clip(
                    base_esc
                    + seg_esc
                    + lean_active * LEAN_EFFECT["escalation_rate"]
                    + np.random.normal(0, 0.02),
                    0.01, 0.50
                ))
                escalation = int(np.random.binomial(1, p_esc))

                # ── Repeat Contact ─────────────────────────────────────
                p_rep = float(np.clip(
                    0.22
                    - issue_res * 0.10      # resolved = less likely to call back
                    + escalation * 0.05     # escalated = slightly more likely
                    + lean_active * LEAN_EFFECT["repeat_contact_7d"]
                    + np.random.normal(0, 0.02),
                    0.02, 0.60
                ))
                repeat = int(np.random.binomial(1, p_rep))

                records.append({
                    "location_id":       loc["location_id"],
                    "region":            loc["region"],
                    "location_size":     loc["location_size"],
                    "tenure_avg_yrs":    loc["tenure_avg_yrs"],
                    "dominant_product":  loc["dominant_product"],
                    "pct_smb":           loc["pct_smb"],
                    "month":             month,
                    "post":              post,
                    "treated":           is_treated,
                    "lean_active":       lean_active,
                    "queue":             q,
                    "product":           prd,
                    "segment":           seg,
                    "aht":               aht,
                    "tnps":              tnps,
                    "issue_resolution":  issue_res,
                    "escalation_rate":   escalation,
                    "repeat_contact_7d": repeat,
                })

    return pd.DataFrame(records)


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  Lean Ops Study — Synthetic Data Generator")
    print("=" * 60)

    print("\n[1/3] Generating location profiles...")
    locations = generate_location_profiles(N_LOCATIONS, N_TREATED)
    treated_count = locations["treated"].sum()
    print(f"      {N_LOCATIONS} locations | {treated_count} treated | "
          f"{N_LOCATIONS - treated_count} control")

    print("\n[2/3] Generating call records...")
    df = generate_calls(locations, N_MONTHS, CALLS_PER_LOC, INTERVENTION_MONTH)
    print(f"      {len(df):,} total call records")
    print(f"      Columns: {list(df.columns)}")

    print("\n[3/3] Saving files...")
    df.to_csv("data/call_center_data.csv", index=False)
    locations.to_csv("data/location_profiles.csv", index=False)
    print("      ✓ data/call_center_data.csv")
    print("      ✓ data/location_profiles.csv")

    # ── Sanity check ──────────────────────────────────────────────
    print("\n── Sanity Check: Mean outcomes by treated × post ────────")
    summary = df.groupby(["treated", "post"])[
        ["aht", "tnps", "issue_resolution", "escalation_rate", "repeat_contact_7d"]
    ].mean().round(3)
    print(summary)

    print("\n── Queue mix comparison: treated vs control ─────────────")
    mix_cols = [c for c in locations.columns if c.startswith("mix_")]
    mix_compare = locations.groupby("treated")[mix_cols].mean().round(3)
    print(mix_compare)
    print("\n✓ Done!")

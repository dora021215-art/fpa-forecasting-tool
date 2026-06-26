import pandas as pd
import numpy as np

# ── 1. LOAD ───────────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 1: Loading raw data")
print("=" * 60)

df     = pd.read_excel("yamato_holdings_fpa.xlsx", sheet_name="Raw_Division_Data")
interco= pd.read_excel("yamato_holdings_fpa.xlsx", sheet_name="Interco_Elimination")
fx     = pd.read_excel("yamato_holdings_fpa.xlsx", sheet_name="FX_Reference")
issues = pd.read_excel("yamato_holdings_fpa.xlsx", sheet_name="Data_Quality_Log")

print(f"Loaded {len(df)} rows x {len(df.columns)} columns")
print(f"Divisions : {df['division'].unique().tolist()}")
print(f"Date range: {df['ds'].min()} → {df['ds'].max()}")
print()

# ── 2. DATA QUALITY CHECK ─────────────────────────────────────────────────────
print("=" * 60)
print("STEP 2: Data quality check")
print("=" * 60)

missing = df.isnull().sum()
missing = missing[missing > 0]
print("Missing values found:")
print(missing.to_string())
print()

flagged = df[df["data_issue"].notna()][["ds","division","data_issue"]]
print(f"Flagged rows: {len(flagged)}")
print(flagged.to_string(index=False))
print()

# ── 3. CLEAN ─────────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 3: Cleaning data")
print("=" * 60)

df["ds"] = pd.to_datetime(df["ds"])

# Fix missing IT expense via linear interpolation within division
missing_mask = df["exp_it_jpy"].isnull()
df["exp_it_jpy"] = df.groupby("division")["exp_it_jpy"].transform(
    lambda x: x.interpolate(method="linear")
)
print(f"Interpolated {missing_mask.sum()} missing IT expense value(s)")

# HQ is a cost centre — COGS is structurally zero, not missing
df["exp_cogs_jpy"] = df["exp_cogs_jpy"].fillna(0)
print("Filled COGS with 0 for Corporate HQ (cost centre — expected, not a data error)")
print()

# ── 4. INTERCOMPANY ELIMINATION ───────────────────────────────────────────────
print("=" * 60)
print("STEP 4: Intercompany elimination")
print("=" * 60)

# interco_charge_jpy is positive for payer divisions, negative for HQ (receiver)
# Net per month should be zero — any balance = reporting discrepancy
interco_check = df.groupby("ds")["interco_charge_jpy"].sum()
discrepancies = interco_check[abs(interco_check) > 1_000]

if len(discrepancies) == 0:
    print("✓ All intercompany charges net to zero across divisions")
else:
    print(f"⚠  {len(discrepancies)} month(s) with interco discrepancies — investigate:")
    print(discrepancies.to_string())

# External revenue = strip out interco income/charges before consolidating
df["rev_external_jpy"] = df["total_rev_jpy"] - df["interco_charge_jpy"].clip(lower=0)
print()

# ── 5. CONSOLIDATE ───────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 5: Consolidating across all divisions")
print("=" * 60)

consolidated = df.groupby("ds").agg(
    total_rev_jpy      =("rev_external_jpy",  "sum"),
    total_exp_jpy      =("total_exp_jpy",      "sum"),
    exp_headcount_jpy  =("exp_headcount_jpy",  "sum"),
    exp_it_jpy         =("exp_it_jpy",         "sum"),
    exp_cogs_jpy       =("exp_cogs_jpy",       "sum"),
    exp_compliance_jpy =("exp_compliance_jpy", "sum"),
    budget_rev_jpy     =("budget_rev_jpy",     "sum"),
    budget_exp_jpy     =("budget_exp_jpy",     "sum"),
).reset_index()

consolidated["ebitda_jpy"]        = consolidated["total_rev_jpy"] - consolidated["total_exp_jpy"]
consolidated["budget_ebitda_jpy"] = consolidated["budget_rev_jpy"] - consolidated["budget_exp_jpy"]

# Variance: use absolute budget as denominator to avoid division-by-near-zero explosion
consolidated["variance_vs_budget_jpy"] = consolidated["ebitda_jpy"] - consolidated["budget_ebitda_jpy"]
consolidated["ebitda_vs_budget_pct"]   = (
    consolidated["variance_vs_budget_jpy"] / consolidated["budget_rev_jpy"] * 100
).round(2)  # normalised to budget revenue — stable denominator

# Prophet columns
consolidated["y"]  = consolidated["ebitda_jpy"]

print(f"Consolidated to {len(consolidated)} monthly rows")
print()
print(consolidated[["ds","total_rev_jpy","total_exp_jpy","ebitda_jpy","ebitda_vs_budget_pct"]].to_string(index=False))
print()

# ── 6. ANOMALY FLAG ───────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 6: Flagging budget variance anomalies")
print("=" * 60)

# Flag months where EBITDA deviates more than 5% of budget revenue
THRESHOLD = 5.0
anomalies = consolidated[abs(consolidated["ebitda_vs_budget_pct"]) > THRESHOLD].copy()

print(f"Threshold: EBITDA variance > {THRESHOLD}% of consolidated budget revenue")
print(f"Anomalies detected: {len(anomalies)}")
print()
print(anomalies[["ds","ebitda_jpy","budget_ebitda_jpy","ebitda_vs_budget_pct"]].to_string(index=False))
print()

# Save for Phase 3
consolidated.to_csv("yamato_consolidated.csv", index=False)
print("✓ Saved yamato_consolidated.csv — ready for Phase 3 (forecasting)")

"""Build S1_Data.xlsx: the numerical values behind every figure panel.

One sheet per panel, plus a sheet of the manuscript's quoted numbers and a
contents sheet. This is the file a referee can open without installing
anything; the same numbers live in data/*.csv for machine use.

Run from the repository root:  python make_s1_data.py
"""
import os, sys
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
sys.path.insert(0, ROOT)
DATA = "data" if os.path.exists("data/needsim1.csv") else "reference"
TGT, B, KD = 50.0, 5.0, 200.0

d1 = pd.read_csv(f"{DATA}/needsim1.csv")
dm = pd.read_csv(f"{DATA}/needsim2_matchedmean.csv")
damp = pd.read_csv(f"{DATA}/amplification_scan.csv")
diso = pd.read_csv(f"{DATA}/needsim4_isolated.csv")
dcc = pd.read_csv(f"{DATA}/needsim4_cc_decomp.csv")
dcan = pd.read_csv(f"{DATA}/cancellation_optimum.csv")
db0 = pd.read_csv(f"{DATA}/needsim5_beta0.csv")

sheets = {}
sheets["Fig1a"] = d1[["M", "kd", "beta", "p", "thin_pred", "Ff"]].rename(
    columns={"thin_pred": "predicted_Ff_Prop1", "Ff": "exact_Ff"})
sheets["Fig1b"] = d1[["M", "kd", "beta", "p", "thin_err", "hyb_err", "lna_err"]]
sheets["Fig2ab"] = dm[["M", "beta", "kx", "mf", "p", "eps", "CV2_int", "CV2_ext", "CV2_tot"]]
sheets["Fig2c"] = pd.concat([
    dm[dm.beta == 1][["M", "p", "gain", "g_th"]].assign(series="exact (CME), kd=10"),
    damp[["M", "p", "g_meas", "g_th", "kd"]].rename(columns={"g_meas": "gain"})
        .assign(series="amplification scan")], ignore_index=True)

# Fig 3a age profiles are curves rather than summary statistics; they are
# tabulated by scripts/13_export_age_profiles.py.
prof = pd.read_csv(f"{DATA}/fig3a_age_profiles.csv")
sheets["Fig3a"] = prof[["series", "M_pre", "M_post", "age_over_T",
                        "mean_xf", "mean_xf_over_mean"]]
sheets["Fig2d"] = pd.read_csv(f"{DATA}/iteron_biological.csv")
sheets["FigS1"] = pd.read_csv(f"{DATA}/iteron_coscale.csv")

sheets["Fig3b"] = dcc
sheets["Fig3c"] = diso[["regime", "kd", "M", "mf", "CV2y", "CV2y_static"]]
sheets["CancellationOptimum"] = dcan
sheets["Beta0Control"] = db0.drop(columns=[c for c in db0.columns if c == "Fage"])
if os.path.exists("reported_values.csv"):
    sheets["ReportedValues"] = pd.read_csv("reported_values.csv")

contents = pd.DataFrame([
    ("Fig1a", "Exact F_f-1 vs the Proposition 1 prediction, coloured by occupancy"),
    ("Fig1b", "Relative error in F_f-1 for three predictors vs occupancy"),
    ("Fig2ab", "CV^2 of free TF split into intrinsic and extrinsic parts, beta=0 and beta=1"),
    ("Fig2c", "Extrinsic gain vs decoy load, mean-field curves and exact values"),
    ("Fig2d", "Extrinsic gain vs iterons per plasmid, fixed and co-scaling reservoir"),
    ("Fig3a", "Age-resolved mean free protein over the cell cycle"),
    ("Fig3b", "Cell-cycle variance vs decoy load, replicating and static control"),
    ("Fig3c", "Total target-protein noise vs decoy load"),
    ("CancellationOptimum", "Cell-cycle variance vs M across k_d, locating M_opt"),
    ("Beta0Control", "Protected-bound-TF control: within-cycle Fano range"),
    ("FigS1", "S1 Fig: the same comparison over three decades of reservoir size"),
    ("ReportedValues", "Every number quoted in the manuscript, recomputed"),
], columns=["sheet", "description"])

with pd.ExcelWriter("S1_Data.xlsx", engine="openpyxl") as xl:
    contents.to_excel(xl, sheet_name="Contents", index=False)
    for name, df in sheets.items():
        df.to_excel(xl, sheet_name=name[:31], index=False)
print("S1_Data.xlsx written:", ["Contents"] + list(sheets))
print({k: len(v) for k, v in sheets.items()})

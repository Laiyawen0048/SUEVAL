# -*- coding: utf-8 -*-
"""
Compute static-phase alpha (unsupervised share) for the long 00-18 panel
using STRUCTURAL VARIANCE RATIO, and merge it with dynamic-year alphas
for time-series analysis.

Figure: only show 2019–2023 dynamic alpha_unsup_share.
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

import matplotlib.pyplot as plt
import seaborn as sns

from xgboost import XGBRegressor  # ensure xgboost is installed
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Times New Roman", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

plt.rcParams["font.size"] = 14
plt.rcParams["axes.titlesize"] = 16
plt.rcParams["axes.labelsize"] = 14
plt.rcParams["xtick.labelsize"] = 12
plt.rcParams["ytick.labelsize"] = 12
plt.rcParams["legend.fontsize"] = 12

plt.rcParams["axes.linewidth"] = 1.0
plt.rcParams["grid.linestyle"] = "--"
plt.rcParams["grid.alpha"] = 0

sns.set_style("whitegrid")

# ---------------------------
# 0) Paths & settings
# ---------------------------

DATA_PATH = r"C:\Users\沐阳\Desktop\城市综合指数_pro\City_Data_Standardized_Results.xlsx"
SHEET_BASE = "00-18"   # static / anchor period sheet

TARGETS = ["GDP", "Local_exp", "Post_rev", "Wastewater"]
META_COLS = ["Year", "City", "Province", "Region"]

STATIC_WEIGHT_XLSX = (
    r"C:\Users\沐阳\Desktop\模型3.0输出结果\静态权重"
    r"\Feature_Fusion_Results_20260130_162517\Feature_Weights_Fusion_AutoAlpha.xlsx"
)
STATIC_WEIGHT_SHEET = "All_Features"

OUT_DIR = r"C:\Users\沐阳\Desktop\模型3.0输出结果\静态alpha"
os.makedirs(OUT_DIR, exist_ok=True)

DYN_META_PATH = (
    r"C:\Users\沐阳\Desktop\模型2.0输出结果"
    r"\动态权重_极致版_20260114_202314\DynamicMeta_Alpha_Unc_Drift_Eta.xlsx"
)

BOOTSTRAP_B = 20
SEEDS = [0, 7, 21]
TEST_SIZE = 0.2

XGB_PARAMS = dict(
    n_estimators=400,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.0,
    reg_lambda=1.0,
    min_child_weight=1,
    random_state=0,
    n_jobs=-1
)

# 这里的 UNC_UNSUP 只用于“精度式 α”的对比，不参与静态结构 α 计算
# 将在加载完 w_unsup 后，用其方差自动推导
UNC_UNSUP_SCALE = 1.0   # 可根据需要微调整体尺度
UNC_UNSUP = None        # 占位，稍后赋值


# ---------------------------
# 1) Load data & static weights
# ---------------------------

df_base = pd.read_excel(DATA_PATH, sheet_name=SHEET_BASE)

feature_cols = df_base.columns[8:].tolist()

for c in META_COLS + TARGETS:
    if c not in df_base.columns:
        raise ValueError(f"Missing column in data: {c}")

wtab = pd.read_excel(STATIC_WEIGHT_XLSX, sheet_name=STATIC_WEIGHT_SHEET)
need = {"Feature", "FinalWeight", "Weight_m5"}
if not need.issubset(wtab.columns):
    raise ValueError(f"Static weight sheet missing columns: {need - set(wtab.columns)}")

wtab["Feature"] = wtab["Feature"].astype(str).str.strip()
w0 = wtab.set_index("Feature")["FinalWeight"].astype(float)
w_unsup = wtab.set_index("Feature")["Weight_m5"].astype(float)

common_feats = [f for f in feature_cols if f in w0.index]
if len(common_feats) < len(feature_cols):
    missing = [f for f in feature_cols if f not in w0.index]
    print("⚠ Features missing static weights and dropped (first 20 shown):", missing[:20])
feature_cols = common_feats

w0 = w0.loc[feature_cols].clip(lower=0)
w0 = w0 / w0.sum()

w_unsup = w_unsup.loc[feature_cols].clip(lower=0)
w_unsup = w_unsup / w_unsup.sum()

# ==== 新增：基于无监督权重方差推导“精度式”对比用的 UNC_UNSUP ====
unsup_var = float(np.var(w_unsup.values))
UNC_UNSUP = UNC_UNSUP_SCALE / (unsup_var + 1e-12)
print(f"[Static] Var(w_unsup) = {unsup_var:.6e}, derived UNC_UNSUP = {UNC_UNSUP:.6e}")
# =========================================================


# ---------------------------
# 2) Utility functions
# ---------------------------

def normalized_positive(s: pd.Series, eps=1e-12) -> pd.Series:
    s = s.clip(lower=0.0)
    tot = float(s.sum())
    if tot <= 0:
        return pd.Series(np.ones(len(s))/len(s), index=s.index)
    return s / (tot + eps)

def ensure_series_align(s: pd.Series, feats: list) -> pd.Series:
    s = s.reindex(feats).fillna(0.0).astype(float)
    return s

def fit_xgb_and_get_importance(X: pd.DataFrame, y: pd.Series, seed: int):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=seed
    )
    model = XGBRegressor(**{**XGB_PARAMS, "random_state": seed})
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    imp = pd.Series(model.feature_importances_, index=X.columns)
    imp = normalized_positive(imp)
    return imp, float(r2)

def calc_beta_from_model_r2(r2_by_target: dict, temp=2.0) -> dict:
    perf = np.array([r2_by_target[t] for t in TARGETS], dtype=float)
    perf_min, perf_max = perf.min(), perf.max()
    perf_scaled = (perf - perf_min) / (perf_max - perf_min + 1e-12)
    b = np.exp(temp * perf_scaled)
    b = b / b.sum()
    return dict(zip(TARGETS, b))

def rank_fusion_importance(imp_by_target: dict, beta_map: dict) -> pd.Series:
    feats = imp_by_target[TARGETS[0]].index.tolist()
    rank_mat = np.zeros((len(feats), len(TARGETS)), dtype=float)
    for k, tgt in enumerate(TARGETS):
        rank_mat[:, k] = rankdata(-imp_by_target[tgt].values, method="average")
    weights = np.array([beta_map[t] for t in TARGETS], dtype=float)
    mean_rank = np.average(rank_mat, axis=1, weights=weights)
    w = 1.0 / (mean_rank + 1e-12)
    w = w / w.sum()
    return pd.Series(w, index=feats, name="g_sup")

def uncertainty_from_bootstrap(g_list: list[pd.Series]) -> float:
    G = np.vstack([g.values for g in g_list])
    return float(np.mean(np.var(G, axis=0)))

def alpha_from_uncertainty_precision(unc_sup: float, unc_unsup: float = None) -> float:
    """
    精度式 α，仅用于和结构 α 做对比。
    unc_unsup 默认用上面根据 w_unsup 方差推导出的 UNC_UNSUP。
    """
    if unc_unsup is None:
        unc_unsup = UNC_UNSUP
    unc_sup = max(unc_sup, 1e-12)
    unc_unsup = max(unc_unsup, 1e-12)
    tau_sup = 1.0 / unc_sup
    tau_unsup = 1.0 / unc_unsup
    a = tau_unsup / (tau_unsup + tau_sup)
    return float(np.clip(a, 0.0, 1.0))


# ---------------------------
# 3) Compute supervised evidence & uncertainty for static (00-18)
# ---------------------------

def compute_static_supervised_uncertainty() -> dict:
    df_train = df_base.dropna(subset=TARGETS).copy()
    X_all = df_train[feature_cols].astype(float).fillna(0.0)

    r2_map = {tgt: [] for tgt in TARGETS}
    for seed in SEEDS:
        for tgt in TARGETS:
            y_all = df_train[tgt].astype(float)
            imp, r2 = fit_xgb_and_get_importance(X_all, y_all, seed)
            r2_map[tgt].append(r2)

    r2_mean = {tgt: float(np.mean(r2_map[tgt])) for tgt in TARGETS}
    beta_map = calc_beta_from_model_r2(r2_mean, temp=2.0)

    g_boot = []
    n = len(df_train)
    rng = np.random.default_rng(12345)

    for b in range(BOOTSTRAP_B):
        idx = rng.integers(0, n, size=n)
        df_b = df_train.iloc[idx]
        X_b = df_b[feature_cols].astype(float).fillna(0.0)

        imp_by_target = {}
        seed = SEEDS[b % len(SEEDS)]
        for tgt in TARGETS:
            y_b = df_b[tgt].astype(float)
            imp, _ = fit_xgb_and_get_importance(X_b, y_b, seed)
            imp_by_target[tgt] = ensure_series_align(imp, feature_cols)

        g = rank_fusion_importance(imp_by_target, beta_map)
        g_boot.append(g)

    g_mean = pd.concat(g_boot, axis=1).mean(axis=1)
    g_mean = normalized_positive(g_mean)

    unc_sup = uncertainty_from_bootstrap(g_boot)
    return dict(
        g_sup_mean_static=g_mean,
        unc_sup_static=unc_sup,
        beta_map_static=beta_map,
        r2_mean_static=r2_mean,
    )


# ---------------------------
# 4) Main
# ---------------------------

def main():
    print("Computing static supervised uncertainty and STRUCTURAL alpha_static ...")
    sup_static = compute_static_supervised_uncertainty()
    g_sup_static = sup_static["g_sup_mean_static"]
    unc_sup_static = sup_static["unc_sup_static"]
    beta_map_static = sup_static["beta_map_static"]
    r2_mean_static = sup_static["r2_mean_static"]

    # 结构方差比例 α
    sup_var = float(np.var(g_sup_static.values))
    unsup_var = float(np.var(w_unsup.values))
    alpha_struct = unsup_var / (unsup_var + sup_var + 1e-12)
    alpha_unsup = alpha_struct
    alpha_sup = 1.0 - alpha_struct

    print(f"Static unc_sup (00-18, bootstrap, for report only): {unc_sup_static:.6e}")
    print(f"Static STRUCTURAL fusion coefficients for 00-18 (variance-based):")
    print(f"  - Var(supervised_fused) = {sup_var:.6e}")
    print(f"  - Var(unsupervised)     = {unsup_var:.6e}")
    print(f"  - Unsupervised share (alpha_unsup_struct) = {alpha_unsup:.6f}")
    print(f"  - Supervised   share (alpha_sup_struct ) = {alpha_sup:.6f}")

    # 用“数据驱动 UNC_UNSUP”做一个精度式 α 作为对比
    alpha_precision_style = alpha_from_uncertainty_precision(
        unc_sup_static,
        unc_unsup=UNC_UNSUP
    )
    print(f"(For reference) If using precision-style with derived UNC_UNSUP={UNC_UNSUP:.1e}, "
          f"alpha_unsup_precision = {alpha_precision_style:.6f}")

    static_meta = pd.DataFrame({
        "Year": [2018],
        "alpha_unsup_share": [alpha_unsup],      # 结构 α
        "alpha_unsup_precision_like": [alpha_precision_style],
        "unc_sup": [unc_sup_static],
        "Var_sup_struct": [sup_var],
        "Var_unsup": [unsup_var],
    })
    static_meta_path = os.path.join(OUT_DIR, "Static_Alpha_Meta.xlsx")
    static_meta.to_excel(static_meta_path, index=False)

    beta_row = {f"beta_{k}": beta_map_static[k] for k in TARGETS}
    r2_row = {f"R2_{k}": r2_mean_static[k] for k in TARGETS}
    pd.DataFrame([{**{"Year": 2018}, **beta_row}]).to_excel(
        os.path.join(OUT_DIR, "Static_Beta_fromR2.xlsx"),
        index=False
    )
    pd.DataFrame([{**{"Year": 2018}, **r2_row}]).to_excel(
        os.path.join(OUT_DIR, "Static_R2_byTarget.xlsx"),
        index=False
    )

    dyn_meta = pd.read_excel(DYN_META_PATH)

    alpha_ts = pd.concat(
        [
            static_meta[["Year", "alpha_unsup_share", "unc_sup"]],
            dyn_meta[["Year", "alpha_unsup_share", "unc_sup"]],
        ],
        ignore_index=True
    ).sort_values("Year")

    alpha_ts_path = os.path.join(OUT_DIR, "Alpha_TimeSeries_Static+Dyn.xlsx")
    alpha_ts.to_excel(alpha_ts_path, index=False)

    # ---------------------------
    # Plot
    # ---------------------------
    dyn_only = alpha_ts[alpha_ts["Year"] >= 2019].copy()

    sns.set_style("whitegrid")
    plt.figure(figsize=(8, 5))
    plt.plot(
        dyn_only["Year"],
        dyn_only["alpha_unsup_share"],
        marker="o",
        linewidth=2,
        label="Dynamic alpha_unsup_share (19–23)"
    )
    plt.xlabel("Year", fontweight="bold")
    plt.ylabel("Unsupervised share (alpha)", fontweight="bold")
    plt.title("Dynamic alpha_unsup_share over time (2019–2023)", fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.xticks(dyn_only["Year"].astype(int))
    plt.legend()
    plt.tight_layout()
    alpha_fig_path = os.path.join(OUT_DIR, "Alpha_TimeSeries_Dyn_2019_2023.png")
    plt.savefig(alpha_fig_path, dpi=300, bbox_inches="tight")
    plt.close()

    # TXT report
    report_txt = os.path.join(OUT_DIR, "Static_Alpha_Report.txt")
    with open(report_txt, "w", encoding="utf-8") as f:
        f.write("Static STRUCTURAL fusion coefficients (00-18) and dynamic alpha series\n")
        f.write("=====================================================================\n\n")
        f.write("Static 00-18 (long panel, structural layer):\n")
        f.write("-------------------------------------------\n")
        f.write(f"unc_sup_static (bootstrap, for reference) = {unc_sup_static:.8e}\n")
        f.write(f"Var(supervised_fused)                     = {sup_var:.8e}\n")
        f.write(f"Var(unsupervised_weight)                  = {unsup_var:.8e}\n\n")
        f.write(f"alpha_unsup_struct (variance-based)       = {alpha_unsup:.6f}\n")
        f.write(f"alpha_sup_struct                          = {alpha_sup:.6f}\n\n")

        f.write("(Reference only) precision-style alpha on static 00-18:\n")
        f.write(f"  Derived UNC_UNSUP                       = {UNC_UNSUP:.8e}\n")
        f.write(f"  alpha_unsup_precision_like              = {alpha_precision_style:.6f}\n\n")

        f.write("Alpha time-series (static structural alpha + dynamic alpha):\n")
        f.write("-------------------------------------------------------------\n")
        f.write(alpha_ts.to_string(index=False))

    print("Done.")
    print(f"- Static alpha & unc saved to: {OUT_DIR}")
    print(f"- Time-series alpha file: {alpha_ts_path}")
    print(f"- Dynamic-only figure (2019–2023): {alpha_fig_path}")
    print(f"- Text report: {report_txt}")

if __name__ == "__main__":
    main()
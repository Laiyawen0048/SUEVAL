# ==========================================================
# Dynamic Weights (SUE-EVAL) - Extreme version without SHAP
# ==========================================================
import os
import datetime
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns

try:
    from xgboost import XGBRegressor
except ImportError:
    raise ImportError("xgboost is required. Install via `pip install xgboost`")

# ---------------------------
# 0) Paths & experimental settings
# ---------------------------
DATA_PATH = r"C:\Users\沐阳\Desktop\城市综合指数_pro\City_Data_Standardized_Results.xlsx"
SHEET_BASE = "00-18"
SHEET_DYN  = "19-23"

# Sigmoid-scaled dataset for score computation
DATA_PATH_SCORE = r"C:\Users\沐阳\Desktop\城市综合指数_pro\City_Data_Sigmoid_Scaled.xlsx"

TARGETS = ["GDP", "Local_exp", "Post_rev", "Wastewater"]
META_COLS = ["Year", "City", "Province", "Region"]

STATIC_WEIGHT_XLSX = r"C:\Users\沐阳\Desktop\模型3.0输出结果\静态权重\Feature_Fusion_Results_20260130_162517\Feature_Weights_Fusion_AutoAlpha.xlsx"
STATIC_WEIGHT_SHEET = "All_Features"   # expected columns: Feature, FinalWeight, Weight_m5

time_tag = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = rf"C:\Users\沐阳\Desktop\模型3.0输出结果\动态权重_{time_tag}"
os.makedirs(OUT_DIR, exist_ok=True)

TOPK = 20

# Rolling-window and bootstrap configuration for dynamic estimation
ROLLING_WINDOW_YEARS = 5       # number of years in each rolling window
BOOTSTRAP_B = 20               # bootstrap replicates
SEEDS = [0, 7, 21]             # seeds for reproducibility in model fitting
TEST_SIZE = 0.2                # held-out fraction for internal XGBoost evaluation

# XGBoost hyperparameters (kept fixed; can be tuned separately)
XGB_PARAMS = dict(
    n_estimators=800,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.7,
    colsample_bytree=0.7,
    reg_alpha=0.0,
    reg_lambda=1.5,
    min_child_weight=1,
    random_state=0,
    n_jobs=-1
)

# State-space model hyperparameters and variance controls
rho = 0.25      # shrinkage toward static baseline theta0 in prior update
q = 0.02        # process noise increment for theta variance
r0 = 0.08       # base observation noise
c_r = 50.0      # supervised-uncertainty amplification factor in observation noise
eps_log = 1e-12

# Drift-to-eta mapping scale (eta = sigmoid(D/d0))
d0 = 0.30

# Baseline unsupervised uncertainty (used to combine supervised/unsupervised evidence)
UNC_UNSUP = 1e-4

# -------- Global Matplotlib style (consistent for publication-quality figures) --------
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
plt.rcParams["axes.unicode_minus"] = False

plt.rcParams["font.size"] = 16
plt.rcParams["axes.titlesize"] = 18
plt.rcParams["axes.labelsize"] = 16
plt.rcParams["xtick.labelsize"] = 14
plt.rcParams["ytick.labelsize"] = 14
plt.rcParams["legend.fontsize"] = 14
plt.rcParams["axes.linewidth"] = 1.0
plt.rcParams["grid.linestyle"] = "--"
plt.rcParams["grid.alpha"] = 0

sns.set_style("whitegrid")


# ---------------------------
# 1) Load data for model fitting and dynamic weight estimation
# ---------------------------
# Read base (static) and dynamic sheets and concatenate into a single DataFrame
df_base = pd.read_excel(DATA_PATH, sheet_name=SHEET_BASE)
df_dyn  = pd.read_excel(DATA_PATH, sheet_name=SHEET_DYN)
df_all  = pd.concat([df_base, df_dyn], ignore_index=True)

# Candidate features are assumed to begin at column index 8 of the base sheet
feature_cols = df_base.columns[8:].tolist()

# Validate presence of required metadata and target columns
for c in META_COLS + TARGETS:
    if c not in df_all.columns:
        raise ValueError(f"Missing required column in dataset: {c}")

# Load Sigmoid-normalized dataset for score calculation (kept separate from training data)
df_base_score = pd.read_excel(DATA_PATH_SCORE, sheet_name=SHEET_BASE)
df_dyn_score  = pd.read_excel(DATA_PATH_SCORE, sheet_name=SHEET_DYN)

# Sanity checks for column consistency between original and Sigmoid-scaled tables
if list(df_base_score.columns) != list(df_base.columns):
    print("Warning: column names/order differ between base raw and Sigmoid-scaled base tables; verify alignment.")
if list(df_dyn_score.columns) != list(df_dyn.columns):
    print("Warning: column names/order differ between dynamic raw and Sigmoid-scaled dynamic tables; verify alignment.")


# ---------------------------
# 2) Load static weights and align feature universe
# ---------------------------
wtab = pd.read_excel(STATIC_WEIGHT_XLSX, sheet_name=STATIC_WEIGHT_SHEET)
required_cols = {"Feature", "FinalWeight", "Weight_m5"}
if not required_cols.issubset(wtab.columns):
    raise ValueError(f"Static weight table is missing required columns: {required_cols - set(wtab.columns)}")

# Normalize and align static weights to the feature set present in the data
wtab["Feature"] = wtab["Feature"].astype(str).str.strip()
w0 = wtab.set_index("Feature")["FinalWeight"].astype(float)
w_unsup = wtab.set_index("Feature")["Weight_m5"].astype(float)

# Restrict features to those that exist both in data and static weight table
common_feats = [f for f in feature_cols if f in w0.index]
if len(common_feats) < len(feature_cols):
    missing = [f for f in feature_cols if f not in w0.index]
    print("Warning: the following features are missing from the static weight table and will be excluded (first 20 shown):", missing[:20])
feature_cols = common_feats

# Ensure non-negative weights and renormalize to sum-to-one
w0 = w0.loc[feature_cols].clip(lower=0.0)
w0 = w0 / w0.sum()

w_unsup = w_unsup.loc[feature_cols].clip(lower=0.0)
w_unsup = w_unsup / w_unsup.sum()

# Initialize theta0 (logit-space representation of static weights) for state-space model
theta0 = np.log(w0.values + eps_log)


# ---------------------------
# 3) Utility functions (deterministic helpers)
# ---------------------------

def softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax returning a probability vector."""
    x = x - np.max(x)
    ex = np.exp(x)
    return ex / np.sum(ex)

def sigmoid(x: float) -> float:
    """Logistic sigmoid mapping real input to (0,1)."""
    return 1.0 / (1.0 + np.exp(-x))

def l1_drift(w_new: np.ndarray, w_old: np.ndarray) -> float:
    """Compute L1 distance between two weight vectors (interpretation: total absolute change)."""
    return float(np.sum(np.abs(w_new - w_old)))

def ensure_series_align(s: pd.Series, feats: list) -> pd.Series:
    """Reindex a Series to the specified feature order and coerce missing entries to zero."""
    s = s.reindex(feats).fillna(0.0).astype(float)
    return s

def normalized_positive(s: pd.Series, eps=1e-12) -> pd.Series:
    """Ensure non-negativity and renormalize a pandas Series to sum to one."""
    s = s.clip(lower=0.0)
    tot = float(s.sum())
    if tot <= 0:
        return pd.Series(np.ones(len(s))/len(s), index=s.index)
    return s / (tot + eps)

def rolling_years(t: int, W: int) -> list[int]:
    """Return a list of years representing the rolling window ending at year t."""
    return list(range(t - W + 1, t + 1))

def calc_beta_from_model_r2(r2_by_target: dict, temp=2.0) -> dict:
    """
    Map per-target predictive performance (R^2) to a normalized importance (beta) vector.
    Uses exponential scaling (temperature parameter) followed by L1 normalization.
    """
    perf = np.array([r2_by_target[t] for t in TARGETS], dtype=float)
    perf_min, perf_max = perf.min(), perf.max()
    perf_scaled = (perf - perf_min) / (perf_max - perf_min + 1e-12)
    b = np.exp(temp * perf_scaled)
    b = b / b.sum()
    return dict(zip(TARGETS, b))

def rank_fusion_importance(imp_by_target: dict, beta_map: dict) -> pd.Series:
    """
    Combine feature importances across targets via weighted rank fusion.
    - imp_by_target: dict[target] -> Series(feature_importance)
    - beta_map: dict[target] -> importance weight for that target
    Returns a normalized importance Series (sums to one).
    """
    feats = imp_by_target[TARGETS[0]].index.tolist()
    rank_mat = np.zeros((len(feats), len(TARGETS)), dtype=float)
    for k, tgt in enumerate(TARGETS):
        # higher model importance => lower rank number (we invert by negating)
        rank_mat[:, k] = rankdata(-imp_by_target[tgt].values, method="average")
    weights = np.array([beta_map[t] for t in TARGETS], dtype=float)
    mean_rank = np.average(rank_mat, axis=1, weights=weights)
    w = 1.0 / (mean_rank + 1e-12)
    w = w / w.sum()
    return pd.Series(w, index=feats, name="g_sup")

def uncertainty_from_bootstrap(g_list: list[pd.Series]) -> float:
    """Estimate supervised uncertainty as the mean per-feature variance across bootstrap replicates."""
    G = np.vstack([g.values for g in g_list])
    return float(np.mean(np.var(G, axis=0)))

def alpha_from_uncertainty_precision(unc_sup: float, unc_unsup: float = UNC_UNSUP) -> float:
    """
    Compute mixing coefficient alpha that balances unsupervised and supervised evidence.
    Interpreted as a posterior weight on unsupervised evidence given precisions (inverse variances).
    """
    unc_sup = max(unc_sup, 1e-12)
    tau_sup = 1.0 / unc_sup
    tau_unsup = 1.0 / unc_unsup
    a = tau_unsup / (tau_unsup + tau_sup)
    return float(np.clip(a, 0.0, 1.0))


# ---------------------------
# 4) XGBoost-based supervised evidence extraction
# ---------------------------

def fit_xgb_and_get_importance(X: pd.DataFrame, y: pd.Series, seed: int) -> tuple[pd.Series, float]:
    """
    Fit an XGBoost regressor and return normalized feature importances and test R^2.
    A simple train/test split is used for a fast estimate of predictive performance.
    """
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

def supervised_evidence_for_year(t: int) -> dict:
    """
    For year t, compute supervised feature evidence using:
      - a rolling-window sample of historical data (length ROLLING_WINDOW_YEARS)
      - multiple seeds and bootstrap replicates
    Returns:
      - g_sup_mean: aggregated supervised importance (Series, normalized)
      - unc_sup: supervised uncertainty (scalar)
      - beta_map: per-target fusion weights derived from R^2
      - r2_mean: mean R^2 by target
    """
    yrs = rolling_years(t, ROLLING_WINDOW_YEARS)
    df_train = df_all[df_all["Year"].isin(yrs)].copy()
    df_train = df_train.dropna(subset=TARGETS)

    X_all = df_train[feature_cols].astype(float).fillna(0.0)

    # Evaluate per-target predictive performance across multiple seeds
    r2_map = {tgt: [] for tgt in TARGETS}
    for seed in SEEDS:
        for tgt in TARGETS:
            y_all = df_train[tgt].astype(float)
            imp, r2 = fit_xgb_and_get_importance(X_all, y_all, seed)
            r2_map[tgt].append(r2)

    r2_mean = {tgt: float(np.mean(r2_map[tgt])) for tgt in TARGETS}
    beta_map = calc_beta_from_model_r2(r2_mean, temp=2.0)

    # Bootstrap to obtain a distribution of fused importance vectors
    g_boot = []
    n = len(df_train)
    rng = np.random.default_rng(12345 + t)

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

    # Aggregate bootstrap results to mean importance and estimate uncertainty
    g_mean = pd.concat(g_boot, axis=1).mean(axis=1)
    g_mean = normalized_positive(g_mean)

    unc_sup = uncertainty_from_bootstrap(g_boot)
    return dict(g_sup_mean=g_mean, unc_sup=unc_sup, beta_map=beta_map, r2_mean=r2_mean)


# ---------------------------
# 5) State-space update across dynamic years
# ---------------------------
years_dyn_all = sorted(df_dyn["Year"].unique().tolist())
years_dyn = [y for y in years_dyn_all if y >= 2019]
if not years_dyn:
    raise ValueError("No dynamic years >= 2019 found in the dynamic sheet.")

last_year = max(years_dyn)
print("Dynamic years:", years_dyn, "Last dynamic year:", last_year)

# Initialize state variables in logit space
theta_prev = theta0.copy()
p_prev = 1.0  # prior variance proxy (scalar for simplicity in this implementation)

# Containers to store intermediate and final results
weights = {}
alphas = {}
drifts = {}
etas = {}
uncs = {}
betas = {}
r2s = {}
dist_to_static = {}

for t in years_dyn:
    print(f"=== Year {t} ===")
    sup = supervised_evidence_for_year(t)
    g_sup = sup["g_sup_mean"]
    unc_sup = sup["unc_sup"]
    beta_map = sup["beta_map"]
    r2_mean = sup["r2_mean"]

    # Compute mixing coefficient alpha based on relative uncertainties
    alpha_t = alpha_from_uncertainty_precision(unc_sup, unc_unsup=UNC_UNSUP)

    # Construct the observation-derived target weight (convex combination of unsupervised and supervised evidence)
    w_star = alpha_t * w_unsup + (1 - alpha_t) * g_sup
    w_star = normalized_positive(w_star)

    # Prior update for theta (shrink toward static baseline theta0)
    theta_prior = (1 - rho) * theta_prev + rho * theta0
    y_obs = np.log(w_star.values + eps_log)

    # Observation noise scales with supervised uncertainty
    r_t = r0 * (1.0 + c_r * unc_sup)

    # Prior variance increment (process noise)
    p_prior = p_prev + q

    # Kalman-like gain (scalar approximation)
    k_gain = p_prior / (p_prior + r_t)

    # Posterior update (linear-Gaussian approximation in logit space)
    theta_post = theta_prior + k_gain * (y_obs - theta_prior)
    p_post = (1 - k_gain) * p_prior

    # Transform posterior theta to probability simplex via softmax
    w_t = softmax(theta_post)
    w_t_series = pd.Series(w_t, index=feature_cols, name=f"w_{t}")

    # Compute diagnostics: L1 drift, eta mapping, uncertainty, distance to static
    w_prev = softmax(theta_prev)
    D_t = l1_drift(w_t, w_prev)
    eta_t = sigmoid(D_t / d0)

    weights[t] = w_t_series
    alphas[t] = alpha_t
    drifts[t] = D_t
    etas[t] = eta_t
    uncs[t] = unc_sup
    betas[t] = beta_map
    r2s[t] = r2_mean
    dist_to_static[t] = float(np.sum(np.abs(w_t - w0.values)))

    # Step forward in time
    theta_prev = theta_post
    p_prev = p_post

# Aggregate dynamic weights into a DataFrame and persist
W = pd.DataFrame(weights).T
W.index.name = "Year"
W.to_excel(os.path.join(OUT_DIR, "DynamicWeights.xlsx"), index=True)

# Save meta diagnostics per year
meta_df = pd.DataFrame({
    "Year": years_dyn,
    "alpha_unsup_share": [alphas[y] for y in years_dyn],
    "unc_sup": [uncs[y] for y in years_dyn],
    "drift_L1": [drifts[y] for y in years_dyn],
    "eta": [etas[y] for y in years_dyn],
    "L1_to_static": [dist_to_static[y] for y in years_dyn],
})
meta_df.to_excel(os.path.join(OUT_DIR, "DynamicMeta_Alpha_Unc_Drift_Eta.xlsx"), index=False)

beta_df = pd.DataFrame([{**{"Year": y}, **{f"beta_{k}": betas[y][k] for k in TARGETS}} for y in years_dyn])
r2_df = pd.DataFrame([{**{"Year": y}, **{f"R2_{k}": r2s[y][k] for k in TARGETS}} for y in years_dyn])
beta_df.to_excel(os.path.join(OUT_DIR, "BetaWeights_fromRollingR2.xlsx"), index=False)
r2_df.to_excel(os.path.join(OUT_DIR, "RollingR2_byTarget.xlsx"), index=False)


# ---------------------------
# 6) Compute scores for dynamic years using Sigmoid-scaled features
# ---------------------------
score_rows = []
for t in years_dyn:
    df_t = df_dyn_score[df_dyn_score["Year"] == t].copy()
    if df_t.empty:
        continue

    Z = df_t[feature_cols].astype(float).fillna(0.0).values

    # Static aggregate, dynamic aggregate, and final blended aggregate (using eta)
    SA = Z @ w0.loc[feature_cols].values
    SD = Z @ W.loc[t, feature_cols].values
    Delta = SD - SA
    eta_t = etas[t]
    Sfinal = SA + eta_t * Delta

    out = df_t[META_COLS + TARGETS + feature_cols].copy()
    out["S_A"] = SA
    out["S_D"] = SD
    out["Delta"] = Delta
    out["eta"] = eta_t
    out["S_final"] = Sfinal
    score_rows.append(out)

scores = pd.concat(score_rows, ignore_index=True)
scores.to_excel(os.path.join(OUT_DIR, "Scores_dynamic_years.xlsx"), index=False)


# ---------------------------
# 7) Compare last_year dynamic weights vs static and produce TOP-K lists
# ---------------------------
if last_year not in W.index:
    raise ValueError(f"Dynamic weight matrix does not contain year {last_year}")

w_last = W.loc[last_year, feature_cols]

compare = pd.DataFrame({
    "Feature": feature_cols,
    "w_static": w0.loc[feature_cols].values,
    "w_dynamic": w_last.values
})
compare["delta"] = compare["w_dynamic"] - compare["w_static"]
compare["abs_delta"] = np.abs(compare["delta"])
compare = compare.sort_values("abs_delta", ascending=False).reset_index(drop=True)
compare.to_excel(os.path.join(OUT_DIR, f"Compare_Static_vs_{last_year}.xlsx"), index=False)

top20_change = compare.head(TOPK)
top20_change.to_excel(os.path.join(OUT_DIR, f"Top{TOPK}_LargestChanges_Static_vs_{last_year}.xlsx"), index=False)

top20_dyn = compare.sort_values("w_dynamic", ascending=False).head(TOPK)
top20_static = compare.sort_values("w_static", ascending=False).head(TOPK)
top20_dyn.to_excel(os.path.join(OUT_DIR, f"Top{TOPK}_ByWeight_{last_year}.xlsx"), index=False)
top20_static.to_excel(os.path.join(OUT_DIR, f"Top{TOPK}_ByWeight_Static.xlsx"), index=False)


# ---------------------------
# 8) Visualizations (publication-ready)
# ---------------------------
def savefig(name):
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, name), dpi=300, bbox_inches="tight")
    plt.close()

plt.figure(figsize=(11,6))
ax = plt.gca()
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
ax.plot(meta_df["Year"], meta_df["alpha_unsup_share"], marker="o", label="alpha")
ax.plot(meta_df["Year"], meta_df["eta"], marker="s", label="eta")
ax.plot(meta_df["Year"], meta_df["drift_L1"], marker="^", label="L1 drift")
ax.plot(meta_df["Year"], meta_df["unc_sup"], marker="d", label="unc_sup")
ax.plot(meta_df["Year"], meta_df["L1_to_static"], marker="x", label="L1")
ax.set_xlabel("Year", fontweight="bold"); ax.set_ylabel("Value", fontweight="bold")
ax.set_title("Dynamic control signals: alpha, eta, drift, supervised uncertainty", fontweight="bold")
ax.legend(loc="center left", bbox_to_anchor=(0.8, 0.75))
savefig("TS_alpha_eta_drift_unc_L1_static.png")

plt.figure(figsize=(7,7))
plt.scatter(compare["w_static"], compare["w_dynamic"], s=12, alpha=0.7)
mx = max(compare["w_static"].max(), compare["w_dynamic"].max())
plt.plot([0, mx], [0, mx], linestyle="--", linewidth=1)
plt.xlabel("Static weight (w0)", fontweight="bold")
plt.ylabel(f"Dynamic weight (w_{last_year})", fontweight="bold")
plt.title(f"{last_year} dynamic weights vs static weights", fontweight="bold")
savefig(f"Scatter_w{last_year}_vs_wstatic.png")

tmp = top20_change.sort_values("delta")
plt.figure(figsize=(10, 0.45*len(tmp)+2))
plt.barh(tmp["Feature"], tmp["delta"], color=np.where(tmp["delta"]>=0, "#54A24B", "#E45756"))
plt.axvline(0, color="black", linewidth=1)
plt.xlabel(f"delta = w_{last_year} - w_static", fontweight="bold")
ax = plt.gca()
for label in ax.get_yticklabels():
    label.set_fontweight("bold")
plt.title(f"Top-{TOPK} features with largest |change| ({last_year} vs static)", fontweight="bold")
savefig(f"Top{TOPK}_DeltaBar_{last_year}.png")

top_feats = top20_dyn["Feature"].tolist()
plt.figure(figsize=(12, 7))
ax = plt.gca()
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
for f in top_feats:
    ax.plot(W.index, W[f], linewidth=1.8, alpha=0.85,
            label=f if len(top_feats)<=10 else None)
ax.set_xlabel("Year", fontweight="bold"); ax.set_ylabel("Weight", fontweight="bold")
ax.set_title(f"Dynamic weight trajectories (Top-{TOPK} by weight in {last_year})", fontweight="bold")
if len(top_feats) <= 10:
    ax.legend()
savefig(f"Trajectories_Top{TOPK}_By_{last_year}.png")

heat = W[top_feats].copy()
plt.figure(figsize=(12, 6))
sns.heatmap(heat, cmap="viridis", cbar_kws={"label":"Weight"})
plt.title(f"Heatmap of dynamic weights (Top-{TOPK} by weight in {last_year})", fontweight="bold")
plt.xlabel("Feature", fontweight="bold"); plt.ylabel("Year", fontweight="bold")
ax = plt.gca()
for label in ax.get_xticklabels():
    label.set_fontweight("bold")
for label in ax.get_yticklabels():
    label.set_fontweight("bold")
savefig(f"Heatmap_Top{TOPK}_By_{last_year}.png")


# ---------------------------
# 9) Simple textual report (metadata & configuration)
# ---------------------------
L1_static_last = float(np.sum(np.abs(w_last.values - w0.loc[feature_cols].values)))
with open(os.path.join(OUT_DIR, "report.txt"), "w", encoding="utf-8") as f:
    f.write("Dynamic weight report (Extreme version, no SHAP)\n")
    f.write("="*70 + "\n")
    f.write(f"Base sheet: {SHEET_BASE} | Dyn sheet: {SHEET_DYN}\n")
    f.write(f"Targets: {TARGETS}\n")
    f.write(f"Num features: {len(feature_cols)}\n")
    f.write(f"Rolling window years: {ROLLING_WINDOW_YEARS}\n")
    f.write(f"Bootstrap B: {BOOTSTRAP_B} | Seeds: {SEEDS}\n")
    f.write(f"State params: rho={rho}, q={q}, r0={r0}, c_r={c_r}\n")
    f.write(f"Eta mapping: sigmoid(D/d0), d0={d0}\n")
    f.write(f"Unc_unsup (precision baseline) = {UNC_UNSUP}\n\n")
    f.write(f"L1 distance between w_{last_year} and w_static: {L1_static_last:.6f} (range [0,2])\n\n")
    f.write(f"Dynamic years: {years_dyn}\n")
    f.write(f"Last dynamic year used for comparison: {last_year}\n\n")
    f.write(f"Outputs written to: {OUT_DIR}\n")


# =========================================================
# 9.1) Dimension (D) mapping for hierarchical aggregation
# =========================================================
D_MAP = {
    "D1": ["PIAV","SIAV","TIAV","GIOV","Domestic_GIOV","HMT_GIOV","Foreign_GIOV","Ind_ent","VAT","Ind_profit","Ind_curr_assets"],
    "D2": ["Reg_pop","Nat_growth_rate","Pop_density"],
    "D3": ["Nonprivate_emp","Private_emp","Unemployed","Avg_staff"],
    "D4": ["PI_emp","SI_emp","TI_emp","Agri_emp","Mining_emp","Manu_emp","Utility_emp",
           "Const_emp","Transpost_emp","ICT_emp","Trade_emp","Hotel_emp","Finance_emp","RE_emp","Lease_emp","R&D_emp","Water_emp",
           "Reservice_emp","Edu_emp","Health_emp","Culture_emp","Public_emp"],
    "D5": ["Total_wages","Avg_wage","Retail_sales","Wholesale_sales","Wholesale_ent","Savings"],
    "D6": ["FAI","RE_inv","Res_inv","New_projects"],
    "D7": ["Domestic_ent","HMT_ent","Foreign_ent","FDI"],
    "D8": ["Grain_output","Oilcrop_output","Veg_output","Fruit_output","Meat_output","Dairy_output","Aquatic_output"],
    "D9": ["Local_rev"],
    "D10": ["Sci_exp","Edu_exp"],
    "D11": ["Loans","Deposits"],
    "D12": ["Univ_num","Mid_sch_num","Univ_teachers","Mid_teachers","Pri_teachers","Univ_enroll","Mid_enroll","Pri_enroll","Voc_enroll"],
    "D13": ["Libraries","Theaters","Library_col","Books_per_100"],
    "D14": ["Health_inst","Hospitals","Hospital_beds","Physicians"],
    "D15": ["Passenger","Rail_passenger","Road_passenger","Freight","Rail_freight","Road_freight"],
    "D16": ["Post_offices","Telecom_rev","Landline_users","Mobile_users","Internet_users"],
    "D17": ["SO2_emission","Waste_treat_rate","Sewage_plant_rate"],
    "D18": ["Pension_ins","Medical_ins","Unemp_ins"],
    "D19": ["Land_area"],
}

def d_sort_key(dname: str):
    return int(dname[1:])  # convert 'D1'->1, 'D2'->2, etc.


# =========================================================
# 9.2) Multi-level scores using dynamic weights (4 output sheets)
# =========================================================
OUT_EXCEL_MULTI = os.path.join(OUT_DIR, "Scores_MultiLevel_Dynamic.xlsx")
OUT_REPORT_MULTI = os.path.join(OUT_DIR, "Scores_MultiLevel_Dynamic_Report.txt")

def run_dynamic_results_4tables():
    """
    Generate four tables using:
      - w0 (static weights),
      - W  (dynamic weights by year),
      - df_dyn_score (Sigmoid-scaled features for dynamic years),
      - feature_cols and META_COLS.

    Sheets:
      1) City-Year per-feature scores (Static A / Dynamic D / Final F) + totals
      2) City-Year aggregated D-dimension scores (A/D/F) + totals
      3) Province-Year aggregated per-feature scores + totals
      4) Province-Year aggregated D-dimension scores + totals
    """
    # ========= 9.2.1 City-Year per-feature scores ==========
    all_rows = []
    for t in years_dyn:
        df_t = df_dyn_score[df_dyn_score["Year"] == t].copy()
        if df_t.empty:
            continue

        Z = df_t[feature_cols].astype(float).fillna(0.0)

        w_static = w0.loc[feature_cols]
        w_dynamic = W.loc[t, feature_cols]
        eta_t = etas[t]
        w_final = w_static + eta_t * (w_dynamic - w_static)

        feat_score_A = Z.mul(w_static, axis=1)
        feat_score_D = Z.mul(w_dynamic, axis=1)
        feat_score_F = Z.mul(w_final, axis=1)

        total_A = feat_score_A.sum(axis=1)
        total_D = feat_score_D.sum(axis=1)
        total_F = feat_score_F.sum(axis=1)

        tmp = df_t[META_COLS].reset_index(drop=True).copy()

        out_A = feat_score_A.copy()
        out_A.columns = [f"ScoreA_{c}" for c in out_A.columns]

        out_D = feat_score_D.copy()
        out_D.columns = [f"ScoreD_{c}" for c in out_D.columns]

        out_F = feat_score_F.copy()
        out_F.columns = [f"ScoreF_{c}" for c in out_F.columns]

        tmp = pd.concat(
            [
                tmp,
                out_A.reset_index(drop=True),
                out_D.reset_index(drop=True),
                out_F.reset_index(drop=True),
                total_A.rename("Total_A").reset_index(drop=True),
                total_D.rename("Total_D").reset_index(drop=True),
                total_F.rename("Total_F").reset_index(drop=True),
            ],
            axis=1
        )
        all_rows.append(tmp)

    table1 = pd.concat(all_rows, ignore_index=True)

    # ========= 9.2.2 City-Year D-dimension scores ==========
    feat_cols_A = [c for c in table1.columns if c.startswith("ScoreA_")]
    feat_cols_D = [c for c in table1.columns if c.startswith("ScoreD_")]
    feat_cols_F = [c for c in table1.columns if c.startswith("ScoreF_")]

    feat_name_from_A = {c.replace("ScoreA_", ""): c for c in feat_cols_A}

    d_score_cols = []
    for d in sorted(D_MAP.keys(), key=d_sort_key):
        feats_in = [f for f in D_MAP[d] if f in feature_cols]
        if len(feats_in) == 0:
            colA = pd.Series(np.zeros(len(table1)), name=f"{d}_A")
            colD = pd.Series(np.zeros(len(table1)), name=f"{d}_D")
            colF = pd.Series(np.zeros(len(table1)), name=f"{d}_F")
        else:
            colsA = [feat_name_from_A[f] for f in feats_in if f in feat_name_from_A]
            colsD = [c.replace("ScoreA_", "ScoreD_") for c in colsA]
            colsF = [c.replace("ScoreA_", "ScoreF_") for c in colsA]

            colA = table1[colsA].sum(axis=1).rename(f"{d}_A")
            colD = table1[colsD].sum(axis=1).rename(f"{d}_D")
            colF = table1[colsF].sum(axis=1).rename(f"{d}_F")

        d_score_cols.extend([colA, colD, colF])

    d_scores_df = pd.concat(d_score_cols, axis=1)

    table2 = pd.concat(
        [
            table1[META_COLS + ["Total_A", "Total_D", "Total_F"]].reset_index(drop=True),
            d_scores_df.reset_index(drop=True),
        ],
        axis=1
    )

    # ========= 9.3 Province-Year per-feature aggregation ==========
    year_col = None
    for c in META_COLS:
        if "year" in c.lower():
            year_col = c
            break
    if year_col is None:
        raise ValueError("Year column not found in META_COLS.")

    group_keys = ["Province", year_col]
    score_cols_feat = feat_cols_A + feat_cols_D + feat_cols_F + ["Total_A", "Total_D", "Total_F"]

    table3 = (
        pd.concat(
            [
                table1[group_keys].reset_index(drop=True),
                table1[score_cols_feat].reset_index(drop=True),
            ],
            axis=1
        )
        .groupby(group_keys, as_index=False)
        .sum()
        .copy()
    )

    # ========= 9.4 Province-Year D-dimension aggregation ==========
    d_cols_all = [c for c in table2.columns if (c.startswith("D") and c.endswith(("_A", "_D", "_F")))]
    table4 = (
        table2[group_keys + d_cols_all + ["Total_A", "Total_D", "Total_F"]]
        .groupby(group_keys, as_index=False)
        .sum()
        .copy()
    )

    # Persist the four tables to an Excel workbook
    with pd.ExcelWriter(OUT_EXCEL_MULTI, engine="openpyxl") as writer:
        table1.to_excel(writer, sheet_name="1_CityYear_FeatureScores", index=False)
        table2.to_excel(writer, sheet_name="2_CityYear_DScores", index=False)
        table3.to_excel(writer, sheet_name="3_ProvYear_FeatureScores", index=False)
        table4.to_excel(writer, sheet_name="4_ProvYear_DScores", index=False)

    # Basic diagnostics on coverage of features in the static weight table
    data_feats = set(feature_cols)
    weight_feats = set(w0.index)
    data_no_weight = sorted(list(data_feats - weight_feats))
    d_missing = {d: [f for f in feats if f not in data_feats] for d, feats in D_MAP.items()}

    with open(OUT_REPORT_MULTI, "w", encoding="utf-8") as f:
        f.write("SUE-EVAL Dynamic Multi-Level Score Report\n")
        f.write("=" * 70 + "\n")
        f.write(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("Outputs:\n")
        f.write(f"  Excel (4 sheets): {OUT_EXCEL_MULTI}\n")
        f.write("    1_CityYear_FeatureScores: city-year feature-level scores (A/D/Final) + totals\n")
        f.write("    2_CityYear_DScores      : city-year dimension-level scores (A/D/Final) + totals\n")
        f.write("    3_ProvYear_FeatureScores: province-year aggregated feature scores + totals\n")
        f.write("    4_ProvYear_DScores      : province-year aggregated dimension scores + totals\n\n")
        f.write("Weight coverage diagnostics:\n")
        f.write(f"  Dynamic-period features: {len(feature_cols)}\n")
        f.write(f"  Features with static weights: {len(data_feats & weight_feats)}\n")
        f.write(f"  Features without static weights: {len(data_no_weight)}\n\n")
        f.write("Missing features per D dimension:\n")
        for d in sorted(D_MAP.keys(), key=d_sort_key):
            f.write(f"  {d}: {len(d_missing[d])}\n")
        f.write("=" * 70 + "\n")

    print("✅ Generated SUE-EVAL dynamic multi-level output (4 sheets)")
    print(f"  - Excel: {OUT_EXCEL_MULTI}")
    print(f"  - Report: {OUT_REPORT_MULTI}")

run_dynamic_results_4tables()


# =========================================================
# 9.7) Static multi-level scores for base period (2000-2018), using Sigmoid-scaled data
# =========================================================
OUT_EXCEL_STATIC_BASE = os.path.join(OUT_DIR, "Scores_MultiLevel_Static_2000_2018.xlsx")
OUT_REPORT_STATIC_BASE = os.path.join(OUT_DIR, "Scores_MultiLevel_Static_2000_2018_Report.txt")

def run_static_results_4tables_for_df(df_input: pd.DataFrame,
                                      years_desc: str,
                                      out_excel_path: str,
                                      out_report_path: str):
    """
    Compute static multi-level scores for the provided DataFrame (using static weights w0).
    Produces four sheets analogous to the dynamic procedure, but only containing A (static) scores.
    """
    all_rows = []
    years_all = sorted(df_input["Year"].unique().tolist())
    for t in years_all:
        df_t = df_input[df_input["Year"] == t].copy()
        if df_t.empty:
            continue

        Z = df_t[feature_cols].astype(float).fillna(0.0)
        w_static = w0.loc[feature_cols]

        feat_score_A = Z.mul(w_static, axis=1)
        total_A = feat_score_A.sum(axis=1)

        tmp = df_t[META_COLS].reset_index(drop=True).copy()

        out_A = feat_score_A.copy()
        out_A.columns = [f"ScoreA_{c}" for c in out_A.columns]

        tmp = pd.concat(
            [
                tmp,
                out_A.reset_index(drop=True),
                total_A.rename("Total_A").reset_index(drop=True),
            ],
            axis=1
        )
        all_rows.append(tmp)

    if len(all_rows) == 0:
        raise ValueError(f"No usable data found for {years_desc}.")

    table1 = pd.concat(all_rows, ignore_index=True)

    feat_cols_A = [c for c in table1.columns if c.startswith("ScoreA_")]
    feat_name_from_A = {c.replace("ScoreA_", ""): c for c in feat_cols_A}

    d_score_cols = []
    for d in sorted(D_MAP.keys(), key=d_sort_key):
        feats_in = [f for f in D_MAP[d] if f in feature_cols]
        if len(feats_in) == 0:
            colA = pd.Series(np.zeros(len(table1)), name=f"{d}_A")
        else:
            colsA = [feat_name_from_A[f] for f in feats_in if f in feat_name_from_A]
            if len(colsA) == 0:
                colA = pd.Series(np.zeros(len(table1)), name=f"{d}_A")
            else:
                colA = table1[colsA].sum(axis=1).rename(f"{d}_A")
        d_score_cols.append(colA)

    d_scores_df = pd.concat(d_score_cols, axis=1)

    table2 = pd.concat(
        [
            table1[META_COLS + ["Total_A"]].reset_index(drop=True),
            d_scores_df.reset_index(drop=True),
        ],
        axis=1
    )

    year_col = None
    for c in META_COLS:
        if "year" in c.lower():
            year_col = c
            break
    if year_col is None:
        raise ValueError("Year column not found in META_COLS.")

    group_keys = ["Province", year_col]
    score_cols_feat = feat_cols_A + ["Total_A"]

    table3 = (
        pd.concat(
            [
                table1[group_keys].reset_index(drop=True),
                table1[score_cols_feat].reset_index(drop=True),
            ],
            axis=1
        )
        .groupby(group_keys, as_index=False)
        .sum()
        .copy()
    )

    d_cols_all = [c for c in table2.columns if (c.startswith("D") and c.endswith("_A"))]
    table4 = (
        table2[group_keys + d_cols_all + ["Total_A"]]
        .groupby(group_keys, as_index=False)
        .sum()
        .copy()
    )

    with pd.ExcelWriter(out_excel_path, engine="openpyxl") as writer:
        table1.to_excel(writer, sheet_name="1_CityYear_FeatureScores", index=False)
        table2.to_excel(writer, sheet_name="2_CityYear_DScores", index=False)
        table3.to_excel(writer, sheet_name="3_ProvYear_FeatureScores", index=False)
        table4.to_excel(writer, sheet_name="4_ProvYear_DScores", index=False)

    data_feats = set(feature_cols)
    weight_feats = set(w0.index)
    data_no_weight = sorted(list(data_feats - weight_feats))
    d_missing = {d: [f for f in feats if f not in data_feats] for d, feats in D_MAP.items()}

    with open(out_report_path, "w", encoding="utf-8") as f:
        f.write(f"SUE-EVAL Static Multi-Level Score Report ({years_desc})\n")
        f.write("=" * 70 + "\n")
        f.write(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("Outputs:\n")
        f.write(f"  Excel (4 sheets): {out_excel_path}\n")
        f.write("    1_CityYear_FeatureScores: city-year feature-level scores (A) + Total_A\n")
        f.write("    2_CityYear_DScores      : city-year dimension-level scores (A) + Total_A\n")
        f.write("    3_ProvYear_FeatureScores: province-year aggregated feature scores (A) + Total_A\n")
        f.write("    4_ProvYear_DScores      : province-year aggregated dimension scores (A) + Total_A\n\n")
        f.write("Weight coverage diagnostics:\n")
        f.write(f"  Number of features: {len(feature_cols)}\n")
        f.write(f"  Features with static weights: {len(data_feats & weight_feats)}\n")
        f.write(f"  Features without static weights: {len(data_no_weight)}\n\n")
        f.write("Missing features per D dimension:\n")
        for d in sorted(D_MAP.keys(), key=d_sort_key):
            f.write(f"  {d}: {len(d_missing[d])}\n")
        f.write("=" * 70 + "\n")

    print(f"✅ Generated SUE-EVAL static multi-level output for {years_desc}")
    print(f"  - Excel: {out_excel_path}")
    print(f"  - Report: {out_report_path}")

def run_static_results_4tables_for_base():
    run_static_results_4tables_for_df(
        df_input=df_base_score,  # use Sigmoid-scaled features for the base period
        years_desc=f"{SHEET_BASE} base period (e.g., 2000-2018)",
        out_excel_path=OUT_EXCEL_STATIC_BASE,
        out_report_path=OUT_REPORT_STATIC_BASE,
    )

run_static_results_4tables_for_base()


# =========================================================
# 9.8) Concatenate static (2000-2018) and dynamic (2019-2023) tables to full sample (2000-2023)
# =========================================================
OUT_EXCEL_FULL = os.path.join(OUT_DIR, "Scores_MultiLevel_Full_2000_2023.xlsx")
OUT_REPORT_FULL = os.path.join(OUT_DIR, "Scores_MultiLevel_Full_2000_2023_Report.txt")

def run_concat_static_dynamic_to_full():
    """
    Concatenate the 4-sheet static workbook (2000-2018) and dynamic workbook (2019-2023)
    into a single 4-sheet workbook covering 2000-2023. Static sheets are reindexed to the
    column layout of dynamic sheets; missing columns are filled with NaN.
    """
    if not os.path.exists(OUT_EXCEL_STATIC_BASE):
        raise FileNotFoundError(f"Static multi-level results not found: {OUT_EXCEL_STATIC_BASE}")
    if not os.path.exists(OUT_EXCEL_MULTI):
        raise FileNotFoundError(f"Dynamic multi-level results not found: {OUT_EXCEL_MULTI}")

    with pd.ExcelFile(OUT_EXCEL_STATIC_BASE) as xf_static, \
         pd.ExcelFile(OUT_EXCEL_MULTI) as xf_dyn:

        sheets = [
            "1_CityYear_FeatureScores",
            "2_CityYear_DScores",
            "3_ProvYear_FeatureScores",
            "4_ProvYear_DScores",
        ]

        full_tables = {}
        for s in sheets:
            df_s = pd.read_excel(xf_static, sheet_name=s)
            df_d = pd.read_excel(xf_dyn, sheet_name=s)

            dyn_cols = list(df_d.columns)

            # Reindex static to dynamic column ordering (adds missing columns as NaN)
            df_s2 = df_s.reindex(columns=dyn_cols)
            df_d2 = df_d.reindex(columns=dyn_cols)

            df_full = pd.concat([df_s2, df_d2], ignore_index=True)

            if "Year" in df_full.columns:
                df_full = df_full.sort_values(
                    ["Year"] + [c for c in df_full.columns if c not in ["Year"]]
                ).reset_index(drop=True)

            full_tables[s] = df_full

    with pd.ExcelWriter(OUT_EXCEL_FULL, engine="openpyxl") as writer:
        for s, df_full in full_tables.items():
            df_full.to_excel(writer, sheet_name=s, index=False)

    with open(OUT_REPORT_FULL, "w", encoding="utf-8") as f:
        f.write("SUE-EVAL Full-Sample (2000-2023) Multi-Level Score Report\n")
        f.write("=" * 70 + "\n")
        f.write(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("Construction details:\n")
        f.write("  - 2000-2018: from Scores_MultiLevel_Static_2000_2018.xlsx (static weights w0; only A scores)\n")
        f.write("  - 2019-2023: from Scores_MultiLevel_Dynamic.xlsx (A/D/Final full set)\n")
        f.write("  - Static sheets are reindexed to the column schema of dynamic sheets; missing columns are NaN\n\n")
        f.write("Output:\n")
        f.write(f"  Excel (4 sheets): {OUT_EXCEL_FULL}\n")
        f.write("    1_CityYear_FeatureScores: city-year feature-level scores (A/D/Final) + totals (2000-2023)\n")
        f.write("    2_CityYear_DScores      : city-year D-dimension scores (A/D/Final) + totals (2000-2023)\n")
        f.write("    3_ProvYear_FeatureScores: province-year feature aggregations (2000-2023)\n")
        f.write("    4_ProvYear_DScores      : province-year D-dimension aggregations (2000-2023)\n")
        f.write("=" * 70 + "\n")

    print("✅ Generated full-sample (2000-2023) multi-level score sheets")
    print(f"  - Excel: {OUT_EXCEL_FULL}")
    print(f"  - Report: {OUT_REPORT_FULL}")

run_concat_static_dynamic_to_full()
# ==========================================================
# 10) Analysis & validation on top of SUE-EVAL dynamic weights
#     - Feature-level change significance (bootstrap CI + FDR)
#     - Weight trajectory robustness across parameter settings
#     - Predictive usefulness: static vs rolling-static vs dynamic vs final
# ==========================================================
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import rankdata as _rankdata
from matplotlib.ticker import MaxNLocator

print("=== 10. Analysis & validation (feature change, robustness, predictive usefulness) ===")

# Ensure output directory exists
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------
# 10.0 Paths for rolling-static weights
# ---------------------------
BASE_OUT_DIR = r"C:\Users\沐阳\Desktop\模型3.0输出结果"
ROLLING_STATIC_ROOT = os.path.join(BASE_OUT_DIR, "滚动静态权重")

# Mapping from target year to directory containing rolling-static weights
ROLLING_STATIC_DIR_MAP = {
    2019: "Feature_Fusion_Results_00-19",
    2020: "Feature_Fusion_Results_00-20",
    2021: "Feature_Fusion_Results_00-21",
    2022: "Feature_Fusion_Results_00-22",
    2023: "Feature_Fusion_Results_00-23",
}
ROLLING_STATIC_FILENAME = "Feature_Weights_Fusion_AutoAlpha.xlsx"
ROLLING_STATIC_SHEET = "All_Features"


def load_rolling_static_weights_for_year(year: int, feature_cols: list[str]) -> pd.Series:
    """
    Load pre-computed rolling-static weights for a given year.
    Returns a normalized weight vector aligned to feature_cols.
    """
    if year not in ROLLING_STATIC_DIR_MAP:
        raise ValueError(f"ROLLING_STATIC_DIR_MAP missing year {year}")
    dir_name = ROLLING_STATIC_DIR_MAP[year]
    fpath = os.path.join(ROLLING_STATIC_ROOT, dir_name, ROLLING_STATIC_FILENAME)
    if not os.path.exists(fpath):
        raise FileNotFoundError(f"Rolling-static file not found: {fpath}")

    wtab_r = pd.read_excel(fpath, sheet_name=ROLLING_STATIC_SHEET)
    if "Feature" not in wtab_r.columns or "FinalWeight" not in wtab_r.columns:
        raise ValueError(f"Rolling-static file {fpath} must contain 'Feature' and 'FinalWeight' columns")

    wtab_r["Feature"] = wtab_r["Feature"].astype(str).str.strip()
    w_full = wtab_r.set_index("Feature")["FinalWeight"].astype(float)

    feats = [f for f in feature_cols if f in w_full.index]
    if len(feats) == 0:
        return pd.Series(dtype=float)

    w = w_full.loc[feats].clip(lower=0.0)
    s = float(w.sum())
    if s <= 0:
        return pd.Series(np.ones(len(feats)) / len(feats), index=feats)
    return w / s


print("Loading rolling-static weights ...")
W_roll_dict = {}
for t in years_dyn:
    try:
        w_roll_t = load_rolling_static_weights_for_year(t, feature_cols)
        if w_roll_t.empty:
            print(f"Warning: rolling-static weights for year {t} have no overlap with feature_cols; skipped.")
            continue
        W_roll_dict[t] = w_roll_t
    except FileNotFoundError as e:
        print(e)
        print(f"Warning: skip rolling-static for year {t}")
    except Exception as e:
        print(f"Error loading rolling-static weights for year {t}: {e}")
        print(f"Warning: skip rolling-static for year {t}")

print("Loaded rolling-static for years:", list(W_roll_dict.keys()))

# ---------------------------
# 10.1 Feature-level change significance (bootstrap CI & FDR)
# ---------------------------
SIG_BOOTSTRAP_B = 2000
np.random.seed(2026)

print("=== 10.1 Feature-level change significance (bootstrap) ===")
year_ref = last_year  # use the last dynamic year as reference

# Observed change vs static for each feature in year_ref
delta_obs = W.loc[year_ref, feature_cols].values - w0.loc[feature_cols].values

# Matrix of dynamic weights over all dynamic years: shape (T, J)
W_mat = W[feature_cols].loc[years_dyn].values  # T x J
T, J = W_mat.shape
Delta_static = delta_obs

# Null distribution: sign-flip bootstrap on year-to-year increments
Delta_null_boot = np.zeros((SIG_BOOTSTRAP_B, J))

# Precompute year-to-year differences
if T >= 2:
    d_mat_base = W_mat[1:, :] - W_mat[:-1, :]  # (T-1, J)
else:
    d_mat_base = np.zeros((0, J))

for b in range(SIG_BOOTSTRAP_B):
    if T < 2:
        # With only one year, the null is degenerate and coincides with the observed change
        Delta_null_boot[b, :] = Delta_static
        continue

    # Random sign for each year-to-year increment (for each feature)
    signs = np.random.choice([-1.0, 1.0], size=d_mat_base.shape)
    d_mat_null = d_mat_base * signs

    # Reconstruct a null path by cumulative summation of perturbed increments
    W_null = np.zeros_like(W_mat)
    W_null[0, :] = W_mat[0, :]
    for t_idx in range(1, T):
        W_null[t_idx, :] = W_null[t_idx - 1, :] + d_mat_null[t_idx - 1, :]

    # Re-normalize to simplex
    W_null = np.clip(W_null, 1e-12, None)
    W_null = (W_null.T / W_null.sum(axis=1)).T  # row-normalization

    idx_ref = years_dyn.index(year_ref)
    w_null_ref = W_null[idx_ref, :]
    Delta_null_boot[b, :] = w_null_ref - w0.loc[feature_cols].values

# Compute two-sided p-values and percentile CIs feature-wise
p_vals, ci_low, ci_high = [], [], []
for j_idx in range(J):
    null_samples = Delta_null_boot[:, j_idx]
    obs = Delta_static[j_idx]
    p = (np.sum(np.abs(null_samples) >= np.abs(obs)) + 1.0) / (SIG_BOOTSTRAP_B + 1.0)
    p_vals.append(p)
    ci_low.append(np.percentile(null_samples, 2.5))
    ci_high.append(np.percentile(null_samples, 97.5))

p_vals = np.array(p_vals)
ci_low = np.array(ci_low)
ci_high = np.array(ci_high)

# Multiple testing correction (Benjamini–Hochberg FDR)
rej, p_adj, _, _ = multipletests(p_vals, alpha=0.05, method="fdr_bh")

sig_df = pd.DataFrame({
    "Feature": feature_cols,
    "Delta_obs": Delta_static,
    "CI_2.5": ci_low,
    "CI_97.5": ci_high,
    "p_raw": p_vals,
    "p_FDR": p_adj,
    "Significant_0.05_FDR": rej
}).sort_values("p_FDR")

sig_df_path = os.path.join(OUT_DIR, f"FeatureChange_Significance_{year_ref}_vs_static.xlsx")
sig_df.to_excel(sig_df_path, index=False)
print(f"Saved feature-level significance table to {sig_df_path}")
print("Number of FDR<0.05 significant features:", int(rej.sum()))

# Volcano plot: observed delta vs -log10(FDR-adjusted p-value)
plt.figure(figsize=(8, 6))
ax = plt.gca()
p_plot = np.clip(sig_df["p_FDR"].values, 1e-12, 1.0)
neglogp = -np.log10(p_plot)
ax.scatter(sig_df["Delta_obs"], neglogp, s=30, alpha=0.8)
sig_mask = sig_df["Significant_0.05_FDR"].values.astype(bool)
if sig_mask.any():
    ax.scatter(sig_df.loc[sig_mask, "Delta_obs"],
               -np.log10(np.clip(sig_df.loc[sig_mask, "p_FDR"].values, 1e-12, 1.0)),
               s=50, facecolors='none', edgecolors='red', linewidths=1.2, label='FDR<0.05')
ax.set_xlabel(f"Delta = w_{year_ref} (dynamic) - w_static", fontweight="bold")
ax.set_ylabel("-log10(p_FDR)", fontweight="bold")
ax.set_title(f"Feature change volcano ({year_ref} vs static)", fontweight="bold")
if sig_mask.any():
    ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, f"Fig10_FeatureChange_Volcano_{year_ref}.png"), dpi=300)
plt.close()

# Top-|Δ| bar plot with bootstrap CIs
top_sig = sig_df[sig_df["Significant_0.05_FDR"]].copy()
if top_sig.shape[0] == 0:
    # If no FDR-significant features, show top-|Δ| features
    top_sig = sig_df.reindex(sig_df["Delta_obs"].abs().sort_values(ascending=False).index).head(10)
    title_tag = "Top-|Δ| (no FDR-significant features)"
else:
    top_sig = top_sig.reindex(top_sig["Delta_obs"].abs().sort_values(ascending=False).index).head(20)
    title_tag = "Top significant changes"

plt.figure(figsize=(8, 0.45 * len(top_sig) + 1.5))
ax = plt.gca()
colors = np.where(top_sig["Delta_obs"] >= 0, "#54A24B", "#E45756")
ax.barh(top_sig["Feature"], top_sig["Delta_obs"], color=colors)
for i, (low, high) in enumerate(zip(top_sig["CI_2.5"], top_sig["CI_97.5"])):
    ax.plot([low, high], [i, i], color='k', linewidth=1)
ax.set_xlabel("Delta (dynamic - static)", fontweight="bold")
ax.set_title(f"{title_tag} ({year_ref}) with bootstrap CI", fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, f"Fig10_FeatureChange_TopSig_{year_ref}.png"), dpi=300)
plt.close()

# ---------------------------------------------
# 10.1 (alternative) Time-series sign-flip test
# ---------------------------------------------
print("=== 10.1 (alternative) Time-series feature-level test ===")

B_TS = 2000  # bootstrap iterations for time-series test
np.random.seed(2027)

# Dynamic weights over all dynamic years: shape (T, J)
W_mat_all = W.loc[years_dyn, feature_cols].values  # T x J
T_ts, J_ts = W_mat_all.shape

# Static weights as baseline
w0_vec = w0.loc[feature_cols].values  # J

# D_{t,j} = w_{t,j}^{dyn} - w_j^{static}
D_mat = W_mat_all - w0_vec[None, :]   # T x J

# Observed test statistic: mean deviation across time for each feature
T_obs = D_mat.mean(axis=0)            # J

# Bootstrap null of T_j under time-wise sign flips
T_null = np.zeros((B_TS, J_ts))
for b in range(B_TS):
    # Random sign for each time point, same sign applied to all features at that time
    signs_t = np.random.choice([-1.0, 1.0], size=(T_ts, 1))  # T x 1
    D_null = D_mat * signs_t                                  # T x J
    T_null[b, :] = D_null.mean(axis=0)

# Two-sided p-values and CIs under the time-series null
p_vals_ts = []
ci_low_ts, ci_high_ts = [], []
for j in range(J_ts):
    null_samples = T_null[:, j]
    obs = T_obs[j]
    p = (np.sum(np.abs(null_samples) >= np.abs(obs)) + 1.0) / (B_TS + 1.0)
    p_vals_ts.append(p)
    ci_low_ts.append(np.percentile(null_samples, 2.5))
    ci_high_ts.append(np.percentile(null_samples, 97.5))

p_vals_ts = np.array(p_vals_ts)
ci_low_ts = np.array(ci_low_ts)
ci_high_ts = np.array(ci_high_ts)

# BH-FDR over features
rej_ts, p_adj_ts, _, _ = multipletests(p_vals_ts, alpha=0.05, method="fdr_bh")

sig_ts_df = pd.DataFrame({
    "Feature": feature_cols,
    "MeanDelta_obs": T_obs,
    "CI_2.5_ts": ci_low_ts,
    "CI_97.5_ts": ci_high_ts,
    "p_raw_ts": p_vals_ts,
    "p_FDR_ts": p_adj_ts,
    "Significant_0.05_FDR_ts": rej_ts
}).sort_values("p_FDR_ts")

sig_ts_path = os.path.join(OUT_DIR, "FeatureChange_TimeSeriesSignificance_vs_static.xlsx")
sig_ts_df.to_excel(sig_ts_path, index=False)
print("Saved time-series feature-level significance table to", sig_ts_path)
print("Number of FDR<0.05 significant features (time-series test):", int(rej_ts.sum()))

# Volcano plot for time-series test (using raw p-values)
plt.figure(figsize=(8, 6))
ax = plt.gca()
p_plot_ts = np.clip(sig_ts_df["p_raw_ts"].values, 1e-12, 1.0)
neglogp_ts = -np.log10(p_plot_ts)
ax.scatter(sig_ts_df["MeanDelta_obs"], neglogp_ts, s=30, alpha=0.8)
sig_mask_ts = sig_ts_df["Significant_0.05_FDR_ts"].values.astype(bool)
if sig_mask_ts.any():
    ax.scatter(sig_ts_df.loc[sig_mask_ts, "MeanDelta_obs"],
               -np.log10(np.clip(sig_ts_df.loc[sig_mask_ts, "p_raw_ts"].values, 1e-12, 1.0)),
               s=50, facecolors='none', edgecolors='red', linewidths=1.2, label='FDR<0.05')
ax.set_xlabel("Mean Delta over years = mean_t (w_t^dyn - w_static)", fontweight="bold")
ax.set_ylabel("-log10(p_raw_ts)", fontweight="bold")
ax.set_title("Time-series feature change test (all dynamic years vs static)", fontweight="bold")
if sig_mask_ts.any():
    ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "Fig10_FeatureChange_Volcano_TimeSeries.png"), dpi=300)
plt.close()

# ---------------------------
# 10.2 Robustness across parameter settings (window sensitivity)
# ---------------------------
print("=== 10.2 Robustness check across parameter settings ===")
ALT_ROLLING_WINDOW_YEARS_LIST = [3, 7]
ALT_BOOTSTRAP_B = 10


def run_dynamic_weights_light(W_years: int, B_boot: int) -> pd.DataFrame:
    """
    Lightweight re-estimation of dynamic weights under alternative settings
    for the rolling window length and bootstrap count (for robustness analysis).
    Uses the same state-space structure as the main dynamic estimation.
    """
    global ROLLING_WINDOW_YEARS, BOOTSTRAP_B
    old_W = ROLLING_WINDOW_YEARS
    old_B = BOOTSTRAP_B
    ROLLING_WINDOW_YEARS = W_years
    BOOTSTRAP_B = B_boot

    theta_prev_l = theta0.copy()
    p_prev_l = 1.0
    weights_l = {}

    for t in years_dyn:
        sup = supervised_evidence_for_year(t)
        g_sup = sup["g_sup_mean"]
        unc_sup = sup["unc_sup"]

        alpha_t = alpha_from_uncertainty_precision(unc_sup, unc_unsup=UNC_UNSUP)
        w_star = alpha_t * w_unsup + (1 - alpha_t) * g_sup
        w_star = normalized_positive(w_star)

        theta_prior = (1 - rho) * theta_prev_l + rho * theta0
        y_obs = np.log(w_star.values + eps_log)

        r_t = r0 * (1.0 + c_r * unc_sup)
        p_prior = p_prev_l + q
        k_gain = p_prior / (p_prior + r_t)

        theta_post = theta_prior + k_gain * (y_obs - theta_prior)
        p_post = (1 - k_gain) * p_prior

        w_t = softmax(theta_post)
        weights_l[t] = pd.Series(w_t, index=feature_cols)

        theta_prev_l = theta_post
        p_prev_l = p_post

    # Restore original settings
    ROLLING_WINDOW_YEARS = old_W
    BOOTSTRAP_B = old_B

    return pd.DataFrame(weights_l).T


W_base = W.copy()
robust_rows = []

for W_alt in ALT_ROLLING_WINDOW_YEARS_LIST:
    print(f"Running lightweight dynamic with window={W_alt}, B={ALT_BOOTSTRAP_B} ...")
    W_alt_mat = run_dynamic_weights_light(W_alt, ALT_BOOTSTRAP_B)

    # Align years between baseline and alternative
    common_years = [y for y in years_dyn if y in W_alt_mat.index and y in W_base.index]
    for t in common_years:
        v_base = W_base.loc[t, feature_cols].values
        v_alt = W_alt_mat.loc[t, feature_cols].values

        cos_sim = np.dot(v_base, v_alt) / (np.linalg.norm(v_base) * np.linalg.norm(v_alt) + 1e-12)
        rank_base = _rankdata(v_base)
        rank_alt = _rankdata(v_alt)
        spearman = np.corrcoef(rank_base, rank_alt)[0, 1]

        robust_rows.append({
            "Year": t,
            "W_alt": W_alt,
            "B_alt": ALT_BOOTSTRAP_B,
            "CosineSim": cos_sim,
            "Spearman": spearman
        })

robust_df = pd.DataFrame(robust_rows)
robust_df_path = os.path.join(OUT_DIR, "Robustness_Weights_WindowSensitivity.xlsx")
robust_df.to_excel(robust_df_path, index=False)
print("Saved robustness diagnostics to", robust_df_path)

# Time-series similarity plot (mean ± 1 SD across alternative windows)
plt.figure(figsize=(10, 5))
ax = plt.gca()
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
for metric in ["CosineSim", "Spearman"]:
    pivot = robust_df.pivot_table(index="Year", columns="W_alt", values=metric)
    mean_series = pivot.mean(axis=1)
    std_series = pivot.std(axis=1)
    ax.plot(mean_series.index, mean_series.values, marker='o', label=f"{metric} (mean across alt W)")
    ax.fill_between(mean_series.index, mean_series - std_series, mean_series + std_series, alpha=0.2)
ax.set_xlabel("Year", fontweight="bold")
ax.set_ylabel("Similarity", fontweight="bold")
ax.set_title("Robustness: similarity between baseline and alt-window dynamics", fontweight="bold")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "Fig10_Robustness_Similarity_TimeSeries.png"), dpi=300)
plt.close()

# Boxplot of cosine similarities by alternative window length
plt.figure(figsize=(8, 4))
ax = plt.gca()
sns.boxplot(x="W_alt", y="CosineSim", data=robust_df, ax=ax)
ax.set_title("Distribution of Cosine Similarities across years (by W_alt)", fontweight="bold")
ax.set_xlabel(ax.get_xlabel(), fontweight="bold")
ax.set_ylabel(ax.get_ylabel(), fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "Fig10_Robustness_Cosine_Box_byWindow.png"), dpi=300)
plt.close()

# ---------------------------
# 10.3 Predictive usefulness:
#      static vs rolling-static vs dynamic vs final
# ---------------------------
print("=== 10.3 Predictive usefulness: static vs rolling-static vs dynamic vs final ===")


def dm_test(e1: np.ndarray, e2: np.ndarray):
    """
    Simplified Diebold–Mariano test under squared-error loss and iid assumption.
    H0: E[(e1^2 - e2^2)] = 0, i.e., equal expected squared loss.
    Returns (DM statistic, two-sided p-value).
    """
    from scipy.stats import norm
    d = (e1**2 - e2**2)
    n = len(d)
    if n < 5:
        return np.nan, np.nan
    d_bar = d.mean()
    sd = d.std(ddof=1)
    if sd <= 0:
        return np.nan, np.nan
    dm_stat = d_bar / (sd / np.sqrt(n))
    p_val = 2 * (1 - norm.cdf(np.abs(dm_stat)))
    return float(dm_stat), float(p_val)


MISSING_LOG_PATH = os.path.join(OUT_DIR, "RollingStatic_MissingFeatures_Log.xlsx")
missing_log_rows = []


def ensure_features_for_SR(df_year: pd.DataFrame,
                           w_roll: pd.Series,
                           feature_cols: list[str]) -> pd.DataFrame:
    """
    Ensure that df_year contains all features required to compute S_R
    (rolling-static index). For any missing feature:
      - if present in df_all, fill with its mean over df_all;
      - otherwise fill with zero.
    Also logs which features were filled for diagnostic purposes.
    """
    feats_needed = [f for f in feature_cols if f in w_roll.index]
    feats_missing = [f for f in feats_needed if f not in df_year.columns]

    filled_info = {}
    for f in feats_missing:
        if f in df_all.columns:
            col_mean = float(df_all[f].dropna().mean())
            df_year[f] = col_mean
            filled_info[f] = ("mean_filled_df_all", col_mean)
        else:
            df_year[f] = 0.0
            filled_info[f] = ("zero_filled", 0.0)

    if feats_missing:
        missing_log_rows.append({
            "Year": int(df_year["Year"].iloc[0]),
            "MissingCount": len(feats_missing),
            "MissingSample": ", ".join(feats_missing[:20]) + ("..." if len(feats_missing) > 20 else ""),
            "FilledInfo": str(filled_info)
        })

    return df_year


def fit_and_eval(x_tr, x_te, y_tr, y_te):
    """
    Fit a simple OLS model: y = alpha + beta * x on training data,
    then compute in-sample R² and out-of-sample MSE, RMSE,
    and the vector of out-of-sample errors.
    """
    X_tr = sm.add_constant(x_tr)
    X_te = sm.add_constant(x_te)
    model = sm.OLS(y_tr, X_tr).fit()
    y_hat_te = model.predict(X_te)

    r2 = model.rsquared
    r2_adj = model.rsquared_adj
    mse_te = mean_squared_error(y_te, y_hat_te)
    rmse_te = mean_squared_error(y_te, y_hat_te, squared=False)
    e_te = y_te - y_hat_te
    return model, r2, r2_adj, mse_te, rmse_te, e_te


TARGETS_FOR_EVAL = ["GDP", "Local_exp", "Post_rev", "Wastewater"]
all_pred_results = []

for tgt in TARGETS_FOR_EVAL:
    print(f"  -> Evaluating target: {tgt}")
    pred_rows = []

    for t in years_dyn:
        df_t = scores[scores["Year"] == t].copy()
        if df_t.empty or tgt not in df_t.columns:
            continue

        if t not in W_roll_dict:
            print(f"Warning: no rolling-static weights for year {t}, skip for target {tgt}")
            continue
        w_roll_t = W_roll_dict[t]

        # Ensure all features required for S_R exist in df_t
        df_t = ensure_features_for_SR(df_t, w_roll_t, feature_cols)

        feat_used = [f for f in feature_cols if f in w_roll_t.index]
        if len(feat_used) == 0:
            print(f"Year {t}, target {tgt}: no overlapping features for rolling-static, skip.")
            continue

        # Compute rolling-static index S_R
        Z_t = df_t[feat_used].astype(float).values
        w_roll_vec = w_roll_t.loc[feat_used].values
        df_t["S_R"] = Z_t @ w_roll_vec

        # Pseudo out-of-sample split at city level (to avoid within-city leakage)
        cities = df_t["City"].unique()
        if len(cities) >= 10:
            train_c, test_c = train_test_split(cities, test_size=0.2, random_state=2026)
            df_train = df_t[df_t["City"].isin(train_c)]
            df_test  = df_t[df_t["City"].isin(test_c)]
        else:
            df_train = df_t
            df_test  = df_t

        y_tr = df_train[tgt].astype(float).values
        y_te = df_test[tgt].astype(float).values

        SA_tr = df_train["S_A"].astype(float).values
        SR_tr = df_train["S_R"].astype(float).values
        SD_tr = df_train["S_D"].astype(float).values
        SF_tr = df_train["S_final"].astype(float).values

        SA_te = df_test["S_A"].astype(float).values
        SR_te = df_test["S_R"].astype(float).values
        SD_te = df_test["S_D"].astype(float).values
        SF_te = df_test["S_final"].astype(float).values

        # Fit separate OLS models using different indices as regressors
        m_A, r2_A, r2adj_A, mse_A, rmse_A, e_A = fit_and_eval(SA_tr, SA_te, y_tr, y_te)
        m_R, r2_R, r2adj_R, mse_R, rmse_R, e_R = fit_and_eval(SR_tr, SR_te, y_tr, y_te)
        m_D, r2_D, r2adj_D, mse_D, rmse_D, e_D = fit_and_eval(SD_tr, SD_te, y_tr, y_te)
        m_F, r2_F, r2adj_F, mse_F, rmse_F, e_F = fit_and_eval(SF_tr, SF_te, y_tr, y_te)

        # Diebold–Mariano tests: compare Final vs (Static, Rolling-static, Dynamic)
        dm_FA, p_FA = dm_test(e_F, e_A)
        dm_FR, p_FR = dm_test(e_F, e_R)
        dm_FD, p_FD = dm_test(e_F, e_D)

        pred_rows.append({
            "Target": tgt,
            "Year": t,
            "R2_A_static": r2_A,
            "R2_R_rollstatic": r2_R,
            "R2_D_dynamic": r2_D,
            "R2_F_final": r2_F,
            "R2adj_A_static": r2adj_A,
            "R2adj_R_rollstatic": r2adj_R,
            "R2adj_D_dynamic": r2adj_D,
            "R2adj_F_final": r2adj_F,
            "MSE_A_oos": mse_A,
            "MSE_R_oos": mse_R,
            "MSE_D_oos": mse_D,
            "MSE_F_oos": mse_F,
            "RMSE_A_oos": rmse_A,
            "RMSE_R_oos": rmse_R,
            "RMSE_D_oos": rmse_D,
            "RMSE_F_oos": rmse_F,
            "DM_F_vs_A": dm_FA,
            "p_DM_F_vs_A": p_FA,
            "DM_F_vs_R": dm_FR,
            "p_DM_F_vs_R": p_FR,
            "DM_F_vs_D": dm_FD,
            "p_DM_F_vs_D": p_FD,
        })

    pred_df_tgt = pd.DataFrame(pred_rows)
    pred_df_tgt.to_excel(os.path.join(OUT_DIR, f"PredictiveUsefulness_{tgt}.xlsx"), index=False)
    all_pred_results.append(pred_df_tgt)

# Concatenate predictive results across all targets
if all_pred_results:
    pred_all = pd.concat(all_pred_results, ignore_index=True)
else:
    pred_all = pd.DataFrame(columns=[
        "Target","Year",
        "R2_A_static","R2_R_rollstatic","R2_D_dynamic","R2_F_final",
        "R2adj_A_static","R2adj_R_rollstatic","R2adj_D_dynamic","R2adj_F_final",
        "MSE_A_oos","MSE_R_oos","MSE_D_oos","MSE_F_oos",
        "RMSE_A_oos","RMSE_R_oos","RMSE_D_oos","RMSE_F_oos",
        "DM_F_vs_A","p_DM_F_vs_A",
        "DM_F_vs_R","p_DM_F_vs_R",
        "DM_F_vs_D","p_DM_F_vs_D"
    ])

pred_all_path = os.path.join(OUT_DIR, "PredictiveUsefulness_AllTargets.xlsx")
pred_all.to_excel(pred_all_path, index=False)
print("Saved predictive usefulness tables for all targets to", pred_all_path)

# Log any imputation performed when computing rolling-static indices
if missing_log_rows:
    pd.DataFrame(missing_log_rows).to_excel(MISSING_LOG_PATH, index=False)
    print("Saved missing-features log to", MISSING_LOG_PATH)

# ----------------------------------------------------------
# Visualization for predictive usefulness
# ----------------------------------------------------------
# 1) Per-target R² and MSE time series, plus average DM p-values (Final vs others)
for tgt in TARGETS_FOR_EVAL:
    pred_t = pred_all[pred_all["Target"] == tgt].copy()
    if pred_t.empty:
        continue

    # R² time series (per target, per year)
    plt.figure(figsize=(10, 5))
    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.plot(pred_t["Year"], pred_t["R2_A_static"], marker='o', label="R2_A (static, 00-18)")
    ax.plot(pred_t["Year"], pred_t["R2_R_rollstatic"], marker='s', label="R2_R (rolling-static, 00-t)")
    ax.plot(pred_t["Year"], pred_t["R2_D_dynamic"], marker='^', label="R2_D (dynamic)")
    ax.plot(pred_t["Year"], pred_t["R2_F_final"], marker='d', label="R2_F (final)")
    ax.set_xlabel("Year", fontweight="bold")
    ax.set_ylabel("In-sample R-squared", fontweight="bold")
    ax.set_title(f"Predictive goodness-of-fit (R²) over years (Target = {tgt})", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"Fig10_PredictiveUsefulness_R2_TimeSeries_{tgt}.png"), dpi=300)
    plt.close()

    # MSE time series (pseudo out-of-sample via city-based split)
    plt.figure(figsize=(10, 5))
    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.plot(pred_t["Year"], pred_t["MSE_A_oos"], marker='o', label="MSE_A (static)")
    ax.plot(pred_t["Year"], pred_t["MSE_R_oos"], marker='s', label="MSE_R (rolling-static)")
    ax.plot(pred_t["Year"], pred_t["MSE_D_oos"], marker='^', label="MSE_D (dynamic)")
    ax.plot(pred_t["Year"], pred_t["MSE_F_oos"], marker='d', label="MSE_F (final)")
    ax.set_xlabel("Year", fontweight="bold")
    ax.set_ylabel("Out-of-sample MSE", fontweight="bold")
    ax.set_title(f"Predictive performance (MSE) over years (Target = {tgt})", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"Fig10_PredictiveUsefulness_MSE_TimeSeries_{tgt}.png"), dpi=300)
    plt.close()

    # Aggregated DM p-values: Final vs {Static, Rolling-static, Dynamic} (averaged across years)
    plt.figure(figsize=(8, 4))
    ax = plt.gca()
    dm_cols = ["p_DM_F_vs_A", "p_DM_F_vs_R", "p_DM_F_vs_D"]
    labels = ["F vs A", "F vs R", "F vs D"]
    mean_p = pred_t[dm_cols].mean(axis=0)
    ax.bar(labels, mean_p.values, color=["#4c72b0", "#55a868", "#c44e52"])
    ax.axhline(0.05, color="red", linestyle="--", linewidth=1, label="0.05 threshold")
    ax.set_ylabel("Mean DM p-value (Final vs baseline)", fontweight="bold")
    ax.set_title(f"Diebold–Mariano test (Target = {tgt})", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"Fig10_PredictiveUsefulness_DM_Bar_{tgt}.png"), dpi=300)
    plt.close()

# 2) Overall (across targets) mean R² and mean MSE by year
if not pred_all.empty:
    overall_rows = []
    for t in sorted(pred_all["Year"].unique()):
        sub = pred_all[pred_all["Year"] == t]
        if sub.empty:
            continue
        overall_rows.append({
            "Year": t,
            "R2_A_mean": sub["R2_A_static"].mean(),
            "R2_R_mean": sub["R2_R_rollstatic"].mean(),
            "R2_D_mean": sub["R2_D_dynamic"].mean(),
            "R2_F_mean": sub["R2_F_final"].mean(),
            "MSE_A_mean": sub["MSE_A_oos"].mean(),
            "MSE_R_mean": sub["MSE_R_oos"].mean(),
            "MSE_D_mean": sub["MSE_D_oos"].mean(),
            "MSE_F_mean": sub["MSE_F_oos"].mean()
        })
    overall_df = pd.DataFrame(overall_rows)
    overall_df.to_excel(os.path.join(OUT_DIR, "PredictiveUsefulness_MeanAcrossTargets_byYear.xlsx"), index=False)

    # Mean in-sample R² across targets
    plt.figure(figsize=(10, 5))
    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.plot(overall_df["Year"], overall_df["R2_A_mean"], marker='o', label="Static (00-18)")
    ax.plot(overall_df["Year"], overall_df["R2_R_mean"], marker='s', label="Rolling-static (00-t)")
    ax.plot(overall_df["Year"], overall_df["R2_D_mean"], marker='^', label="Dynamic")
    ax.plot(overall_df["Year"], overall_df["R2_F_mean"], marker='d', label="Final")
    ax.set_xlabel("Year", fontweight="bold")
    ax.set_ylabel("Mean in-sample R-squared", fontweight="bold")
    ax.set_title("Predictive goodness-of-fit (mean R² across all targets)", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "Fig10_PredictiveUsefulness_R2_TimeSeries_AllTargetsMean.png"), dpi=300)
    plt.close()

    # Mean pseudo out-of-sample MSE across targets
    plt.figure(figsize=(10, 5))
    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.plot(overall_df["Year"], overall_df["MSE_A_mean"], marker='o', label="Static (00-18)")
    ax.plot(overall_df["Year"], overall_df["MSE_R_mean"], marker='s', label="Rolling-static (00-t)")
    ax.plot(overall_df["Year"], overall_df["MSE_D_mean"], marker='^', label="Dynamic")
    ax.plot(overall_df["Year"], overall_df["MSE_F_mean"], marker='d', label="Final")
    ax.set_xlabel("Year", fontweight="bold")
    ax.set_ylabel("Mean out-of-sample MSE", fontweight="bold")
    ax.set_title("Predictive performance (mean MSE across all targets)", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "Fig10_PredictiveUsefulness_MSE_TimeSeries_AllTargetsMean.png"), dpi=300)
    plt.close()

print("=== 10.3 Predictive usefulness finished. ===")
print("Saved predictive usefulness figures and tables to:", OUT_DIR)

# ----------------------------------------------------------
# 10.4 Rolling time-forecast validation (true out-of-time)
# ----------------------------------------------------------
print("=== 10.4 Rolling time-forecast validation (static vs rolling-static vs dynamic vs final) ===")

def rolling_time_forecast_validation(scores: pd.DataFrame,
                                     years_dyn: list[int],
                                     target: str,
                                     min_train_years: int = 2):
    """
    Rolling time-forecast validation for a single target variable.
    For each forecast origin:
      - Train on all years from years_dyn[0] up to TrainYears_end (inclusive).
      - Test on the subsequent year (TestYear).
      - For each index variant S_* (Static, Rolling-static, Dynamic, Final),
        fit OLS: Y_it = α + β * S_variant_it + ε_it on the training panel,
        evaluate on the test year.
    Returns:
      A DataFrame with out-of-time R², MSE, RMSE and DM-test statistics
      comparing Final vs the baselines.
    """
    rows = []
    years_sorted = sorted(years_dyn)

    for end_idx in range(min_train_years, len(years_sorted)):
        train_years = years_sorted[:end_idx]
        test_year  = years_sorted[end_idx]

        df_train = scores[scores["Year"].isin(train_years)].copy()
        df_test  = scores[scores["Year"] == test_year].copy()

        if df_train.empty or df_test.empty or target not in df_train.columns:
            continue

        needed_cols = ["S_A", "S_D", "S_final"]
        if not all(c in df_train.columns for c in needed_cols):
            continue
        if not all(c in df_test.columns for c in needed_cols):
            continue

        has_SR = ("S_R" in df_train.columns) and ("S_R" in df_test.columns)

        y_tr = df_train[target].astype(float).values
        y_te = df_test[target].astype(float).values

        SA_tr = df_train["S_A"].astype(float).values
        SD_tr = df_train["S_D"].astype(float).values
        SF_tr = df_train["S_final"].astype(float).values

        SA_te = df_test["S_A"].astype(float).values
        SD_te = df_test["S_D"].astype(float).values
        SF_te = df_test["S_final"].astype(float).values

        # Fit and evaluate time-forecast regression for each index variant
        _, r2_A, r2adj_A, mse_A, rmse_A, e_A = fit_and_eval(SA_tr, SA_te, y_tr, y_te)
        _, r2_D, r2adj_D, mse_D, rmse_D, e_D = fit_and_eval(SD_tr, SD_te, y_tr, y_te)
        _, r2_F, r2adj_F, mse_F, rmse_F, e_F = fit_and_eval(SF_tr, SF_te, y_tr, y_te)

        row = {
            "Target": target,
            "TrainYears_start": min(train_years),
            "TrainYears_end": max(train_years),
            "TestYear": test_year,
            "n_train": len(df_train),
            "n_test": len(df_test),
            "has_S_R": has_SR,
            "R2_time_A": r2_A,
            "R2_time_D": r2_D,
            "R2_time_F": r2_F,
            "R2adj_time_A": r2adj_A,
            "R2adj_time_D": r2adj_D,
            "R2adj_time_F": r2adj_F,
            "MSE_time_A": mse_A,
            "MSE_time_D": mse_D,
            "MSE_time_F": mse_F,
            "RMSE_time_A": rmse_A,
            "RMSE_time_D": rmse_D,
            "RMSE_time_F": rmse_F,
        }

        # Rolling-static variant if S_R is available
        if has_SR:
            SR_tr = df_train["S_R"].astype(float).values
            SR_te = df_test["S_R"].astype(float).values
            _, r2_R, r2adj_R, mse_R, rmse_R, e_R = fit_and_eval(SR_tr, SR_te, y_tr, y_te)
            row.update({
                "R2_time_R": r2_R,
                "R2adj_time_R": r2adj_R,
                "MSE_time_R": mse_R,
                "RMSE_time_R": rmse_R,
            })
        else:
            row.update({
                "R2_time_R": np.nan,
                "R2adj_time_R": np.nan,
                "MSE_time_R": np.nan,
                "RMSE_time_R": np.nan,
            })

        # DM tests: Final vs A, D, and R (if available)
        dm_FA, p_FA = dm_test(e_F, e_A)
        dm_FD, p_FD = dm_test(e_F, e_D)
        row.update({
            "DM_time_F_vs_A": dm_FA,
            "p_DM_time_F_vs_A": p_FA,
            "DM_time_F_vs_D": dm_FD,
            "p_DM_time_F_vs_D": p_FD,
        })

        if has_SR:
            dm_FR, p_FR = dm_test(e_F, e_R)
            row.update({
                "DM_time_F_vs_R": dm_FR,
                "p_DM_time_F_vs_R": p_FR,
            })
        else:
            row.update({
                "DM_time_F_vs_R": np.nan,
                "p_DM_time_F_vs_R": np.nan,
            })

        rows.append(row)

    return pd.DataFrame(rows)


# Execute rolling time-forecast validation for all targets
time_forecast_results = []
for tgt in TARGETS_FOR_EVAL:
    print(f"  -> Rolling time-forecast for target: {tgt}")
    df_res_tgt = rolling_time_forecast_validation(scores, years_dyn, tgt, min_train_years=2)
    out_path_tgt = os.path.join(OUT_DIR, f"TimeForecast_Rolling_{tgt}.xlsx")
    df_res_tgt.to_excel(out_path_tgt, index=False)
    print(f"     Saved rolling time-forecast results to: {out_path_tgt}")
    time_forecast_results.append(df_res_tgt)

# Concatenate rolling time-forecast results across targets
if time_forecast_results:
    time_forecast_all = pd.concat(time_forecast_results, ignore_index=True)
else:
    time_forecast_all = pd.DataFrame()

time_forecast_all_path = os.path.join(OUT_DIR, "TimeForecast_Rolling_AllTargets.xlsx")
time_forecast_all.to_excel(time_forecast_all_path, index=False)
print("Saved rolling time-forecast summary for all targets to:", time_forecast_all_path)

# ----------------------------------------------------------
# Visualization: R², MSE, and DM for rolling time-forecast
# ----------------------------------------------------------
if not time_forecast_all.empty:
    # Aggregate by TestYear: mean R² and MSE across targets
    rows_year = []
    for y in sorted(time_forecast_all["TestYear"].unique()):
        sub = time_forecast_all[time_forecast_all["TestYear"] == y]
        rows_year.append({
            "TestYear": y,
            "R2_time_A_mean": sub["R2_time_A"].mean(),
            "R2_time_D_mean": sub["R2_time_D"].mean(),
            "R2_time_F_mean": sub["R2_time_F"].mean(),
            "R2_time_R_mean": sub["R2_time_R"].mean(),
            "MSE_time_A_mean": sub["MSE_time_A"].mean(),
            "MSE_time_D_mean": sub["MSE_time_D"].mean(),
            "MSE_time_F_mean": sub["MSE_time_F"].mean(),
            "MSE_time_R_mean": sub["MSE_time_R"].mean(),
        })
    overall_time_df = pd.DataFrame(rows_year)
    overall_time_df.to_excel(os.path.join(OUT_DIR, "TimeForecast_Rolling_MeanStats_byTestYear.xlsx"), index=False)

    # Mean out-of-time R² (by test year, averaged across targets)
    plt.figure(figsize=(10, 5))
    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.plot(overall_time_df["TestYear"], overall_time_df["R2_time_A_mean"], marker='o', label="Static (S_A)")
    ax.plot(overall_time_df["TestYear"], overall_time_df["R2_time_R_mean"], marker='s', label="Rolling-static (S_R)")
    ax.plot(overall_time_df["TestYear"], overall_time_df["R2_time_D_mean"], marker='^', label="Dynamic (S_D)")
    ax.plot(overall_time_df["TestYear"], overall_time_df["R2_time_F_mean"], marker='d', label="Final (S_final)")
    ax.set_xlabel("Test Year", fontweight="bold")
    ax.set_ylabel("Mean out-of-time R-squared", fontweight="bold")
    ax.set_title("Rolling time-forecast performance (mean R² across targets)", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "Fig10_TimeForecast_R2_TimeSeries_AllTargetsMean.png"), dpi=300)
    plt.close()

    # Mean out-of-time MSE (by test year, averaged across targets)
    plt.figure(figsize=(10, 5))
    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.plot(overall_time_df["TestYear"], overall_time_df["MSE_time_A_mean"], marker='o', label="Static (S_A)")
    ax.plot(overall_time_df["TestYear"], overall_time_df["MSE_time_R_mean"], marker='s', label="Rolling-static (S_R)")
    ax.plot(overall_time_df["TestYear"], overall_time_df["MSE_time_D_mean"], marker='^', label="Dynamic (S_D)")
    ax.plot(overall_time_df["TestYear"], overall_time_df["MSE_time_F_mean"], marker='d', label="Final (S_final)")
    ax.set_xlabel("Test Year", fontweight="bold")
    ax.set_ylabel("Mean out-of-time MSE", fontweight="bold")
    ax.set_title("Rolling time-forecast performance (mean MSE across targets)", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "Fig10_TimeForecast_MSE_TimeSeries_AllTargetsMean.png"), dpi=300)
    plt.close()

    # Optional: DM p-value heatmap (Final vs baselines) across targets and test years
    dm_heat_rows = []
    for tgt in TARGETS_FOR_EVAL:
        sub = time_forecast_all[time_forecast_all["Target"] == tgt]
        if sub.empty:
            continue
        for _, r in sub.iterrows():
            dm_heat_rows.append({
                "Target": tgt,
                "TestYear": int(r["TestYear"]),
                "p_F_vs_A": r["p_DM_time_F_vs_A"],
                "p_F_vs_R": r["p_DM_time_F_vs_R"],
                "p_F_vs_D": r["p_DM_time_F_vs_D"],
            })
    dm_heat_df = pd.DataFrame(dm_heat_rows)
    if not dm_heat_df.empty:
        dm_long = dm_heat_df.melt(
            id_vars=["Target", "TestYear"],
            value_vars=["p_F_vs_A", "p_F_vs_R", "p_F_vs_D"],
            var_name="Comparison",
            value_name="p_value"
        )
        dm_long["p_value_capped"] = dm_long["p_value"].clip(upper=0.2)

        plt.figure(figsize=(10, 6))
        ax = plt.gca()
        pivot = dm_long.pivot_table(
            index=["Target"],
            columns=["TestYear", "Comparison"],
            values="p_value_capped"
        )
        sns.heatmap(
            pivot,
            annot=False,
            cmap="viridis_r",
            vmin=0,
            vmax=0.2,
            ax=ax,
            cbar_kws={"label": "DM p-value (capped at 0.2)"}
        )
        ax.set_title("DM test p-values: Final vs baselines (rolling time-forecast)", fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "Fig10_TimeForecast_DM_Heatmap.png"), dpi=300)
        plt.close()

print("=== 10.4 Rolling time-forecast validation finished. ===")

# ----------------------------------------------------------
# 10.6 Rank stability and group-wise FDR control
# ----------------------------------------------------------
print("=== 10.6 Rank stability & group-wise FDR control ===")

# 10.6.1 City-level rank stability under different indices

def compute_city_rank_stability(scores: pd.DataFrame,
                                years_dyn: list[int],
                                score_cols: list[str]) -> dict:
    """
    Compute rank stability of cities under different index variants.

    Returns a dictionary with:
      - 'within_index_year_pair_spearman': DataFrame with Spearman rank
        correlation of city rankings across (Year1, Year2) for each index.
      - 'within_year_index_pair_spearman': DataFrame with Spearman rank
        correlation between index pairs within the same year.
    """
    res_year_pair = []
    res_index_pair = []

    years_sorted = sorted(years_dyn)

    # (1) For each index, rank stability across years (pairwise year comparisons)
    for sc in score_cols:
        for i in range(len(years_sorted)):
            for j in range(i + 1, len(years_sorted)):
                y1, y2 = years_sorted[i], years_sorted[j]
                df1 = scores[scores["Year"] == y1][["City", sc]].dropna()
                df2 = scores[scores["Year"] == y2][["City", sc]].dropna()

                common_cities = sorted(set(df1["City"]) & set(df2["City"]))
                if len(common_cities) < 5:
                    continue
                s1 = df1.set_index("City").loc[common_cities][sc].values
                s2 = df2.set_index("City").loc[common_cities][sc].values

                r1 = _rankdata(s1)
                r2 = _rankdata(s2)
                rho = np.corrcoef(r1, r2)[0, 1]

                res_year_pair.append({
                    "ScoreType": sc,
                    "Year1": y1,
                    "Year2": y2,
                    "n_common_cities": len(common_cities),
                    "Spearman_rank": rho
                })

    # (2) For each year, rank agreement across index pairs (e.g., S_A vs S_D vs S_final)
    score_pairs = []
    for i in range(len(score_cols)):
        for j in range(i + 1, len(score_cols)):
            score_pairs.append((score_cols[i], score_cols[j]))

    for y in years_sorted:
        dfy = scores[scores["Year"] == y][["City"] + score_cols].dropna()
        if dfy.empty:
            continue
        for sc1, sc2 in score_pairs:
            s1 = dfy[sc1].values
            s2 = dfy[sc2].values
            if len(s1) < 5:
                continue
            r1 = _rankdata(s1)
            r2 = _rankdata(s2)
            rho = np.corrcoef(r1, r2)[0, 1]

            res_index_pair.append({
                "Year": y,
                "Score1": sc1,
                "Score2": sc2,
                "n_cities": len(s1),
                "Spearman_rank": rho
            })

    return dict(
        within_index_year_pair_spearman=pd.DataFrame(res_year_pair),
        within_year_index_pair_spearman=pd.DataFrame(res_index_pair)
    )


rank_stab = compute_city_rank_stability(
    scores,
    years_dyn,
    score_cols=["S_A", "S_D", "S_final"]  # S_R could be added if available consistently
)

rank_year_pair_df = rank_stab["within_index_year_pair_spearman"]
rank_index_pair_df = rank_stab["within_year_index_pair_spearman"]

rank_year_pair_path = os.path.join(OUT_DIR, "RankStability_WithinIndex_YearPairs.xlsx")
rank_index_pair_path = os.path.join(OUT_DIR, "RankStability_WithinYear_IndexPairs.xlsx")
rank_year_pair_df.to_excel(rank_year_pair_path, index=False)
rank_index_pair_df.to_excel(rank_index_pair_path, index=False)
print("Saved rank stability tables to:", rank_year_pair_path, "and", rank_index_pair_path)

# Simple diagnostic: mean rank stability as function of year difference (within each index)
if not rank_year_pair_df.empty:
    plt.figure(figsize=(8, 5))
    ax = plt.gca()
    for sc in rank_year_pair_df["ScoreType"].unique():
        sub = rank_year_pair_df[rank_year_pair_df["ScoreType"] == sc]
        sub = sub.assign(YearDiff=sub["Year2"] - sub["Year1"])
        grp = sub.groupby("YearDiff")["Spearman_rank"].mean()
        ax.plot(grp.index, grp.values, marker='o', label=sc)
    ax.set_xlabel("Year difference (|Year2 - Year1|)", fontweight="bold")
    ax.set_ylabel("Mean Spearman rank corr.", fontweight="bold")
    ax.set_title("Rank stability over time (within each index)", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "Fig10_RankStability_WithinIndex_YearDiff.png"), dpi=300)
    plt.close()

# Simple diagnostic: within-year rank agreement between indices (Static vs Dynamic vs Final)
if not rank_index_pair_df.empty:
    plt.figure(figsize=(8, 5))
    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    for (s1, s2), sub in rank_index_pair_df.groupby(["Score1", "Score2"]):
        label = f"{s1} vs {s2}"
        ax.plot(sub["Year"], sub["Spearman_rank"], marker='o', label=label)
    ax.set_xlabel("Year", fontweight="bold")
    ax.set_ylabel("Spearman rank corr.", fontweight="bold")
    ax.set_title("Rank agreement between indices (within year)", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "Fig10_RankStability_WithinYear_ScorePairs.png"), dpi=300)
    plt.close()

# ==========================================================
# 11) Case study: TIAV (lightweight analysis)
# ==========================================================
FEATURE_CASE = "TIAV"
TARGET_FOR_CASE = "GDP"

if FEATURE_CASE not in feature_cols:
    raise ValueError(f"Feature {FEATURE_CASE} is not present in feature_cols. Please check column names.")

print(f"=== 11) Case study for feature: {FEATURE_CASE} (lightweight) ===")

years_arr = np.array(years_dyn)
w_case_base = W[FEATURE_CASE].loc[years_dyn].values  # baseline dynamic weight trajectory

CASE_BOOTSTRAP_B = 100
np.random.seed(2027)

T = len(years_dyn)
w_case_boot = np.zeros((CASE_BOOTSTRAP_B, T))

# Sign-flip bootstrap on year-to-year increments for the case feature
if T > 1:
    d = w_case_base[1:] - w_case_base[:-1]
else:
    d = np.zeros(0)

for b in range(CASE_BOOTSTRAP_B):
    if T == 1:
        w_case_boot[b, 0] = w_case_base[0]
        continue

    signs = np.random.choice([-1.0, 1.0], size=T - 1)
    d_null = d * signs

    w_null = np.zeros(T)
    w_null[0] = w_case_base[0]
    for t_idx in range(1, T):
        w_null[t_idx] = w_null[t_idx - 1] + d_null[t_idx - 1]

    w_null = np.clip(w_null, 0.0, 1.0)
    scale = w_case_base.mean() / (w_null.mean() + 1e-12)
    w_null = w_null * scale

    w_case_boot[b, :] = w_null

# Approximate 95% pointwise confidence band for the feature-specific trajectory
w_case_ci_low = np.percentile(w_case_boot, 2.5, axis=0)
w_case_ci_high = np.percentile(w_case_boot, 97.5, axis=0)

# Cross-sectional yearly means of the feature and the target for context
feat_mean_per_year = []
target_mean_per_year = []

for t in years_dyn:
    df_t = df_all[df_all["Year"] == t].copy()
    if df_t.empty:
        feat_mean_per_year.append(np.nan)
        target_mean_per_year.append(np.nan)
        continue

    feat_mean_per_year.append(df_t[FEATURE_CASE].astype(float).mean())
    target_mean_per_year.append(df_t[TARGET_FOR_CASE].astype(float).mean())

feat_mean_per_year = np.array(feat_mean_per_year, dtype=float)
target_mean_per_year = np.array(target_mean_per_year, dtype=float)

def zscore(x):
    """Z-score normalization with NaN robustness (returns NaN if variance=0)."""
    m = np.nanmean(x)
    s = np.nanstd(x)
    if s <= 0:
        return x * np.nan
    return (x - m) / s

feat_z = zscore(feat_mean_per_year)
target_z = zscore(target_mean_per_year)

# Two-panel figure: (1) dynamic weight + CI, (2) z-scored feature and target means
plt.figure(figsize=(10, 7))

ax1 = plt.subplot(2, 1, 1)
ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
ax1.plot(years_arr, w_case_base, color="#1f77b4", marker="o",
         label=f"w_t({FEATURE_CASE}) baseline")
ax1.fill_between(years_arr, w_case_ci_low, w_case_ci_high,
                 color="#1f77b4", alpha=0.2, label="Approx. 95% CI")
ax1.set_ylabel("Weight", fontweight="bold")
ax1.set_title(f"Case study: dynamic weight of {FEATURE_CASE}", fontweight="bold")
ax1.grid(True, alpha=0.3)
ax1.legend(loc="best")

ax2 = plt.subplot(2, 1, 2, sharex=ax1)
ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
ax2.plot(years_arr, feat_z, color="#2ca02c", marker="s",
         label=f"{FEATURE_CASE} (z-score, mean over cities)")
ax2.plot(years_arr, target_z, color="#d62728", marker="^",
         label=f"{TARGET_FOR_CASE} (z-score, mean over cities)")
ax2.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
ax2.set_xlabel("Year", fontweight="bold")
ax2.set_ylabel("z-score", fontweight="bold")
ax2.set_title(f"{FEATURE_CASE} and {TARGET_FOR_CASE}: cross-sectional means over years", fontweight="bold")
ax2.grid(True, alpha=0.3)
ax2.legend(loc="best")

plt.tight_layout()
fig_name = f"CaseStudy_{FEATURE_CASE}_Weight_CI_Feature_Target.png"
plt.savefig(os.path.join(OUT_DIR, fig_name), dpi=300, bbox_inches="tight")
plt.close()

print(f"Case study figure saved to: {os.path.join(OUT_DIR, fig_name)}")
print("✅ Extreme dynamic-weight version, scores, comparison tables, and visualizations have been written to:")
print(OUT_DIR)
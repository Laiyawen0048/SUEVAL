# ==============================================
# Feature Weight Fusion (Adaptive α + Rank-based Robust Aggregation
# + Performance-weighting + Unsupervised Completion + Visualization)
# Enhanced: TOP10 tables + multiple publication-ready figures
# ==============================================

import os
import datetime
import pandas as pd
import numpy as np
from scipy.stats import rankdata
import matplotlib.pyplot as plt

# -------- Global Matplotlib style (for paper-level clarity) --------
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
plt.rcParams["axes.unicode_minus"] = False

# global font sizes
plt.rcParams["font.size"] = 16          # base font size
plt.rcParams["axes.titlesize"] = 18     # axes title
plt.rcParams["axes.labelsize"] = 16     # x/y label
plt.rcParams["xtick.labelsize"] = 14
plt.rcParams["ytick.labelsize"] = 14
plt.rcParams["legend.fontsize"] = 14

# === Step 0. 自动创建输出文件夹 ===
time_tag = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = rf"C:\Users\沐阳\Desktop\静态权重_00-23\Feature_Fusion_Results_{time_tag}"
os.makedirs(output_dir, exist_ok=True)

# === Step 1. 读取数据（监督：指定4个文件）===
sup_dir = r"C:\Users\沐阳\Desktop\模型3.0输出结果\静态_有监督"

sup_files = [
    os.path.join(sup_dir, "xgb_GDP_analysis_results.xlsx"),
    os.path.join(sup_dir, "xgb_Local_exp_analysis_results.xlsx"),
    os.path.join(sup_dir, "xgb_Post_rev_analysis_results.xlsx"),
    os.path.join(sup_dir, "xgb_Wastewater_analysis_results.xlsx"),
]

for fp in sup_files:
    if not os.path.exists(fp):
        raise FileNotFoundError(f"Supervised file not found: {fp}")

def read_supervised_feature_importance(xlsx_path: str) -> pd.DataFrame:
    """
    Read supervised feature importance sheet: sheet='feature_importance'
    Required columns: Feature, Importance
    Output: Feature, Weight (Weight = Importance)
    """
    df = pd.read_excel(xlsx_path, sheet_name="feature_importance")
    if not {"Feature", "Importance"}.issubset(df.columns):
        raise ValueError(
            f"'feature_importance' in {os.path.basename(xlsx_path)} "
            f"does not contain columns 'Feature' and 'Importance'; "
            f"actual columns: {df.columns.tolist()}"
        )
    df = df[["Feature", "Importance"]].copy()
    df["Feature"] = df["Feature"].astype(str).str.strip()
    df = df[df["Feature"].notna() & (df["Feature"] != "")]
    df["Importance"] = pd.to_numeric(df["Importance"], errors="coerce").fillna(0.0)

    df = df.rename(columns={"Importance": "Weight"})
    df = df.groupby("Feature", as_index=False)["Weight"].mean()
    return df

sup1 = read_supervised_feature_importance(sup_files[0])
sup2 = read_supervised_feature_importance(sup_files[1])
sup3 = read_supervised_feature_importance(sup_files[2])
sup4 = read_supervised_feature_importance(sup_files[3])

# === Unsupervised (assume two columns: Feature, Weight) ===
unsup = pd.read_excel(
    r"C:\Users\沐阳\Desktop\模型3.0输出结果\静态_无监督\feature_weights_autoencoder_with_val.xlsx"
)
unsup = unsup.iloc[:, :2].copy()
unsup.columns = ["Feature", "Weight"]
unsup["Feature"] = unsup["Feature"].astype(str).str.strip()
unsup = unsup[unsup["Feature"].notna() & (unsup["Feature"] != "")]
unsup["Weight"] = pd.to_numeric(unsup["Weight"], errors="coerce").fillna(0.0)
unsup = unsup.groupby("Feature", as_index=False)["Weight"].mean()

# === Step 2. 对齐特征集 & 特征补全（以无监督为全集基准）===
full_features = list(unsup["Feature"].unique())

def fill_to_full_features(df: pd.DataFrame, full_features_list) -> pd.DataFrame:
    missing = set(full_features_list) - set(df["Feature"])
    if missing:
        df = pd.concat(
            [df, pd.DataFrame({"Feature": list(missing), "Weight": [0.0] * len(missing)})],
            ignore_index=True
        )
    return df.groupby("Feature", as_index=False)["Weight"].mean()

sup1 = fill_to_full_features(sup1, full_features)
sup2 = fill_to_full_features(sup2, full_features)
sup3 = fill_to_full_features(sup3, full_features)
sup4 = fill_to_full_features(sup4, full_features)

print(f"✅ Total features in unsupervised universe: {len(full_features)}")
print(f"sup1 (GDP) after completion: {len(sup1)}")
print(f"sup2 (Local_exp) after completion: {len(sup2)}")
print(f"sup3 (Post_rev) after completion: {len(sup3)}")
print(f"sup4 (Wastewater) after completion: {len(sup4)}\n")

# === Step 3. Merge into final table ===
final = pd.DataFrame({"Feature": full_features})
final = final.merge(sup1.rename(columns={"Weight": "Weight_m1"}), on="Feature", how="left")
final = final.merge(sup2.rename(columns={"Weight": "Weight_m2"}), on="Feature", how="left")
final = final.merge(sup3.rename(columns={"Weight": "Weight_m3"}), on="Feature", how="left")
final = final.merge(sup4.rename(columns={"Weight": "Weight_m4"}), on="Feature", how="left")
final = final.merge(unsup.rename(columns={"Weight": "Weight_m5"}), on="Feature", how="left")
final = final.fillna(0)

cols = ["Weight_m1", "Weight_m2", "Weight_m3", "Weight_m4", "Weight_m5"]
model_names = ["GDP", "Local_exp", "Post_rev", "Wastewater", "Unsuper"]

# === Step 4. Performance-weighted rank fusion for supervised models ===
performance = np.array([0.9975, 0.9881, 0.9563, 0.8861])  # to be updated if needed

perf_min, perf_max = performance.min(), performance.max()
eps = 1e-8
perf_scaled = (performance - perf_min) / (perf_max - perf_min + eps)

temp = 2.0
beta = np.exp(temp * perf_scaled)
beta = beta / beta.sum()

print("Supervised model performance: ", performance)
print("Scaled performance (0-1): ", perf_scaled.round(4))
print("Softmax weights beta: ", beta.round(4), "\n")

rank_mat = np.zeros((len(final), 4))
for j, col in enumerate(cols[:4]):
    rank_mat[:, j] = rankdata(-final[col], method="average")

weighted_mean_rank = np.average(rank_mat, axis=1, weights=beta)
rank_based_weight = 1.0 / (weighted_mean_rank + 1e-8)
final["Supervised_Fused"] = rank_based_weight / rank_based_weight.sum()

# === Step 5. Adaptive α (variance-driven) ===
sup_var = np.var(final["Supervised_Fused"])
unsup_var = np.var(final["Weight_m5"])
alpha = unsup_var / (unsup_var + sup_var + 1e-8)
print(f"🤖 Adaptive α (unsupervised share): {alpha:.4f}")
print(f"📊 Variance: Supervised={sup_var:.8f} | Unsupervised={unsup_var:.8f}\n")

# === Step 6. Final fusion (sum=1) ===
final["FinalWeight"] = alpha * final["Weight_m5"] + (1 - alpha) * final["Supervised_Fused"]
final["FinalWeight"] = final["FinalWeight"] / final["FinalWeight"].sum()
final = final.sort_values("FinalWeight", ascending=False).reset_index(drop=True)

# ========== Helper: TOP10 bar plot (publication style) ==========
def plot_top10_bar(df, weight_col, title, filename, xlabel="Weight", sort_desc=True):
    tmp = df[["Feature", weight_col]].copy()
    tmp = tmp.sort_values(weight_col, ascending=not sort_desc).head(10)
    tmp = tmp.iloc[::-1]  # show from largest at top

    plt.figure(figsize=(10, 6))
    plt.barh(
        tmp["Feature"],
        tmp[weight_col],
        color="#4C78A8",
        edgecolor="black",
        linewidth=1.0,
    )
    plt.gca().tick_params(axis="y", labelsize=12)
    for label in plt.gca().get_yticklabels():
        label.set_fontweight("bold")
    for i, v in enumerate(tmp[weight_col]):
        plt.text(
            v,
            i,
            f"{v:.3f}",
            va="center",
            ha="left",
            fontsize=10,
        )
    plt.title(title, fontweight="bold")
    plt.xlabel(xlabel, fontweight="bold")
    plt.tight_layout()
    out_path = os.path.join(output_dir, filename)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved figure: {out_path}")

# ========== TOP10 visualizations ==========
# 1) Individual supervised models & unsupervised
plot_top10_bar(final, "Weight_m1", "GDP model: top-10 features", "Top10_GDP.png")
plot_top10_bar(final, "Weight_m2", "Local_exp model: top-10 features", "Top10_LocalExp.png")
plot_top10_bar(final, "Weight_m3", "Post_rev model: top-10 features", "Top10_PostRev.png")
plot_top10_bar(final, "Weight_m4", "Wastewater model: top-10 features", "Top10_Wastewater.png")
plot_top10_bar(final, "Weight_m5", "Unsupervised (AutoEncoder): top-10 features", "Top10_Unsuper.png")

# 2) Supervised_Fused TOP10
plot_top10_bar(final, "Supervised_Fused", "Supervised fused weights: top-10 features", "Top10_Supervised_Fused.png")

# 3) FinalWeight TOP10
plot_top10_bar(final, "FinalWeight", "Final fused weights: top-10 features", "Top10_FinalWeight.png")

# ========== Visualizing supervised fusion process on FinalWeight top-10 ==========
top10_feats = final.head(10)["Feature"].tolist()
top10_multi = final[final["Feature"].isin(top10_feats)].copy()
top10_multi["order"] = top10_multi["Feature"].map({f: i for i, f in enumerate(top10_feats)})
top10_multi = top10_multi.sort_values("order")

# Figure 1: each feature's weights in 4 supervised models + Supervised_Fused
plt.figure(figsize=(16, max(6, 0.6 * len(top10_feats))))
y = np.arange(len(top10_feats))
width = 0.15

for i, (col, name, color) in enumerate(
    zip(["Weight_m1", "Weight_m2", "Weight_m3", "Weight_m4"],
        ["GDP", "Local_exp", "Post_rev", "Wastewater"],
        ["#4C78A8", "#F58518", "#E45756", "#72B7B2"])
):
    plt.barh(
        y + (i - 1.5) * width,
        top10_multi[col],
        height=width,
        label=name,
        color=color,
        alpha=0.9,
        edgecolor="black",
        linewidth=0.8
    )

plt.barh(
    y + 2.5 * width,
    top10_multi["Supervised_Fused"],
    height=width,
    label="Supervised_Fused",
    color="#54A24B",
    alpha=0.9,
    edgecolor="black",
    linewidth=0.8
)

plt.yticks(y, top10_multi["Feature"],fontweight="bold")
plt.gca().invert_yaxis()
plt.xlabel("Weight", fontweight="bold")
plt.title("Top-10 : supervised models vs fused weight", fontweight="bold")
plt.legend(loc="lower right", fontsize=14)
plt.tight_layout()
out_path = os.path.join(output_dir, "Top10_Supervised_Fusion_Process.png")
plt.savefig(out_path, dpi=300, bbox_inches="tight")
plt.show()
plt.close()
print(f"Saved figure: {out_path}")

# Figure 2: same top-10 in Unsupervised vs Supervised_Fused vs FinalWeight
plt.figure(figsize=(14, max(6, 0.6 * len(top10_feats))))
y = np.arange(len(top10_feats))
width = 0.25

plt.barh(
    y - width,
    top10_multi["Weight_m5"],
    height=width,
    label="Unsupervised",
    color="#4C78A8",
    alpha=0.9,
    edgecolor="black",
    linewidth=0.8
)
plt.barh(
    y,
    top10_multi["Supervised_Fused"],
    height=width,
    label="Supervised_Fused",
    color="#F58518",
    alpha=0.9,
    edgecolor="black",
    linewidth=0.8
)
plt.barh(
    y + width,
    top10_multi["FinalWeight"],
    height=width,
    label="FinalWeight",
    color="#54A24B",
    alpha=0.9,
    edgecolor="black",
    linewidth=0.8
)

plt.yticks(y, top10_multi["Feature"],fontweight="bold")
plt.gca().invert_yaxis()
plt.xlabel("Weight", fontweight="bold")
plt.title("Top-10: unsupervised vs supervised-fused vs final weights", fontweight="bold")
plt.legend(loc="lower right", fontsize=14)
plt.tight_layout()
out_path = os.path.join(output_dir, "Top10_Unsuper_Supervised_Final_Compare.png")
plt.savefig(out_path, dpi=300, bbox_inches="tight")
plt.show()
plt.close()
print(f"Saved figure: {out_path}")

# ========== Write TOP10 table into a dedicated Excel sheet ==========
top10_final = final.head(10).copy()
top10_final = top10_final[[
    "Feature", "Weight_m1", "Weight_m2", "Weight_m3", "Weight_m4",
    "Weight_m5", "Supervised_Fused", "FinalWeight"
]]

excel_path = os.path.join(output_dir, "Feature_Weights_Fusion_AutoAlpha.xlsx")
with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
    final.to_excel(writer, sheet_name="All_Features", index=False)
    top10_final.to_excel(writer, sheet_name="Top10_FinalWeight", index=False)

# === Step 8. Meta info & report ===
meta_path = os.path.join(output_dir, "supervised_files_used.txt")
with open(meta_path, "w", encoding="utf-8") as f:
    f.write("Supervised result files used in this fusion:\n")
    for i, fp in enumerate(sup_files, 1):
        f.write(f"sup{i}: {fp}\n")

summary_path = os.path.join(output_dir, "fusion_summary.txt")
with open(summary_path, "w", encoding="utf-8") as f:
    f.write("Feature weight fusion report "
            "(adaptive α + performance-weighted rank fusion + unsupervised completion)\n")
    f.write("=" * 70 + "\n")
    f.write(f"Fusion time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

    f.write("Input files:\n")
    f.write(f"  Supervised directory: {sup_dir}\n")
    f.write("  Supervised files:\n")
    f.write(f"    1) {os.path.basename(sup_files[0])} (GDP)\n")
    f.write(f"    2) {os.path.basename(sup_files[1])} (Local_exp)\n")
    f.write(f"    3) {os.path.basename(sup_files[2])} (Post_rev)\n")
    f.write(f"    4) {os.path.basename(sup_files[3])} (Wastewater)\n")
    f.write("  Unsupervised file: feature_weights_autoencoder_with_val.xlsx\n\n")

    f.write("Feature statistics:\n")
    f.write(f"  Total features in unsupervised universe: {len(full_features)}\n\n")

    f.write(f"Adaptive α (unsupervised share) = {alpha:.8f}\n")
    f.write(f"Variance: Supervised={sup_var:.10f} | Unsupervised={unsup_var:.10f}\n\n")

    f.write("Supervised model performance weights (softmax):\n")
    for name, perf, w in zip(
        ["GDP", "Local_exp", "Post_rev", "Wastewater"],
        performance,
        beta
    ):
        f.write(f"  {name}: perf={perf:.4f}, weight={w:.4f}\n")

    f.write("\nOutputs:\n")
    f.write(f"  Fused weight table (all features + top-10 sheet): {excel_path}\n")
    f.write(f"  Supervised file list: {meta_path}\n")
    f.write(f"  Report: {summary_path}\n")
    f.write("  Figures:\n")
    f.write("    - Top10_GDP.png / Top10_LocalExp.png / Top10_PostRev.png / Top10_Wastewater.png\n")
    f.write("    - Top10_Unsuper.png\n")
    f.write("    - Top10_Supervised_Fused.png\n")
    f.write("    - Top10_FinalWeight.png\n")
    f.write("    - Top10_Supervised_Fusion_Process.png\n")
    f.write("    - Top10_Unsuper_Supervised_Final_Compare.png\n")
    f.write("=" * 70 + "\n")
    f.write(f"Total features: {final.shape[0]}\n")
    f.write(f"Sum of final weights: {final['FinalWeight'].sum():.10f}\n")

print("✅ Feature weight fusion completed.")
print(f"📂 Output directory: {output_dir}")
print(f"📊 α (unsupervised share): {alpha:.3f}")
print(f"📄 Report file: {summary_path}")
print(f"🧩 Final feature count: {final.shape[0]}")
print(f"⚖️ Sum of final weights: {final['FinalWeight'].sum():.6f}")
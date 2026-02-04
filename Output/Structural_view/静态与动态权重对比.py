# -*- coding: utf-8 -*-
"""
Compare Dynamic vs Static Feature Weights — Publication-ready figures and tables

Produces:
  1) Excel workbook with sheets:
       - Summary
       - Dynamic_Ranked
       - Static_Ranked
       - Merged_All
       - Top20_Union_Compare
  2) Tornado plot for TOP-20 union:
       - Left: static |weight|
       - Right: dynamic |weight|
       - Center: Δ = |Dynamic| - |Static| (line + points)
Notes: preserves original algorithmic logic; improves robustness, labels, and style for publication.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ================= Global plotting style (publication-oriented) =================
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

plt.rcParams["font.size"] = 13
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["axes.labelsize"] = 12
plt.rcParams["xtick.labelsize"] = 12
plt.rcParams["ytick.labelsize"] = 12
plt.rcParams["legend.fontsize"] = 10

sns.set_style("whitegrid")

# ====== Edit these file paths as needed ======
DYNAMIC_XLSX = r"C:\Users\沐阳\Desktop\模型3.0输出结果\动态权重_2023\Feature_Weights_Dynamics.xlsx"
STATIC_XLSX  = r"C:\Users\沐阳\Desktop\模型3.0输出结果\静态权重\Feature_Fusion_Results_20260130_162517\Feature_Weights_Fusion_AutoAlpha.xlsx"

OUT_DIR = r"C:\Users\沐阳\Desktop\模型3.0输出结果\静态VS动态"
os.makedirs(OUT_DIR, exist_ok=True)

OUT_EXCEL = os.path.join(OUT_DIR, "Dynamic_vs_Static_Weights_Compare.xlsx")
OUT_PNG   = os.path.join(OUT_DIR, "Top20_Dynamic_vs_Static_Tornado.png")


# ================= Utility functions =================
def load_weights_excel(xlsx_path: str) -> pd.DataFrame:
    """
    Read a weights Excel and return DataFrame with columns: Feature, Weight.

    Supports common column-name variants; falls back to first column as Feature
    and 8th column as Weight for legacy files.
    """
    df = pd.read_excel(xlsx_path)

    def norm(col):
        return str(col).strip().lower()

    feature_col = None
    for c in df.columns:
        if norm(c) in {"feature", "features", "指标", "特征", "变量", "name"}:
            feature_col = c
            break

    weight_col = None
    for c in df.columns:
        if norm(c) in {"finalweight", "final_weight", "weight", "weights", "w", "最终权重", "权重", "trueweight", "trueweights"}:
            weight_col = c
            break

    if feature_col is not None and weight_col is not None:
        out = df[[feature_col, weight_col]].copy()
        out.columns = ["Feature", "Weight"]
    else:
        # Fallback for legacy formats: first column = Feature, 8th column = FinalWeight
        if df.shape[1] < 8:
            raise ValueError(f"Unable to detect Feature/Weight columns and file has <8 columns: {xlsx_path}")
        out = df.iloc[:, [0, 7]].copy()
        out.columns = ["Feature", "Weight"]

    out["Feature"] = out["Feature"].astype(str).str.strip()
    out["Weight"] = pd.to_numeric(out["Weight"], errors="coerce").fillna(0.0)

    # Remove empty feature names and aggregate duplicates by mean
    out = out[out["Feature"].notna() & (out["Feature"] != "")]
    out = out.groupby("Feature", as_index=False)["Weight"].mean()
    return out


def rank_df(wdf: pd.DataFrame) -> pd.DataFrame:
    """
    Return a ranked DataFrame sorted by absolute weight (descending).
    Adds columns: AbsWeight, Rank (1 = largest abs weight).
    """
    w = wdf.copy()
    w["AbsWeight"] = w["Weight"].abs()
    w = w.sort_values(["AbsWeight", "Weight", "Feature"], ascending=[False, False, True]).reset_index(drop=True)
    w["Rank"] = np.arange(1, len(w) + 1)
    return w


# ================= Main workflow =================
def main():
    # 1) Load weight tables
    w_dyn = load_weights_excel(DYNAMIC_XLSX)
    w_sta = load_weights_excel(STATIC_XLSX)

    r_dyn = rank_df(w_dyn).rename(columns={
        "Weight": "Weight_dynamic",
        "AbsWeight": "AbsWeight_dynamic",
        "Rank": "Rank_dynamic",
    })

    r_sta = rank_df(w_sta).rename(columns={
        "Weight": "Weight_static",
        "AbsWeight": "AbsWeight_static",
        "Rank": "Rank_static",
    })

    # 2) Merge and fill missing values
    merged = pd.merge(r_dyn, r_sta, on="Feature", how="outer")

    n_dyn = len(r_dyn)
    n_sta = len(r_sta)

    merged["Weight_dynamic"]    = merged["Weight_dynamic"].fillna(0.0)
    merged["Weight_static"]     = merged["Weight_static"].fillna(0.0)
    merged["AbsWeight_dynamic"] = merged["AbsWeight_dynamic"].fillna(0.0)
    merged["AbsWeight_static"]  = merged["AbsWeight_static"].fillna(0.0)
    merged["Rank_dynamic"]      = merged["Rank_dynamic"].fillna(n_dyn + 1).astype(int)
    merged["Rank_static"]       = merged["Rank_static"].fillna(n_sta + 1).astype(int)

    merged["DeltaWeight"]    = merged["Weight_dynamic"] - merged["Weight_static"]
    merged["DeltaAbsWeight"] = merged["AbsWeight_dynamic"] - merged["AbsWeight_static"]
    merged["DeltaRank"]      = merged["Rank_dynamic"] - merged["Rank_static"]

    # 3) TOP20 union and flags
    top20_dyn = set(r_dyn.head(20)["Feature"])
    top20_sta = set(r_sta.head(20)["Feature"])
    overlap   = top20_dyn & top20_sta
    top_union = sorted(list(top20_dyn | top20_sta))

    top_cmp = merged[merged["Feature"].isin(top_union)].copy()
    top_cmp["InTop20_Dynamic"] = top_cmp["Feature"].isin(top20_dyn)
    top_cmp["InTop20_Static"]  = top_cmp["Feature"].isin(top20_sta)
    top_cmp["InTop20_Both"]    = top_cmp["Feature"].isin(overlap)

    # Sort for plotting: by dynamic absolute weight descending (visual priority)
    top_cmp = top_cmp.sort_values("AbsWeight_dynamic", ascending=False).reset_index(drop=True)

    # Spearman correlation on intersection ranks (if intersection non-trivial)
    common = pd.merge(r_dyn[["Feature", "Rank_dynamic"]], r_sta[["Feature", "Rank_static"]], on="Feature", how="inner")
    spearman = common["Rank_dynamic"].corr(common["Rank_static"], method="spearman") if len(common) >= 3 else np.nan

    # 4) Export Excel workbook with summary sheets
    summary = pd.DataFrame({
        "item": [
            "dynamic_file", "static_file",
            "n_features_dynamic", "n_features_static",
            "n_features_union", "n_features_intersection",
            "top20_overlap_count",
            "spearman_rank_corr_on_intersection"
        ],
        "value": [
            DYNAMIC_XLSX, STATIC_XLSX,
            len(r_dyn), len(r_sta),
            merged.shape[0], len(common),
            len(overlap),
            spearman
        ]
    })

    with pd.ExcelWriter(OUT_EXCEL, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="Summary", index=False)
        r_dyn.to_excel(writer, sheet_name="Dynamic_Ranked", index=False)
        r_sta.to_excel(writer, sheet_name="Static_Ranked", index=False)
        merged.sort_values(["AbsWeight_dynamic", "AbsWeight_static"], ascending=False).to_excel(writer, sheet_name="Merged_All", index=False)
        top_cmp.to_excel(writer, sheet_name="Top20_Union_Compare", index=False)

    # 5) Publication-quality Tornado plot for TOP-20 union
    labels  = top_cmp["Feature"].tolist()
    y       = np.arange(len(labels))
    abs_dyn = top_cmp["AbsWeight_dynamic"].to_numpy()
    abs_sta = top_cmp["AbsWeight_static"].to_numpy()
    in_both = top_cmp["InTop20_Both"].to_numpy()
    delta   = abs_dyn - abs_sta

    fig, ax = plt.subplots(figsize=(8, 0.45 * len(labels) + 2))

    # Left: static (negative direction), Right: dynamic (positive)
    bars_s = ax.barh(y, -abs_sta, color="#4C78A8", alpha=0.95, label="Static |weight|")
    bars_d = ax.barh(y,  abs_dyn, color="#F58518", alpha=0.95, label="Dynamic |weight|")

    # Delta line and points
    ax.plot(delta, y, color="black", linewidth=1.0, zorder=5, label="Δ = |Dynamic| - |Static|")
    ax.scatter(delta, y, color="black", s=18, zorder=6)

    # Determine symmetric x-limits with margin
    max_side = max(abs(np.min(-abs_sta)), np.max(abs_dyn))
    max_delta = np.max(np.abs(delta))
    x_limit = max(max_side, max_delta) * 1.25
    ax.set_xlim(-x_limit, x_limit)

    # Value labels on left bars (static)
    for rect, v in zip(bars_s, abs_sta):
        x_pos = rect.get_x() + rect.get_width()
        ax.text(x_pos - 0.002, rect.get_y() + rect.get_height() / 2, f"{v:.4f}",
                va="center", ha="right", fontsize=9, color="#4C78A8")

    # Value labels on right bars (dynamic)
    for rect, v in zip(bars_d, abs_dyn):
        x_pos = rect.get_x() + rect.get_width()
        ax.text(x_pos + 0.002, rect.get_y() + rect.get_height() / 2, f"{v:.4f}",
                va="center", ha="left", fontsize=9, color="#F58518")

    # Delta numeric labels
    for yi, d in zip(y, delta):
        ax.text(d, yi, f"{d:+.4f}", va="center", ha="left" if d >= 0 else "right",
                fontsize=8, color="black", clip_on=True)

    # Mark features present in both top-20 sets with a red dot at center
    for yi, flag in zip(y, in_both):
        if flag:
            ax.scatter(0, yi, s=30, color="red", zorder=7)

    # Y-axis labels and style
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontweight="bold")
    ax.invert_yaxis()  # highest items on top

    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("|Weight| (Static left, Dynamic right);  Δ = |Dynamic| - |Static|", fontweight="bold")
    ax.set_title(
        f"TOP20 Feature Comparison (Union)\nOverlap(top20)={len(overlap)}    Spearman(intersection ranks)={np.nan if np.isnan(spearman) else f'{spearman:.4f}'}",
        pad=10, fontweight="bold"
    )

    # Remove top and right spines for a clean layout
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(loc="lower right", frameon=True)

    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Final console summary
    print("✅ Comparison complete.")
    print(f"- Excel output: {OUT_EXCEL}")
    print(f"- Tornado plot: {OUT_PNG}")
    print(f"- TOP20 overlap count: {len(overlap)}")
    print(f"- Spearman(rank corr) on intersection: {spearman}")

if __name__ == "__main__":
    main()
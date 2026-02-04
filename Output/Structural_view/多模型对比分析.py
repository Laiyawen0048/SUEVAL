# -*- coding: utf-8 -*-
"""
Multi-model weight comparison:
Traditional (00-18 & 00-23) vs ML vs DL vs SUE-EVAL (static & dynamic-average)

阶段A：基于 2000-2018 的锚定期做多模型静态权重对比；
阶段B：基于 SUE-EVAL-2018 的 TOP3 特征，绘制
        - SUE-EVAL 权重 2018-2023 的变化；
        - EWM 权重 2018-2023（通过 00-18,00-19,...,00-23 窗口）变化；
      进行对比分析。
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---------- Matplotlib & Seaborn style ----------
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
plt.rcParams["grid.alpha"] = 0.4

sns.set_style("whitegrid")

COLOR_SUE = "#F58518"     # 橙色：SUE-EVAL
COLOR_TRAD = "#4C78A8"    # 蓝色：传统基线
PALETTE_TRAD = ["#4C78A8", "#54A24B", "#E45756", "#72B7B2", "#FF9DA6"]

# ========== 1. Paths ==========

# 1) 传统评价模型权重（00-18 & 00-19 & ... & 00-23）
TRADITIONAL_DIR = r"C:\Users\沐阳\Desktop\模型3.0输出结果\多模型对比分析\Traditional_Models"
TRADITIONAL_0018_XLSX = os.path.join(TRADITIONAL_DIR, "weights_all_methods_and_top20_figs_00-18.xlsx")
TRADITIONAL_0019_XLSX = os.path.join(TRADITIONAL_DIR, "weights_all_methods_and_top20_figs_00-19.xlsx")
TRADITIONAL_0020_XLSX = os.path.join(TRADITIONAL_DIR, "weights_all_methods_and_top20_figs_00-20.xlsx")
TRADITIONAL_0021_XLSX = os.path.join(TRADITIONAL_DIR, "weights_all_methods_and_top20_figs_00-21.xlsx")
TRADITIONAL_0022_XLSX = os.path.join(TRADITIONAL_DIR, "weights_all_methods_and_top20_figs_00-22.xlsx")
TRADITIONAL_0023_XLSX = os.path.join(TRADITIONAL_DIR, "weights_all_methods_and_top20_figs_00-23.xlsx")

TRADITIONAL_SHEET = "Weights_All"   # 如有不同请修改

# 2) 机器学习（00-18 & 00-23）——阶段A使用
ML_0018_XLSX = r"C:\Users\沐阳\Desktop\模型3.0输出结果\多模型对比分析\ML_DL_Models\ML_GDP_analysis_results_00-18.xlsx"
ML_0023_XLSX = r"C:\Users\沐阳\Desktop\模型3.0输出结果\多模型对比分析\ML_DL_Models\ML_GDP_analysis_results_00-23.xlsx"
ML_SHEET = "feature_importance"

# 3) 深度学习（00-18 & 00-23）——阶段A使用
DL_0018_XLSX = r"C:\Users\沐阳\Desktop\模型3.0输出结果\多模型对比分析\ML_DL_Models\DL_results_00-18.xlsx"
DL_0023_XLSX = r"C:\Users\沐阳\Desktop\模型3.0输出结果\多模型对比分析\ML_DL_Models\DL_results_00-23.xlsx"

# 4) SUE-EVAL 静态权重（全期静态融合，用于阶段A）
SUE_STATIC_XLSX = r"C:\Users\沐阳\Desktop\模型3.0输出结果\静态权重\Feature_Fusion_Results_20260130_162517\Feature_Weights_Fusion_AutoAlpha.xlsx"
# 列: Feature, Weight_m1 ... Supervised_Fused, FinalWeight

# 5) SUE-EVAL 年度权重（2018-2023，阶段B使用）
SUEVAL_ANNUAL_XLSX = r"C:\Users\沐阳\Desktop\模型3.0输出结果\多模型对比分析\SUEVAL\SUEVAL_weight.xlsx"
# 列: Feature, 2018_Weight, 2019_Weight, 2020_Weight, 2021_Weight, 2022_Weight, 2023_Weight

# 输出目录
OUT_DIR = r"C:\Users\沐阳\Desktop\模型3.0输出结果\多模型对比分析\Outputs_MultiModel_0018_0023"
os.makedirs(OUT_DIR, exist_ok=True)

OUT_EXCEL = os.path.join(OUT_DIR, "MultiModel_Weight_Compare_0018_0023.xlsx")


# ========== 2. Loading helpers ==========

def load_traditional_all_methods(xlsx_path: str, sheet_name: str, tag: str):
    """
    读取传统模型权重：
    列: Feature, EWM, CRITIC, STD, CV, PCA, DEA_CCR, MeanWeight
    返回:
      - wide: 每行一个 Feature，多列为各方法
      - long: (Feature, Method, Weight, PeriodTag)
    """
    df = pd.read_excel(xlsx_path, sheet_name=sheet_name)

    # Feature 列
    if "Feature" in df.columns:
        feature_col = "Feature"
    elif "Unnamed: 0" in df.columns:
        feature_col = "Unnamed: 0"
    else:
        feature_col = df.columns[0]

    methods = [
        "EWM",
        "CRITIC",
        "STD",
        "CV",
        "PCA",
        "DEA_CCR",
        "MeanWeight",
    ]
    for m in methods:
        if m not in df.columns:
            raise ValueError(f"Column '{m}' not found in traditional file: {xlsx_path}")

    wide = df[[feature_col] + methods].copy()
    wide.rename(columns={feature_col: "Feature"}, inplace=True)
    wide["Feature"] = wide["Feature"].astype(str).str.strip()
    for m in methods:
        wide[m] = pd.to_numeric(wide[m], errors="coerce").fillna(0.0)

    wide = wide[wide["Feature"].notna() & (wide["Feature"] != "")]
    wide = wide.groupby("Feature", as_index=False)[methods].mean()

    methods_simple = ["EWM", "CRITIC", "STD", "CV", "PCA", "DEA_CCR", "MeanWeight"]

    long = wide.melt(
        id_vars="Feature",
        value_vars=methods_simple,
        var_name="Method",
        value_name="Weight",
    )
    long["PeriodTag"] = tag
    return wide, long


def load_ml_importance(xlsx_path: str, sheet_name: str, tag: str) -> pd.DataFrame:
    df = pd.read_excel(xlsx_path, sheet_name=sheet_name)
    if not {"Feature", "Importance"}.issubset(df.columns):
        raise ValueError(f"ML file must contain 'Feature' and 'Importance' columns: {xlsx_path}")

    df = df[["Feature", "Importance"]].copy()
    df["Feature"] = df["Feature"].astype(str).str.strip()
    df["Weight"] = pd.to_numeric(df["Importance"], errors="coerce").fillna(0.0)
    df = df[df["Feature"].notna() & (df["Feature"] != "")]
    df = df.groupby("Feature", as_index=False)["Weight"].mean()
    df["Method"] = "ML"
    df["PeriodTag"] = tag
    return df[["Feature", "Method", "Weight", "PeriodTag"]]


def load_dl_weights(xlsx_path: str, tag: str) -> pd.DataFrame:
    df = pd.read_excel(xlsx_path)
    if "Feature" not in df.columns or "Weight" not in df.columns:
        raise ValueError(f"DL file must contain 'Feature' and 'Weight': {xlsx_path}")
    df = df[["Feature", "Weight"]].copy()
    df["Feature"] = df["Feature"].astype(str).str.strip()
    df["Weight"] = pd.to_numeric(df["Weight"], errors="coerce").fillna(0.0)
    df = df[df["Feature"].notna() & (df["Feature"] != "")]
    df = df.groupby("Feature", as_index=False)["Weight"].mean()
    df["Method"] = "DL"
    df["PeriodTag"] = tag
    return df[["Feature", "Method", "Weight", "PeriodTag"]]


def load_sue_static(xlsx_path: str) -> pd.DataFrame:
    """
    SUE-EVAL 静态融合权重，Method 统一命名为 'SUEVAL'
    （阶段A用于分布比较），严格使用 FinalWeight 这一列。
    """
    df = pd.read_excel(xlsx_path)

    # 处理 Feature 列
    if "Feature" not in df.columns:
        df.rename(columns={df.columns[0]: "Feature"}, inplace=True)

    # 强制使用 FinalWeight 列
    if "FinalWeight" not in df.columns:
        raise ValueError(
            f"'FinalWeight' column not found in SUE static file: {xlsx_path}\n"
            f"Available columns: {list(df.columns)}"
        )

    out = df[["Feature", "FinalWeight"]].copy()
    out.columns = ["Feature", "Weight"]

    out["Feature"] = out["Feature"].astype(str).str.strip()
    out["Weight"] = pd.to_numeric(out["Weight"], errors="coerce").fillna(0.0)
    out = out[out["Feature"].notna() & (out["Feature"] != "")]
    out = out.groupby("Feature", as_index=False)["Weight"].mean()
    out["Method"] = "SUEVAL"
    out["PeriodTag"] = "Static_All"
    return out


def load_sue_eval_annual(xlsx_path: str) -> pd.DataFrame:
    """
    读取 SUE-EVAL 年度权重文件：
    Feature, 2018_Weight, 2019_Weight, ..., 2023_Weight
    """
    df = pd.read_excel(xlsx_path)
    if "Feature" not in df.columns:
        df.rename(columns={df.columns[0]: "Feature"}, inplace=True)
    df["Feature"] = df["Feature"].astype(str).str.strip()

    expected_cols = ["2018_Weight", "2019_Weight", "2020_Weight",
                     "2021_Weight", "2022_Weight", "2023_Weight"]
    for c in expected_cols:
        if c not in df.columns:
            raise ValueError(f"Column '{c}' not found in SUE-EVAL annual file: {xlsx_path}")
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    return df[["Feature"] + expected_cols]


def rank_by_abs_weight(df: pd.DataFrame, group_cols=("Method",)) -> pd.DataFrame:
    df = df.copy()
    df["AbsWeight"] = df["Weight"].abs()
    sort_cols = list(group_cols) + ["AbsWeight", "Weight", "Feature"]
    df = df.sort_values(sort_cols, ascending=[True] * len(group_cols) + [False, False, True])
    df["Rank"] = df.groupby(list(group_cols)).cumcount() + 1
    return df


# ========== 3. Main analysis ==========

def main():
    tag_0018 = "00-18"
    tag_0023 = "00-23"

    # 1) 传统 00-18
    wide_trad_0018, long_trad_0018 = load_traditional_all_methods(
        TRADITIONAL_0018_XLSX, TRADITIONAL_SHEET, tag_0018
    )

    # 2) ML & DL 00-18
    ml_0018 = load_ml_importance(ML_0018_XLSX, ML_SHEET, tag_0018)
    dl_0018 = load_dl_weights(DL_0018_XLSX, tag_0018)

    # 3) SUE-EVAL 静态权重（全期，用于A）
    sue_static = load_sue_static(SUE_STATIC_XLSX)

    # 4) SUE-EVAL 年度权重（2018-2023，用于B）
    sue_annual = load_sue_eval_annual(SUEVAL_ANNUAL_XLSX)

    # ========= A 部分：锚定期静态权重对比 =========
    long_anchor = pd.concat(
        [long_trad_0018, ml_0018, dl_0018, sue_static],
        ignore_index=True
    )
    long_anchor_ranked = rank_by_abs_weight(long_anchor, group_cols=("Method",))

    # 保存阶段A结果
    with pd.ExcelWriter(OUT_EXCEL, engine="openpyxl") as writer:
        wide_trad_0018.to_excel(writer, sheet_name="Trad_00-18_Wide", index=False)
        long_trad_0018.to_excel(writer, sheet_name="Trad_00-18_Long", index=False)
        ml_0018.to_excel(writer, sheet_name="ML_00-18_Long", index=False)
        dl_0018.to_excel(writer, sheet_name="DL_00-18_Long", index=False)
        sue_static.to_excel(writer, sheet_name="SUE_Static_Long", index=False)
        sue_annual.to_excel(writer, sheet_name="SUE_Annual_2018_2023", index=False)
        long_anchor_ranked.to_excel(writer, sheet_name="Anchor_AllMethods_Ranked", index=False)

    # ---------- A.1 权重分布：各方法 vs SUE-EVAL ----------
    base_method = "SUEVAL"
    all_methods = sorted(long_anchor_ranked["Method"].unique())

    trad_methods = [
        m for m in all_methods
        if m not in [base_method, "MeanWeight"]
    ]

    max_pairs = 8
    comp_methods = trad_methods[:max_pairs]

    fig, axes = plt.subplots(3, 3, figsize=(18, 10))
    axes = axes.flatten()

    sue_data = long_anchor_ranked[long_anchor_ranked["Method"] == base_method]

    # 子图0：SUEVAL 静态
    ax0 = axes[0]
    sns.histplot(
        sue_data["Weight"],
        bins=40,
        kde=True,
        color=COLOR_SUE,
        alpha=0.5,
        ax=ax0,
    )
    ax0.set_title(f"{base_method}", fontweight="bold")
    ax0.set_xlabel("Weight", fontweight="bold")
    ax0.set_ylabel("Density", fontweight="bold")
    ax0.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax0.grid(True, linestyle="--", alpha=0.4)

    # 其他方法 vs SUE-EVAL
    for i, method in enumerate(comp_methods, start=1):
        ax = axes[i]
        df_m = long_anchor_ranked[long_anchor_ranked["Method"] == method]

        sns.histplot(
            df_m["Weight"],
            bins=40,
            kde=False,
            color=PALETTE_TRAD[(i - 1) % len(PALETTE_TRAD)],
            alpha=0.45,
            label=method,
            ax=ax,
        )
        sns.histplot(
            sue_data["Weight"],
            bins=40,
            kde=False,
            color=COLOR_SUE,
            alpha=0.35,
            label=base_method,
            ax=ax,
        )
        sns.kdeplot(
            df_m["Weight"],
            color=PALETTE_TRAD[(i - 1) % len(PALETTE_TRAD)],
            linewidth=1.5,
            ax=ax,
        )
        sns.kdeplot(
            sue_data["Weight"],
            color=COLOR_SUE,
            linewidth=1.5,
            linestyle="--",
            ax=ax,
        )

        ax.set_title(f"{method} vs {base_method}", fontweight="bold")
        ax.set_xlabel("Weight", fontweight="bold")
        ax.set_ylabel("Density", fontweight="bold")
        ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(loc="upper right", frameon=True, framealpha=0.9)

    # 删除多余子图
    for j in range(1 + len(comp_methods), len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "A_Global_Distribution_vs_SUE_Static.png"), dpi=300)
    plt.show()
    plt.close()

    # ---------- A.2 |weight| 箱线图 ----------
    plt.figure(figsize=(10, 6))
    sns.boxplot(
        data=long_anchor_ranked,
        x="Method",
        y="AbsWeight",
        showfliers=False,
        palette="Set2",
    )
    plt.ylabel(r"Weight", fontweight="bold",fontsize=14)
    plt.xlabel("Method", fontweight="bold",fontsize=14)
    plt.title("Distribution of weight across methods (Anchor 00-18)", fontweight="bold")
    plt.grid(True, axis="y", linestyle="--", alpha=0.4)
    xticklabels = plt.gca().get_xticklabels()
    plt.gca().set_xticklabels([t.get_text() for t in xticklabels],fontsize=12,fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "A_AbsWeight_Boxplot_AnchorMethods.png"), dpi=300)
    plt.show()
    plt.close()

    # ---------- A.3 TOP-K 集合 Jaccard ----------
    top_k = 10
    methods = sorted(long_anchor_ranked["Method"].unique())
    top_sets = {}

    for m in methods:
        feats = (
            long_anchor_ranked.loc[long_anchor_ranked["Method"] == m]
            .sort_values("AbsWeight", ascending=False)
            .head(top_k)["Feature"]
        )
        top_sets[m] = set(feats)

    jaccard = pd.DataFrame(index=methods, columns=methods, dtype=float)
    overlap_count = pd.DataFrame(index=methods, columns=methods, dtype=float)

    for mi in methods:
        for mj in methods:
            A = top_sets[mi]
            B = top_sets[mj]
            inter = len(A & B)
            union = len(A | B) if len(A | B) > 0 else 1
            jaccard.loc[mi, mj] = inter / union
            overlap_count.loc[mi, mj] = inter

    with pd.ExcelWriter(OUT_EXCEL, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
        jaccard.to_excel(writer, sheet_name="A_TopK_Jaccard")
        overlap_count.to_excel(writer, sheet_name="A_TopK_OverlapCount")

    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(
        jaccard,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        cbar_kws={"label": "Jaccard similarity"},
        linewidths=0.5,
        linecolor="white",
    )
    ax.set_title(f"Jaccard similarity of TOP{top_k} feature sets (Anchor 00-18)", fontweight="bold")
    ax.set_xlabel("Method", fontweight="bold")
    ax.set_ylabel("Method", fontweight="bold")

    # 加粗 x/y 轴方法标签，并可选调整字体大小/旋转
    ax.set_xticklabels([t.get_text() for t in ax.get_xticklabels()], fontweight="bold", fontsize=12, rotation=45)
    ax.set_yticklabels([t.get_text() for t in ax.get_yticklabels()], fontweight="bold", fontsize=12, rotation=0)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"A_Top{top_k}_Jaccard_Heatmap.png"), dpi=300)
    plt.show()
    plt.close()

    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(
        overlap_count,
        annot=True,
        fmt=".0f",
        cmap="Oranges",
        cbar_kws={"label": f"Overlap size (TOP{top_k})"},
        linewidths=0.5,
        linecolor="white",
    )
    ax.set_title(f"Overlap size of TOP{top_k} feature sets (Anchor 00-18)", fontweight="bold")
    ax.set_xlabel("Method", fontweight="bold")
    ax.set_ylabel("Method", fontweight="bold")

    # 加粗 x/y 轴方法标签
    ax.set_xticklabels([t.get_text() for t in ax.get_xticklabels()], fontweight="bold", fontsize=12, rotation=45)
    ax.set_yticklabels([t.get_text() for t in ax.get_yticklabels()], fontweight="bold", fontsize=12, rotation=0)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"A_Top{top_k}_OverlapCount_Heatmap.png"), dpi=300)
    plt.show()
    plt.close()

    # ---------- A.4 EWM vs SUEVAL Tornado ----------
    df_ewm = long_anchor_ranked[long_anchor_ranked["Method"] == "EWM"].copy()
    df_sue_static = long_anchor_ranked[long_anchor_ranked["Method"] == "SUEVAL"].copy()

    df_ewm = df_ewm[["Feature", "Weight", "AbsWeight", "Rank"]].rename(
        columns={
            "Weight": "Weight_EWM",
            "AbsWeight": "AbsWeight_EWM",
            "Rank": "Rank_EWM",
        }
    )
    df_sue_static = df_sue_static[["Feature", "Weight", "AbsWeight", "Rank"]].rename(
        columns={
            "Weight": "Weight_SUEVAL",
            "AbsWeight": "AbsWeight_SUEVAL",
            "Rank": "Rank_SUEVAL",
        }
    )

    merged = pd.merge(df_ewm, df_sue_static, on="Feature", how="outer")

    n_trad = len(df_ewm)
    n_sue = len(df_sue_static)

    merged["Weight_EWM"] = merged["Weight_EWM"].fillna(0.0)
    merged["Weight_SUEVAL"] = merged["Weight_SUEVAL"].fillna(0.0)
    merged["AbsWeight_EWM"] = merged["AbsWeight_EWM"].fillna(0.0)
    merged["AbsWeight_SUEVAL"] = merged["AbsWeight_SUEVAL"].fillna(0.0)
    merged["Rank_EWM"] = merged["Rank_EWM"].fillna(n_trad + 1).astype(int)
    merged["Rank_SUEVAL"] = merged["Rank_SUEVAL"].fillna(n_sue + 1).astype(int)

    top_trad = set(merged.sort_values("AbsWeight_EWM", ascending=False).head(top_k)["Feature"])
    top_sue = set(merged.sort_values("AbsWeight_SUEVAL", ascending=False).head(top_k)["Feature"])
    overlap_ts = top_trad & top_sue
    top_union = sorted(list(top_trad | top_sue))

    top_cmp = merged[merged["Feature"].isin(top_union)].copy()
    top_cmp["InTop_EWM"] = top_cmp["Feature"].isin(top_trad)
    top_cmp["InTop_SUEVAL"] = top_cmp["Feature"].isin(top_sue)
    top_cmp["InTop_Both"] = top_cmp["Feature"].isin(overlap_ts)

    top_cmp = top_cmp.sort_values("AbsWeight_SUEVAL", ascending=False).reset_index(drop=True)

    common = pd.merge(
        df_ewm[["Feature", "Rank_EWM"]],
        df_sue_static[["Feature", "Rank_SUEVAL"]],
        on="Feature",
        how="inner",
    )
    spearman = (
        common["Rank_EWM"].corr(common["Rank_SUEVAL"], method="spearman")
        if len(common) >= 3
        else np.nan
    )

    with pd.ExcelWriter(OUT_EXCEL, engine="openpyxl", mode="a", if_sheet_exists="overlay") as writer:
        top_cmp.to_excel(writer, sheet_name="A_TopK_EWM_vs_SUEStatic", index=False)

    # ====== 绘图（按给定范式 + 调整标签）======
    labels = top_cmp["Feature"].tolist()
    y = np.arange(len(labels))
    abs_ewm = top_cmp["AbsWeight_EWM"].to_numpy()
    abs_sue = top_cmp["AbsWeight_SUEVAL"].to_numpy()
    in_both = top_cmp["InTop_Both"].to_numpy()
    delta = abs_sue - abs_ewm  # SUEVAL - EWM

    fig, ax = plt.subplots(figsize=(8, 0.45 * len(labels) + 2))

    # 左右条形：EWM 左，SUEVAL 右
    bars_ewm = ax.barh(
        y, -abs_ewm,
        color=COLOR_TRAD, alpha=0.9,
        label="EWM |weight| (left)"
    )
    bars_sue = ax.barh(
        y, abs_sue,
        color=COLOR_SUE, alpha=0.9,
        label="SUEVAL |weight| (right)"
    )

    # 中间 Δ 线和点（黑色）
    ax.plot(delta, y, color="black", linewidth=1.0, zorder=5,
            label=r"$\Delta = |w_{\mathrm{SUEVAL}}| - |w_{\mathrm{EWM}}|$")
    ax.scatter(delta, y, color="black", s=18, zorder=6)

    # 控制坐标范围，留出文字空间
    max_val = max(abs(np.min(-abs_ewm)), np.max(abs_sue)) if len(labels) > 0 else 1.0
    max_delta = np.max(np.abs(delta)) if len(labels) > 0 else 1.0
    x_limit = max(max_val, max_delta) * 1.3
    ax.set_xlim(-x_limit, x_limit)

    offset_bar = 0.01 * x_limit

    # 左侧数值标签（EWM）
    for rect, v in zip(bars_ewm, abs_ewm):
        x_pos = rect.get_x() + rect.get_width()
        ax.text(
            x_pos - offset_bar,
            rect.get_y() + rect.get_height() / 2,
            f"{v:.3f}",
            va="center", ha="right",
            fontsize=8, color=COLOR_TRAD
        )

    # 右侧数值标签（SUEVAL）
    for rect, v in zip(bars_sue, abs_sue):
        x_pos = rect.get_x() + rect.get_width()
        ax.text(
            x_pos + offset_bar,
            rect.get_y() + rect.get_height() / 2,
            f"{v:.3f}",
            va="center", ha="left",
            fontsize=8, color=COLOR_SUE
        )

    # 不再绘制 Δ 数值标签

    # 交集特征在 x=0 标红点
    for yi, flag in zip(y, in_both):
        if flag:
            ax.scatter(0, yi, s=20, color="red", zorder=7)

    # Y 轴标签：全部黑色加粗
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontweight="bold", fontsize=11, color="black")
    ax.invert_yaxis()

    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel(
        r"|Weight| (EWM left, SUEVAL right);   "
        r"$\Delta = |w_{\mathrm{SUEVAL}}| - |w_{\mathrm{EWM}}|$",
        fontweight="bold"
    )
    ax.set_title(
        f"EWM vs SUEVAL (Union of TOP{top_k})\n"
        f"Overlap(TOP{top_k})={len(overlap_ts)}   "
        f"Spearman(intersection ranks)={spearman:.4f}",
        pad=10,
        fontweight="bold"
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="lower right", frameon=True)

    fig.tight_layout()
    fig.savefig(
        os.path.join(OUT_DIR, f"A_Top{top_k}_EWM_vs_SUE_Static_Tornado.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()
    plt.close()
    # ====================================================
    # B. 基于 SUE-EVAL 2018_Weight TOP3 特征，绘制 SUE-EVAL & EWM 权重随时间变化
    # ====================================================

    # 1) 从 SUEVAL 年度权重中按 2018_Weight 选 TOP3 特征
    sue_annual_top = sue_annual.copy()
    sue_annual_top["Abs2018"] = sue_annual_top["2018_Weight"].abs()
    sue_annual_top = sue_annual_top.sort_values("Abs2018", ascending=False)
    top3_features = sue_annual_top.head(2)["Feature"].tolist()

    print("Top3 features by SUEVAL 2018_Weight:", top3_features)

    # 2) 构造 SUE-EVAL 时间序列（2018-2023）
    sue_long = sue_annual[sue_annual["Feature"].isin(top3_features)].melt(
        id_vars="Feature",
        value_vars=[
            "2018_Weight",
            "2019_Weight",
            "2020_Weight",
            "2021_Weight",
            "2022_Weight",
            "2023_Weight",
        ],
        var_name="YearCol",
        value_name="Weight_SUE",
    )
    sue_long["Year"] = sue_long["YearCol"].str.extract(r"(\d{4})").astype(int)
    sue_long = sue_long[["Feature", "Year", "Weight_SUE"]]

    # 3) 构造 EWM 时间序列（2018-2023），基于 00-18, 00-19, ..., 00-23 文件
    ewm_files = [
        (TRADITIONAL_0018_XLSX, 2018),
        (TRADITIONAL_0019_XLSX, 2019),
        (TRADITIONAL_0020_XLSX, 2020),
        (TRADITIONAL_0021_XLSX, 2021),
        (TRADITIONAL_0022_XLSX, 2022),
        (TRADITIONAL_0023_XLSX, 2023),
    ]

    records_ewm = []
    for fpath, year in ewm_files:
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"EWM file not found: {fpath}")
        wide_tmp, long_tmp = load_traditional_all_methods(fpath, TRADITIONAL_SHEET, tag=str(year))
        ewm_tmp = long_tmp[long_tmp["Method"] == "EWM"][["Feature", "Weight"]].copy()
        ewm_tmp = ewm_tmp[ewm_tmp["Feature"].isin(top3_features)]
        ewm_tmp["Year"] = year
        ewm_tmp.rename(columns={"Weight": "Weight_EWM"}, inplace=True)
        records_ewm.append(ewm_tmp)

    ewm_ts = pd.concat(records_ewm, ignore_index=True)

    # 4) 合并 SUE-EVAL 与 EWM 的时间序列（按 Feature, Year）
    ts_merged = pd.merge(
        sue_long,      # Feature, Year, Weight_SUE
        ewm_ts,        # Feature, Year, Weight_EWM
        on=["Feature", "Year"],
        how="outer",
    )
    ts_merged["Weight_SUE"] = ts_merged["Weight_SUE"].fillna(0.0)
    ts_merged["Weight_EWM"] = ts_merged["Weight_EWM"].fillna(0.0)

    # 保存阶段B数据
    with pd.ExcelWriter(OUT_EXCEL, engine="openpyxl", mode="a", if_sheet_exists="overlay") as writer:
        sue_long.to_excel(writer, sheet_name="B_SUE_Top3_TimeSeries", index=False)
        ewm_ts.to_excel(writer, sheet_name="B_EWM_Top3_TimeSeries", index=False)
        ts_merged.to_excel(writer, sheet_name="B_SUE_EWM_Top3_Merged", index=False)

    # 5) 画多维折线图：每个特征一个子图，横轴 Year，纵轴 Weight，SUE vs EWM 对比
    n_feat = len(top3_features)
    fig, axes = plt.subplots(n_feat, 1, figsize=(10, 4 * n_feat), sharex=True)

    if n_feat == 1:
        axes = [axes]

    for i, feat in enumerate(top3_features):
        ax = axes[i]
        df_f = ts_merged[ts_merged["Feature"] == feat].sort_values("Year")

        # SUE-EVAL 折线（2018-2023）
        ax.plot(
            df_f["Year"],
            df_f["Weight_SUE"],
            marker="s",
            color=COLOR_SUE,
            linewidth=2,
            label="SUEVAL",
        )

        # EWM 折线（2018-2023）
        ax.plot(
            df_f["Year"],
            df_f["Weight_EWM"],
            marker="o",
            color=COLOR_TRAD,
            linewidth=2,
            label="EWM",
        )

        ax.set_title(f"Weight evolution for feature: {feat}", fontweight="bold")
        ax.set_ylabel("Weight", fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.legend(loc="best", frameon=True, framealpha=0.9)

    axes[-1].set_xlabel("Year", fontweight="bold")

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "B_SUE_vs_EWM_Top3_TimeSeries.png"), dpi=300)
    plt.show()
    plt.close()

    print("✅ Multi-model weight comparison finished.")
    print(f"- Excel summary: {OUT_EXCEL}")
    print(f"- Output dir: {OUT_DIR}")


if __name__ == "__main__":
    main()
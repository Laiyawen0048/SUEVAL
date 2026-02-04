# -*- coding: utf-8 -*-
"""
Multi-model WEIGHT evolution analysis (2018–2023)

Models to compare:
- Traditional evaluation models: EWM, CRITIC, STD, CV, PCA, DEA_CCR, MeanWeight
- ML
- DL
- SUEVAL (dynamic yearly weights)

Analyses:
1) Weight change rate based on Top-K features (K=20) anchored at 2018.
   For each Model separately:
   - Take TopK features by |Weight| in 2018 as anchor set T_2018
   - For each year y in 2018–2023:
       ChangeRate(y) = 1 - |TopK_y ∩ T_2018| / K
   ChangeRate=0 -> TopK 完全不变，结构不变；
   ChangeRate=1 -> TopK 完全无交集，结构完全改变。

2) Feature-level trajectories for a specific feature (e.g., 'TIAV') over 2018–2023.
   - For each Model, extract yearly weight of TIAV and plot.

NO city score computations, only weights.
"""

import os
from typing import List

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
plt.rcParams["grid.alpha"] = 0.7

sns.set_style("whitegrid")

# ======================================================
# 1. Paths & constants
# ======================================================

# 基础路径（按你提供的）
BASE_TRAD_PATH = r"C:\Users\沐阳\Desktop\模型3.0输出结果\多模型对比分析\Traditional_Models"
BASE_ML_PATH   = r"C:\Users\沐阳\Desktop\模型3.0输出结果\多模型对比分析\ML_DL_Models"
BASE_DL_PATH   = r"C:\Users\沐阳\Desktop\模型3.0输出结果\多模型对比分析\ML_DL_Models"

# 文件名模板（注意是两位年份后缀：00-18, 00-19, ...）
TRAD_FILE_TEMPLATE = "weights_all_methods_and_top20_figs_00-{end_year}.xlsx"
TRAD_SHEET_NAME    = "Weights_All"

ML_FILE_TEMPLATE = "ML_GDP_analysis_results_00-{end_year}.xlsx"
ML_SHEET_NAME    = "feature_importance"

DL_FILE_TEMPLATE = "DL_results_00-{end_year}.xlsx"

# SUEVAL 动态权重
SUEVAL_XLSX = r"C:\Users\沐阳\Desktop\模型3.0输出结果\多模型对比分析\SUEVAL\SUEVAL_weight.xlsx"
# Columns: Feature, 2018_Weight, 2019_Weight, 2020_Weight, 2021_Weight, 2022_Weight, 2023_Weight

# 输出目录
OUT_DIR = r"C:\Users\沐阳\Desktop\模型3.0输出结果\多模型对比分析\Outputs_MultiModel_Weights"
os.makedirs(OUT_DIR, exist_ok=True)

OUT_EXCEL = os.path.join(OUT_DIR, "MultiModel_Weight_Analysis_2018_2023.xlsx")

YEARS = list(range(2018, 2024))  # 2018-2023
TOP_K = 30
BASE_YEAR = 2018
ANCHOR_FEATURE = "TIAV"


def year_to_suffix(year: int) -> str:
    """
    Convert 4-digit year to 2-digit suffix for file names:
    2018 -> '18', 2019 -> '19', ...
    """
    return str(year)[-2:]


# ========= 全局统一的模型样式（颜色 + marker），两个图共用 =========

MODEL_STYLE = {
    "EWM":       {"color": "#1f77b4", "marker": "o"},
    "CRITIC":    {"color": "#c0c0c0", "marker": "s"},
    "STD":       {"color": "#2ca02c", "marker": "D"},
    "CV":        {"color": "#d62728", "marker": "^"},
    "PCA":       {"color": "#9467bd", "marker": "v"},
    "DEA_CCR":   {"color": "#8c564b", "marker": "P"},
    "MeanWeight":{"color": "#17becf", "marker": "X"},
    "ML":        {"color": "#54A24B", "marker": "h"},
    "DL":        {"color": "#E45756", "marker": "d"},
    "SUEVAL":    {"color": "#F58518", "marker": "*"},
}


# ======================================================
# 2. Loading functions: unified long table = Feature, Year, Model, Weight
# ======================================================

def load_traditional_year(end_year: int) -> pd.DataFrame:
    """
    Load traditional model weights for year end_year (00-end_year),
    and reshape to long: Feature, Year, Model, Weight
    Models: EWM, CRITIC, STD, CV, PCA, DEA_CCR, MeanWeight
    """
    suffix = year_to_suffix(end_year)  # 2018 -> '18'
    fname = TRAD_FILE_TEMPLATE.format(end_year=suffix)
    fpath = os.path.join(BASE_TRAD_PATH, fname)
    if not os.path.exists(fpath):
        raise FileNotFoundError(f"Traditional weight file not found: {fpath}")

    df = pd.read_excel(fpath, sheet_name=TRAD_SHEET_NAME)

    if "Feature" in df.columns:
        feature_col = "Feature"
    elif "Unnamed: 0" in df.columns:
        feature_col = "Unnamed: 0"
    else:
        feature_col = df.columns[0]

    methods = ["EWM", "CRITIC", "STD", "CV", "PCA", "DEA_CCR", "MeanWeight"]
    for m in methods:
        if m not in df.columns:
            raise ValueError(f"Column '{m}' not found in file: {fpath}")

    wide = df[[feature_col] + methods].copy()
    wide.rename(columns={feature_col: "Feature"}, inplace=True)
    wide["Feature"] = wide["Feature"].astype(str).str.strip()
    wide = wide[wide["Feature"].notna() & (wide["Feature"] != "")]

    for m in methods:
        wide[m] = pd.to_numeric(wide[m], errors="coerce").fillna(0.0)

    long = wide.melt(
        id_vars="Feature",
        value_vars=methods,
        var_name="Model",
        value_name="Weight"
    )
    long["Year"] = end_year   # 用 2018,2019... 作为真实年份
    return long[["Feature", "Year", "Model", "Weight"]]


def load_ml_year(end_year: int) -> pd.DataFrame:
    """
    Load ML importance for year end_year (00-end_year),
    and reshape to: Feature, Year, Model='ML', Weight
    """
    suffix = year_to_suffix(end_year)
    fname = ML_FILE_TEMPLATE.format(end_year=suffix)
    fpath = os.path.join(BASE_ML_PATH, fname)
    if not os.path.exists(fpath):
        raise FileNotFoundError(f"ML weight file not found: {fpath}")

    df = pd.read_excel(fpath, sheet_name=ML_SHEET_NAME)
    if "Feature" not in df.columns or "Importance" not in df.columns:
        raise ValueError(f"ML file must contain 'Feature' and 'Importance': {fpath}")

    out = df[["Feature", "Importance"]].copy()
    out["Feature"] = out["Feature"].astype(str).str.strip()
    out = out[out["Feature"].notna() & (out["Feature"] != "")]
    out["Weight"] = pd.to_numeric(out["Importance"], errors="coerce").fillna(0.0)
    out["Year"] = end_year
    out["Model"] = "ML"
    return out[["Feature", "Year", "Model", "Weight"]]


def load_dl_year(end_year: int) -> pd.DataFrame:
    """
    Load DL weights for year end_year (00-end_year),
    and reshape to: Feature, Year, Model='DL', Weight
    """
    suffix = year_to_suffix(end_year)
    fname = DL_FILE_TEMPLATE.format(end_year=suffix)
    fpath = os.path.join(BASE_DL_PATH, fname)
    if not os.path.exists(fpath):
        raise FileNotFoundError(f"DL weight file not found: {fpath}")

    df = pd.read_excel(fpath)
    if "Feature" not in df.columns or "Weight" not in df.columns:
        raise ValueError(f"DL file must contain 'Feature' and 'Weight': {fpath}")

    out = df[["Feature", "Weight"]].copy()
    out["Feature"] = out["Feature"].astype(str).str.strip()
    out = out[out["Feature"].notna() & (out["Feature"] != "")]
    out["Weight"] = pd.to_numeric(out["Weight"], errors="coerce").fillna(0.0)
    out["Year"] = end_year
    out["Model"] = "DL"
    return out[["Feature", "Year", "Model", "Weight"]]


def load_sueval_long(xlsx_path: str) -> pd.DataFrame:
    """
    Load SUEVAL dynamic weights (columns: Feature, 2018_Weight, ..., 2023_Weight),
    reshape to: Feature, Year, Model='SUEVAL', Weight
    """
    if not os.path.exists(xlsx_path):
        raise FileNotFoundError(f"SUEVAL weight file not found: {xlsx_path}")

    df = pd.read_excel(xlsx_path)
    if "Feature" not in df.columns:
        df.rename(columns={df.columns[0]: "Feature"}, inplace=True)

    weight_cols = [c for c in df.columns if str(c).endswith("_Weight")]
    if not weight_cols:
        raise ValueError("No *_Weight columns found in SUEVAL file.")

    out_list = []
    for c in weight_cols:
        year_str = str(c).split("_")[0]  # '2018_Weight' -> '2018'
        year = int(year_str)
        if year not in YEARS:
            continue
        tmp = df[["Feature", c]].copy()
        tmp["Feature"] = tmp["Feature"].astype(str).str.strip()
        tmp = tmp[tmp["Feature"].notna() & (tmp["Feature"] != "")]
        tmp["Weight"] = pd.to_numeric(tmp[c], errors="coerce").fillna(0.0)
        tmp["Year"] = year
        tmp["Model"] = "SUEVAL"
        out_list.append(tmp[["Feature", "Year", "Model", "Weight"]])

    out = pd.concat(out_list, ignore_index=True)
    return out


# ======================================================
# 3. Change rate & TIAV trajectories
# ======================================================

def compute_change_rate_for_model(
    df_model: pd.DataFrame,
    base_year: int,
    top_k: int
) -> pd.DataFrame:
    """
    df_model: Feature, Year, Model, Weight (only one Model)
    - Take TopK features in base_year as T_base (by |Weight|)
    - For each year y, compute ChangeRate(y) = 1 - |T_y ∩ T_base| / top_k
    Return: Year, Model, ChangeRate, OverlapCount
    """
    model_name = df_model["Model"].iloc[0]

    base = df_model[df_model["Year"] == base_year].copy()
    if base.empty:
        # if this model has no data in base_year, skip
        return pd.DataFrame(columns=["Year", "Model", "ChangeRate", "OverlapCount"])

    base["AbsW"] = base["Weight"].abs()
    base = base.sort_values("AbsW", ascending=False)
    top_base = base.head(top_k)["Feature"].tolist()
    set_base = set(top_base)

    records = []
    for y in sorted(df_model["Year"].unique()):
        df_y = df_model[df_model["Year"] == y].copy()
        df_y["AbsW"] = df_y["Weight"].abs()
        df_y = df_y.sort_values("AbsW", ascending=False)
        top_y = df_y.head(top_k)["Feature"].tolist()
        set_y = set(top_y)
        overlap = len(set_base & set_y)
        cr = 1.0 - overlap / float(top_k)
        records.append({
            "Year": y,
            "Model": model_name,
            "ChangeRate": cr,
            "OverlapCount": overlap
        })

    return pd.DataFrame(records)


def plot_change_rate(df_cr: pd.DataFrame, out_png: str, top_k: int, base_year: int):
    """
    第一个图：绘制所有模型的 TopK 特征集合变化率曲线。
    要求：
    - 同一模型使用与第二个图相同的颜色和 marker（由 MODEL_STYLE 决定）
    - Y 轴固定在 [0, 0.6] 范围
    """
    if df_cr.empty:
        return

    plt.figure(figsize=(10, 6))

    for model_name, sub in df_cr.groupby("Model"):
        sub = sub.sort_values("Year")
        style = MODEL_STYLE.get(model_name, {"color": None, "marker": "o"})
        plt.plot(
            sub["Year"],
            sub["ChangeRate"],
            marker=style["marker"],
            label=model_name,
            color=style["color"],
            linewidth=1.8
        )

    plt.axhline(0, color="black", linewidth=0.8)
    # 按你要求，把 Y 轴限制在 [0, 0.6]
    plt.ylim(0.0, 0.6)

    plt.xlabel("Year", fontweight="bold")
    plt.ylabel(f"Change rate of Top-{top_k} feature set vs {base_year}", fontweight="bold")
    plt.title(f"Top-{top_k} feature set change rate (anchor={base_year})", fontweight="bold")
    plt.grid(alpha=0.3, linestyle="--")
    plt.legend(ncol=2, fontsize=10)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.show()
    plt.close()


def extract_tiav_trajectories(df_all: pd.DataFrame, feature_name: str) -> pd.DataFrame:
    """
    Extract trajectories for feature_name (e.g. TIAV) across all Models.
    Return: Year, Model, Feature, Weight
    """
    sub = df_all[df_all["Feature"] == feature_name].copy()
    if sub.empty:
        return pd.DataFrame(columns=["Year", "Model", "Feature", "Weight"])
    sub = sub.sort_values(["Model", "Year"])
    return sub[["Year", "Model", "Feature", "Weight"]]


def plot_tiav_trajectories(df_ft: pd.DataFrame, feature_name: str, out_png: str):
    """
    第二个图：绘制 TIAV 在所有模型下的权重轨迹。
    要求：
    - 颜色和 marker 与第一个图完全一致（使用 MODEL_STYLE）
    """
    if df_ft.empty:
        return

    plt.figure(figsize=(10, 6))

    for model_name, sub in df_ft.groupby("Model"):
        sub = sub.sort_values("Year")
        style = MODEL_STYLE.get(model_name, {"color": None, "marker": "o"})
        plt.plot(
            sub["Year"],
            sub["Weight"],
            marker=style["marker"],
            label=model_name,
            color=style["color"],
            linewidth=1.8
        )

    plt.axhline(0, color="black", linewidth=0.8)
    plt.xlabel("Year", fontweight="bold")
    plt.ylabel(f"Weight of '{feature_name}'", fontweight="bold")
    plt.title(f"Feature '{feature_name}' weight trajectories (2018–2023)", fontweight="bold")
    plt.grid(alpha=0.3, linestyle="--")
    plt.legend(ncol=2, fontsize=10)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.show()
    plt.close()


# ======================================================
# 4. Main
# ======================================================

def main():
    # A. 构造统一的年度长表：Feature, Year, Model, Weight
    all_list: List[pd.DataFrame] = []

    # 传统评价模型：EWM, CRITIC, STD, CV, PCA, DEA_CCR, MeanWeight
    for y in YEARS:
        trad_y = load_traditional_year(end_year=y)
        all_list.append(trad_y)

    # ML
    for y in YEARS:
        ml_y = load_ml_year(end_year=y)
        all_list.append(ml_y)

    # DL
    for y in YEARS:
        dl_y = load_dl_year(end_year=y)
        all_list.append(dl_y)

    # SUEVAL
    sueval_long = load_sueval_long(SUEVAL_XLSX)
    all_list.append(sueval_long)

    df_all = pd.concat(all_list, ignore_index=True)

    # B. TopK 变化率：对每个 Model 单独计算
    cr_list: List[pd.DataFrame] = []
    for model_name, sub in df_all.groupby("Model"):
        cr_model = compute_change_rate_for_model(
            df_model=sub,
            base_year=BASE_YEAR,
            top_k=TOP_K
        )
        if not cr_model.empty:
            cr_list.append(cr_model)

    df_change_rate = pd.concat(cr_list, ignore_index=True) if cr_list else pd.DataFrame()

    out_png_cr = os.path.join(
        OUT_DIR, f"Top{TOP_K}_ChangeRate_Base{BASE_YEAR}.png"
    )
    plot_change_rate(
        df_cr=df_change_rate,
        out_png=out_png_cr,
        top_k=TOP_K,
        base_year=BASE_YEAR
    )

    # C. TIAV 轨迹
    df_tiav = extract_tiav_trajectories(
        df_all=df_all,
        feature_name=ANCHOR_FEATURE
    )

    out_png_tiav = os.path.join(
        OUT_DIR, f"Feature_{ANCHOR_FEATURE}_Weight_Trajectories.png"
    )
    plot_tiav_trajectories(
        df_ft=df_tiav,
        feature_name=ANCHOR_FEATURE,
        out_png=out_png_tiav
    )

    # D. 写 Excel
    with pd.ExcelWriter(OUT_EXCEL, engine="openpyxl") as writer:
        df_all.to_excel(writer, sheet_name="AllWeights_2018_2023", index=False)
        if not df_change_rate.empty:
            df_change_rate.to_excel(writer, sheet_name=f"Top{TOP_K}_ChangeRate", index=False)
        if not df_tiav.empty:
            df_tiav.to_excel(writer, sheet_name=f"Feature_{ANCHOR_FEATURE}_Trajectories", index=False)

    print("✅ Multi-model WEIGHT evolution analysis finished.")
    print(f"- Excel summary: {OUT_EXCEL}")
    print(f"- Change-rate figure: {out_png_cr}")
    print(f"- Feature '{ANCHOR_FEATURE}' trajectory figure: {out_png_tiav}")
    print(f"- Output dir: {OUT_DIR}")


if __name__ == "__main__":
    main()
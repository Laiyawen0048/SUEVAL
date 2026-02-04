# -*- coding: utf-8 -*-
"""
成都市多层级得分对比分析
基于：Scores_MultiLevel_Dynamic.xlsx
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ML_XLSX = r"C:\Users\沐阳\Desktop\模型3.0输出结果\动态权重_20260130_174425\Scores_MultiLevel_Dynamic.xlsx"

OUT_DIR = r"C:\Users\沐阳\Desktop\单样本得分对比_成都市"
os.makedirs(OUT_DIR, exist_ok=True)

OUT_PNG_CHD_TOTAL   = os.path.join(OUT_DIR, "Chengdu_TotalScores_TS.png")
OUT_PNG_CHD_D_TS    = os.path.join(OUT_DIR, "Chengdu_DimensionScores_TS.png")
OUT_PNG_CHD_FEATBAR = os.path.join(OUT_DIR, "Chengdu_FeatureScore_Structure.png")
OUT_PNG_CHD_VS_ALL  = os.path.join(OUT_DIR, "Chengdu_vs_All_YearlyAvg.png")
OUT_EXCEL_CHD_SUM   = os.path.join(OUT_DIR, "Chengdu_Score_Summary.xlsx")

CITY_NAME = "成都市"
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Times New Roman", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 14
plt.rcParams["axes.titlesize"] = 16
plt.rcParams["axes.labelsize"] = 14
plt.rcParams["xtick.labelsize"] = 12
plt.rcParams["ytick.labelsize"] = 12
plt.rcParams["legend.fontsize"] = 12
sns.set_style("whitegrid")


def main():
    # 读取四个 sheet
    df_feat = pd.read_excel(ML_XLSX, sheet_name="1_CityYear_FeatureScores")
    df_dim  = pd.read_excel(ML_XLSX, sheet_name="2_CityYear_DScores")

    # 标准化列名
    def norm(s): return str(s).strip().lower()
    col_map_feat = {norm(c): c for c in df_feat.columns}
    col_map_dim  = {norm(c): c for c in df_dim.columns}

    def find_col(col_map, cands, name):
        for c in cands:
            if norm(c) in col_map:
                return col_map[norm(c)]
        raise ValueError(f"在表中未找到{name}列: 备选={cands}")

    year_col_f = find_col(col_map_feat, ["year", "年份", "年度"], "Year")
    city_col_f = find_col(col_map_feat, ["city", "城市"], "City")
    totalA_f   = find_col(col_map_feat, ["total_a"], "Total_A")
    totalD_f   = find_col(col_map_feat, ["total_d"], "Total_D")
    totalF_f   = find_col(col_map_feat, ["total_f"], "Total_F")

    year_col_d = find_col(col_map_dim, ["year", "年份", "年度"], "Year")
    city_col_d = find_col(col_map_dim, ["city", "城市"], "City")
    totalA_d   = find_col(col_map_dim, ["total_a"], "Total_A")
    totalD_d   = find_col(col_map_dim, ["total_d"], "Total_D")
    totalF_d   = find_col(col_map_dim, ["total_f"], "Total_F")

    # 1) 成都市整体总分时间序列（用 feature-scores 表）
    chd_feat = df_feat[df_feat[city_col_f] == CITY_NAME].copy()
    if chd_feat.empty:
        raise ValueError(f"在 1_CityYear_FeatureScores 中未找到城市：{CITY_NAME}")
    chd_feat = chd_feat.sort_values(year_col_f)

    years = chd_feat[year_col_f].values
    x = np.arange(len(years))  # 0,1,2,... 作为横轴

    plt.figure(figsize=(7, 4.5))
    plt.plot(x, chd_feat[totalA_f], marker="o", color="#4C78A8", label="Total_A (static)")
    plt.plot(x, chd_feat[totalD_f], marker="s", color="#F58518", label="Total_D (dynamic)")
    plt.plot(x, chd_feat[totalF_f], marker="D", color="#54A24B", label="Total_F (final)")

    plt.xlabel("Year", fontweight="bold")
    plt.ylabel("Score", fontweight="bold")
    plt.title(f"ChengDu - City-level scores over time", fontweight="bold")
    plt.grid(alpha=0.3, linestyle="--")
    plt.legend()

    # 关键：用年份作为刻度标签
    plt.xticks(x, years)

    plt.tight_layout()
    plt.savefig(OUT_PNG_CHD_TOTAL, dpi=300)
    plt.show()
    plt.close()

    # 2）：成都市年度增长变化幅度 =========
    # 计算同比增长率：(本年 - 上一年) / 上一年
    chd_feat["Growth_Total_A"] = chd_feat[totalA_f].pct_change()
    chd_feat["Growth_Total_D"] = chd_feat[totalD_f].pct_change()
    chd_feat["Growth_Total_F"] = chd_feat[totalF_f].pct_change()

    # 从第二个年份开始绘制（第一个年份没有上一年）
    years_growth = years[1:]
    x_growth = np.arange(len(years_growth))

    plt.figure(figsize=(7, 4.5))
    plt.plot(x_growth, chd_feat["Growth_Total_A"].iloc[1:], marker="o",
             color="#4C78A8", label="Total_A growth")
    plt.plot(x_growth, chd_feat["Growth_Total_D"].iloc[1:], marker="s",
             color="#F58518", label="Total_D growth")
    plt.plot(x_growth, chd_feat["Growth_Total_F"].iloc[1:], marker="D",
             color="#54A24B", label="Total_F growth")

    plt.axhline(0, color="black", linewidth=0.8)
    plt.xticks(x_growth, years_growth)
    plt.xlabel("Year", fontweight="bold")
    plt.ylabel("Year-on-year growth rate", fontweight="bold")
    plt.title("ChengDu - year-on-year growth of total scores", fontweight="bold")
    plt.grid(alpha=0.3, linestyle="--")
    plt.legend()
    plt.tight_layout()

    OUT_PNG_CHD_GROWTH = os.path.join(OUT_DIR, "Chengdu_TotalScores_Growth_TS.png")
    plt.savefig(OUT_PNG_CHD_GROWTH, dpi=300)
    plt.show()
    plt.close()
    # 3) 成都市 D 维度时间序列（用 D-scores 表）
    chd_dim = df_dim[df_dim[city_col_d] == CITY_NAME].copy()
    if chd_dim.empty:
        raise ValueError(f"在 2_CityYear_DScores 中未找到城市：{CITY_NAME}")
    chd_dim = chd_dim.sort_values(year_col_d)

    # 识别 D 维度列：形如 D1_A, D1_D, D1_F...
    d_cols_A = [c for c in chd_dim.columns if c.endswith("_A") and c.startswith("D")]
    d_names  = sorted({c.split("_")[0] for c in d_cols_A},
                      key=lambda x: int(x[1:]))  # D1, D2,...

    n_dim = len(d_names)
    n_row = int(np.ceil(n_dim / 3))
    n_col = 3

    plt.figure(figsize=(4*n_col, 3.2*n_row))
    for i, d in enumerate(d_names, start=1):
        colA = f"{d}_A"
        colD = f"{d}_D"
        colF = f"{d}_F"
        if colA not in chd_dim.columns or colD not in chd_dim.columns or colF not in chd_dim.columns:
            continue

        plt.subplot(n_row, n_col, i)
        plt.plot(chd_dim[year_col_d], chd_dim[colA], marker="o", color="#4C78A8", label="A")
        plt.plot(chd_dim[year_col_d], chd_dim[colD], marker="s", color="#F58518", label="D")
        plt.plot(chd_dim[year_col_d], chd_dim[colF], marker="D", color="#54A24B", label="F")
        plt.title(d)
        plt.xlabel("Year", fontweight="bold")
        plt.grid(alpha=0.3, linestyle="--")
        if i == 1:
            plt.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(OUT_PNG_CHD_D_TS, dpi=300)
    plt.show()
    plt.close()

    # 4) 成都市：特征结构对比（跨年平均）
    #    计算成都在每个特征上的平均 ScoreA / ScoreD / ScoreF，排序后画条形图
    feat_colsA = [c for c in df_feat.columns if c.startswith("ScoreA_")]
    feat_colsD = [c for c in df_feat.columns if c.startswith("ScoreD_")]
    feat_colsF = [c for c in df_feat.columns if c.startswith("ScoreF_")]

    # 提取成都市子集
    chd_feat_sub = df_feat[df_feat[city_col_f] == CITY_NAME].copy()

    meanA = chd_feat_sub[feat_colsA].mean(axis=0)
    meanD = chd_feat_sub[feat_colsD].mean(axis=0)
    meanF = chd_feat_sub[feat_colsF].mean(axis=0)

    # 还原特征名
    feat_names = [c.replace("ScoreA_", "") for c in feat_colsA]

    feat_df = pd.DataFrame({
        "Feature": feat_names,
        "MeanA": meanA.values,
        "MeanD": meanD.values,
        "MeanF": meanF.values
    })

    # 可以按最终得分排序，截取前 N 个展示
    N_SHOW = 20
    feat_df = feat_df.sort_values("MeanF", ascending=False).head(N_SHOW)

    x = np.arange(len(feat_df))
    width = 0.25

    plt.figure(figsize=(9, 0.45*len(feat_df) + 2))
    plt.barh(x - width, feat_df["MeanA"], height=width,
             label="Static (A)", color="#4C78A8", alpha=0.85)
    plt.barh(x,          feat_df["MeanD"], height=width,
             label="Dynamic (D)", color="#F58518", alpha=0.85)
    plt.barh(x + width, feat_df["MeanF"], height=width,
             label="Final (F)", color="#54A24B", alpha=0.85)

    plt.yticks(x, feat_df["Feature"], fontweight="bold")
    plt.gca().invert_yaxis()
    plt.xlabel("Average feature score (Chengdu, across years)", fontweight="bold")
    plt.title(f"ChengDu - Top {N_SHOW} features: A vs D vs F", fontweight="bold")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_PNG_CHD_FEATBAR, dpi=300)
    plt.show()
    plt.close()

    # 5) 成都 vs 全样本：年度平均总分对比（利用 feature-scores 表）
    grp_all = df_feat.groupby(year_col_f, as_index=False)[[totalA_f, totalD_f, totalF_f]].mean()
    grp_all = grp_all.sort_values(year_col_f)

    # 为全国和成都构造统一的年份序列和索引
    years_all = grp_all[year_col_f].values
    years_chd = chd_feat[year_col_f].values
    years_union = np.union1d(years_all, years_chd)  # 所有涉及到的年份
    x = np.arange(len(years_union))

    # 将年份映射到 x 索引
    year_to_idx = {y: i for i, y in enumerate(years_union)}
    x_all = [year_to_idx[y] for y in years_all]
    x_chd = [year_to_idx[y] for y in years_chd]

    plt.figure(figsize=(7, 4.5))
    # 全国平均
    plt.plot(x_all, grp_all[totalF_f],
             marker="o", color="#9ecae1", label="All cities - mean Total_F")
    # 成都
    plt.plot(x_chd, chd_feat[totalF_f],
             marker="D", color="#08519c", label=f"ChengDu - Total_F")

    plt.xticks(x, years_union)  # 用年份作为刻度标签
    plt.xlabel("Year", fontweight="bold")
    plt.ylabel("Final score", fontweight="bold")
    plt.title(f"ChengDu vs national mean", fontweight="bold")
    plt.grid(alpha=0.3, linestyle="--")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_PNG_CHD_VS_ALL, dpi=300)
    plt.show()
    plt.close()

    # 导出一个成都相关的小汇总表（方便论文引用具体数值）
    out_summary = {
        "Year": years,
        "Chd_Total_A": chd_feat[totalA_f].values,
        "Chd_Total_D": chd_feat[totalD_f].values,
        "Chd_Total_F": chd_feat[totalF_f].values,
    }
    out_df = pd.DataFrame(out_summary)
    out_df.to_excel(OUT_EXCEL_CHD_SUM, index=False)

    print("✅ 成都市多层级得分可视化完成")
    print(f"- 图1 成都总分TS: {OUT_PNG_CHD_TOTAL}")
    print(f"- 图2 成都D维度TS: {OUT_PNG_CHD_D_TS}")
    print(f"- 图3 成都特征结构: {OUT_PNG_CHD_FEATBAR}")
    print(f"- 图4 成都 vs 全国: {OUT_PNG_CHD_VS_ALL}")
    print(f"- 汇总表: {OUT_EXCEL_CHD_SUM}")


if __name__ == "__main__":
    main()
# -*- coding: utf-8 -*-
"""
全样本逐年平均得分对比（静态 S_A / 动态 S_D / 最终 S_final）
基于：Scores_dynamic_years.xlsx
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 路径设置
SCORES_XLSX = r"C:\Users\沐阳\Desktop\模型3.0输出结果\动态权重_20260130_174425\Scores_dynamic_years.xlsx"

OUT_DIR = r"C:\Users\沐阳\Desktop\得分对比_全样本"
os.makedirs(OUT_DIR, exist_ok=True)

OUT_EXCEL_YEAR = os.path.join(OUT_DIR, "Yearly_Avg_Scores_Summary.xlsx")
OUT_PNG_TS     = os.path.join(OUT_DIR, "Yearly_Avg_Scores_TS.png")
OUT_PNG_DELTA  = os.path.join(OUT_DIR, "Yearly_Avg_Delta_vs_SA.png")

# 画图风格
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
    df = pd.read_excel(SCORES_XLSX)

    # 标准化列名查找
    def norm(s): return str(s).strip().lower()
    col_map = {norm(c): c for c in df.columns}

    def find_col(cands, name):
        for c in cands:
            if norm(c) in col_map:
                return col_map[norm(c)]
        raise ValueError(f"在文件中未找到{name}列: 备选={cands}")

    year_col = find_col(["year", "年份", "年度"], "Year")
    sa_col   = find_col(["s_a", "sa", "static_score"], "S_A")
    sd_col   = find_col(["s_d", "sd", "dynamic_score"], "S_D")
    sf_col   = find_col(["s_final", "sfinal", "final_score", "sueval"], "S_final")
    delta_col = find_col(["delta"], "Delta")
    eta_col   = find_col(["eta"], "eta")

    # ========= 逐年全样本平均 =========
    grp = df.groupby(year_col, as_index=False)[[sa_col, sd_col, sf_col, delta_col, eta_col]].mean()
    grp = grp.sort_values(year_col)

    # 计算年度平均差值
    grp["Delta_SD_SA"] = grp[sd_col] - grp[sa_col]
    grp["Delta_SF_SA"] = grp[sf_col] - grp[sa_col]

    # 年份序列 & 类别索引
    years = grp[year_col].values  # 比如 [2019, 2020, 2021, 2022, 2023]
    x = np.arange(len(years))  # [0,1,2,3,4]

    # ========= 图1：年度平均 S_A / S_D / S_final =========
    plt.figure(figsize=(7, 4.5))
    plt.plot(x, grp[sa_col], marker="o", color="#4C78A8", label="Static (S_A)")
    plt.plot(x, grp[sd_col], marker="s", color="#F58518", label="Dynamic (S_D)")
    plt.plot(x, grp[sf_col], marker="D", color="#54A24B", label="Final (S_final)")

    plt.xlabel("Year", fontweight="bold")
    plt.ylabel("Average score (across all cities)", fontweight="bold")
    plt.title("Yearly average scores: static vs dynamic vs final", fontweight="bold")
    plt.grid(alpha=0.3, linestyle="--")
    plt.legend()

    # 关键：把 x=0,1,2,... 映射为年份标签
    plt.xticks(x, years)

    plt.tight_layout()
    plt.savefig(OUT_PNG_TS, dpi=300)
    plt.show()
    plt.close()

    # ========= 图2：年度增长变化幅度 =========
    # 计算同比增长率：(本年 - 上一年) / 上一年
    # 注意：第一个年份没有“上一年”，这里用 NaN 处理，并在画图时从第二年开始画
    grp["Growth_SA"] = grp[sa_col].pct_change()
    grp["Growth_SD"] = grp[sd_col].pct_change()
    grp["Growth_SF"] = grp[sf_col].pct_change()

    # 从第二个年份开始（index 1）
    years_growth = years[1:]
    x_growth = np.arange(len(years_growth))

    plt.figure(figsize=(7, 4.5))
    plt.plot(x_growth, grp["Growth_SA"].iloc[1:], marker="o", color="#4C78A8", label="Static (S_A)")
    plt.plot(x_growth, grp["Growth_SD"].iloc[1:], marker="s", color="#F58518", label="Dynamic (S_D)")
    plt.plot(x_growth, grp["Growth_SF"].iloc[1:], marker="D", color="#54A24B", label="Final (S_final)")

    plt.axhline(0, color="black", linewidth=0.8)
    plt.xticks(x_growth, years_growth)
    plt.xlabel("Year", fontweight="bold")
    plt.ylabel("Year-on-year growth rate", fontweight="bold")
    plt.title("Yearly growth of average scores: static vs dynamic vs final", fontweight="bold")
    plt.grid(alpha=0.3, linestyle="--")
    plt.legend()
    plt.tight_layout()

    OUT_PNG_GROWTH = os.path.join(OUT_DIR, "Yearly_Avg_Scores_Growth.png")
    plt.savefig(OUT_PNG_GROWTH, dpi=300)
    plt.show()
    plt.close()

    # ========= 图3：年度平均差值 Δ =========
    width = 0.35

    plt.figure(figsize=(7, 4.5))
    plt.bar(x - width / 2, grp["Delta_SD_SA"], width,
            label="Mean(S_D - S_A)", color="#F58518", alpha=0.9)
    plt.bar(x + width / 2, grp["Delta_SF_SA"], width,
            label="Mean(S_final - S_A)", color="#54A24B", alpha=0.9)

    plt.axhline(0, color="black", linewidth=0.8)
    plt.xticks(x, years)  # 同样用年份做刻度标签
    plt.xlabel("Year", fontweight="bold")
    plt.ylabel("Mean difference vs S_A", fontweight="bold")
    plt.title("Yearly average gaps: static vs dynamic vs final", fontweight="bold")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_PNG_DELTA, dpi=300)
    plt.show()
    plt.close()
    print("✅ 全样本逐年平均得分对比完成")
    print(f"- 汇总表: {OUT_EXCEL_YEAR}")
    print(f"- 图1: {OUT_PNG_TS}")
    print(f"- 图2: {OUT_PNG_DELTA}")
    print(f"- 图3（增长率）: {OUT_PNG_GROWTH}")


if __name__ == "__main__":
    main()
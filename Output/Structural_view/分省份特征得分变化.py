import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------- 用户可修改的路径/参数 -----------------
# excel_path = r"C:\Users\沐阳\Desktop\城市得分结果\全国水平A-D-F.xlsx"
# sheet_name = "四川省最终"
excel_path = r"C:\Users\沐阳\Desktop\城市得分结果\chengdu.xlsx"
sheet_name = "Sheet3"

top_k = 10
years_to_plot = [2010, 2015, 2020, 2023]
# -------------------------------------------------------

# 样式
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Times New Roman", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
sns.set_style("whitegrid")

# 读取数据
df = pd.read_excel(excel_path, sheet_name=sheet_name)

# 基本检查
if df.shape[1] < 2:
    raise ValueError("The input sheet must have at least two columns (Year + at least one score column).")

# 检查 Year 列
if "Year" not in df.columns:
    raise ValueError("No 'Year' column found in the sheet. Please ensure a 'Year' column exists.")

# 去除 Year 列所在的空值行（若有）并确保 Year 为整数
df = df[df["Year"].notna()].copy()
df["Year"] = df["Year"].astype(int)

# 识别得分列：优先选择第2列到最后一列（基于0起始索引）
# 注意：这是基于列顺序而非列名的选择
score_cols = df.columns[1:].tolist()

# 兜底：如果第2列到最后一列不是数值或为空，则回退到原先的匹配规则
if not score_cols:
    # 兜底回退到原先匹配规则
    score_cols = [c for c in df.columns if c.startswith("Score") or "ScoreA_" in c]

# 若第2列到最后一列包含非数值列（例如字符串），则尽量只保留数值列
numeric_score_cols = [c for c in score_cols if pd.api.types.is_numeric_dtype(df[c])]
if numeric_score_cols:
    score_cols = numeric_score_cols
else:
    # 如果没有数值列，尝试通过原始匹配规则寻找可能的得分列（按需转换）
    fallback = [c for c in df.columns if c.startswith("Score") or "ScoreA_" in c]
    if fallback:
        # 尝试强制转换为数值
        for c in fallback:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        numeric_fallback = [c for c in fallback if pd.api.types.is_numeric_dtype(df[c])]
        if numeric_fallback:
            score_cols = numeric_fallback
        else:
            raise ValueError("No numeric score columns found. Expected numeric columns from 2nd to last, or columns starting with 'Score' or containing 'ScoreA_'.")
    else:
        raise ValueError("No score columns found. Expected columns from 2nd to last, or columns starting with 'Score' or containing 'ScoreA_'.")

# 打印所使用的得分列以便确认
print("Using score columns:", score_cols)

# 为每年提取 top_k 特征
rows = []
years = sorted(df["Year"].unique())
for yr in years:
    sub = df[df["Year"] == yr]
    if sub.empty:
        continue
    # 如果每年有多行（例如多城市），先对得分列取均值
    if len(sub) > 1:
        vals = sub[score_cols].mean(axis=0)
    else:
        vals = sub[score_cols].iloc[0]
    # 排序并取前 top_k
    vals_sorted = vals.sort_values(ascending=False)
    top = vals_sorted.head(top_k)
    for rank, (feat, score) in enumerate(top.items(), start=1):
        rows.append({
            "Year": yr,
            "Rank": rank,
            "Feature": feat,
            "Score": score
        })

top_df = pd.DataFrame(rows, columns=["Year", "Rank", "Feature", "Score"])

# 保存到桌面
desktop = os.path.join(os.path.expanduser("~"), "Desktop")
output_excel = os.path.join(desktop, "Top10_Features_Per_Year.xlsx")
top_df.to_excel(output_excel, index=False)
print(f"Top-10 per year table saved to: {output_excel}")

# 绘图：为指定年份绘制前10特征柱状图（降序）
plots_dir = os.path.join(desktop, "Top10_Feature_Plots")
os.makedirs(plots_dir, exist_ok=True)

for yr in years_to_plot:
    sub_top = top_df[top_df["Year"] == yr].copy()
    if sub_top.empty:
        print(f"Warning: year {yr} not found in data; skipping plot.")
        continue
    # Ensure sorted descending by Score
    sub_top = sub_top.sort_values(by="Score", ascending=False).head(top_k)
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Score", y="Feature", data=sub_top, palette="viridis", edgecolor='k')
    plt.title(f"Top {top_k} Features in {yr}", fontsize=14, fontweight='bold')
    plt.xlabel("Score")
    plt.ylabel("Feature")
    plt.tight_layout()
    plot_path = os.path.join(plots_dir, f"Top{top_k}_Features_{yr}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot for {yr} to: {plot_path}")

print("All done.")
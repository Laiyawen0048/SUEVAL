import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# 0. 画图基础设置
# =========================
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Times New Roman", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# global font sizes
plt.rcParams["font.size"] = 16          # base font size
plt.rcParams["axes.titlesize"] = 18     # axes title
plt.rcParams["axes.labelsize"] = 16     # x/y label
plt.rcParams["xtick.labelsize"] = 14
plt.rcParams["ytick.labelsize"] = 14
plt.rcParams["legend.fontsize"] = 14

# =========================
# 1. 读取各时期无监督权重
# =========================

root = r"C:\Users\沐阳\Desktop\模型3.0输出结果\滚动静态数据源"
file_name = "feature_weights_autoencoder_with_val.xlsx"

# 各时期子目录
period_dirs = {
    "basic": "静态_无监督_00-18",
    "00-19": "静态_无监督_00-19",
    "00-20": "静态_无监督_00-20",
    "00-21": "静态_无监督_00-21",
    "00-22": "静态_无监督_00-22",
    "00-23": "静态_无监督_00-23",
}

dfs = {}
for label, subdir in period_dirs.items():
    path = os.path.join(root, subdir, file_name)
    print(f"读取: {path}")
    df = pd.read_excel(path)
    df = df[['Feature', 'Weight']].copy()
    df['Feature'] = df['Feature'].astype(str)
    df = df.set_index('Feature')
    dfs[label] = df

# 统一特征集合
all_features = sorted(set().union(*[df.index for df in dfs.values()]))

# 拼成宽表：行 = 特征，列 = 时期
weight_wide = pd.DataFrame(index=all_features)
for label, df in dfs.items():
    weight_wide[label] = df.reindex(all_features)['Weight']

# 缺失权重视为 0
weight_wide = weight_wide.fillna(0.0)

print("\n前几行权重矩阵：")
print(weight_wide.head())

# =========================
# 2. 构造排名 & 相关矩阵
# =========================

# 按列（时期）对特征做排名，数值越小排名越靠前
rank_wide = weight_wide.rank(axis=0, method='average', ascending=False)

# Spearman 相关（对 rank 做 Pearson）
corr_spearman = rank_wide.corr(method='pearson')
print("\nSpearman 相关矩阵：")
print(corr_spearman.round(3))

# 如需 Kendall 也可以算一下（可选）
corr_kendall = rank_wide.corr(method='kendall')
print("\nKendall 相关矩阵：")
print(corr_kendall.round(3))

# =========================
# 3. 热力图展示相关矩阵
# =========================

plt.figure(figsize=(8, 6))
sns.heatmap(
    corr_spearman,
    annot=True, fmt=".2f",
    cmap="YlGnBu",
    vmin=0.0, vmax=1.0,
    square=True
)
plt.title("Unsupervised Feature Ranking of Spearman")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 6))
sns.heatmap(
    corr_kendall,
    annot=True, fmt=".2f",
    cmap="YlOrRd",
    vmin=0.0, vmax=1.0,
    square=True
)
plt.title("Unsupervised Feature Ranking of Kendall")
plt.tight_layout()
plt.show()

# =========================
# 4. 与 basic / 上一期 的相关性随时间变化
# =========================

labels = list(period_dirs.keys())          # ["basic", "00-19", ..., "00-23"]

# 排除 basic，只保留滚动窗口
rolling_labels = [l for l in labels if l != "basic"]

# 按结束年份排序（00-19, 00-20, ...）
def sort_key(label):
    if label == "basic":
        return -1
    return int(label.split('-')[1])

rolling_labels = sorted(rolling_labels, key=sort_key)

# 4.1 与 basic 的相关性
corr_with_basic = {}
for label in rolling_labels:
    x = rank_wide['basic']
    y = rank_wide[label]
    corr_with_basic[label] = x.corr(y, method='pearson')

# 4.2 与上一期的相关性
corr_with_prev = {}
for i in range(1, len(rolling_labels)):
    cur_label = rolling_labels[i]
    prev_label = rolling_labels[i - 1]
    x = rank_wide[prev_label]
    y = rank_wide[cur_label]
    corr_with_prev[cur_label] = x.corr(y, method='pearson')

# 4.3 整理成 DataFrame（这里是关键修正部分）
df_corr = pd.DataFrame({'period': rolling_labels})

# 与 basic 的相关性（长度与 rolling_labels 一致）
df_corr['corr_with_basic'] = df_corr['period'].map(corr_with_basic)

# 与上一期的相关性：第一期没有上一期，置为 NaN，其他用字典映射
df_corr['corr_with_prev'] = np.nan
for i, p in enumerate(rolling_labels):
    if i == 0:
        continue
    df_corr.loc[df_corr['period'] == p, 'corr_with_prev'] = corr_with_prev.get(p, np.nan)

print("\n与 basic / 上一期 的排名相关性：")
print(df_corr)

# 4.4 画折线图
plt.figure(figsize=(9, 5))
plt.plot(df_corr['period'], df_corr['corr_with_basic'],
         marker='o', label='Vs basic Spearman')
plt.plot(df_corr['period'], df_corr['corr_with_prev'],
         marker='s', label='Vs T-1 Spearman')

plt.ylim(0.0, 1.05)
plt.xlabel("Unsupervised training window")
plt.ylabel("Feature Ranking Correlation Coefficient")
plt.title("Ranking stability of Unsupervised Feature:basic VS T-1")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# =========================
# 5. （可选）数值差异的 L1 距离
# =========================

l1_with_basic = {}
for label in rolling_labels:
    x = weight_wide['basic']
    y = weight_wide[label]
    l1_with_basic[label] = np.abs(x - y).sum()

l1_with_prev = {}
for i in range(1, len(rolling_labels)):
    cur_label = rolling_labels[i]
    prev_label = rolling_labels[i - 1]
    x = weight_wide[prev_label]
    y = weight_wide[cur_label]
    l1_with_prev[cur_label] = np.abs(x - y).sum()

df_l1 = pd.DataFrame({'period': rolling_labels})
df_l1['l1_with_basic'] = df_l1['period'].map(l1_with_basic)
df_l1['l1_with_prev'] = np.nan
for i, p in enumerate(rolling_labels):
    if i == 0:
        continue
    df_l1.loc[df_l1['period'] == p, 'l1_with_prev'] = l1_with_prev.get(p, np.nan)

print("\n权重 L1 距离：")
print(df_l1)

plt.figure(figsize=(9, 5))
plt.plot(df_l1['period'], df_l1['l1_with_basic'],
         marker='o', label='VS basic of L1')
plt.plot(df_l1['period'], df_l1['l1_with_prev'],
         marker='s', label='VS T-1 L1')

plt.xlabel("Unsupervised training window")
plt.ylabel("L1 Distance）")
plt.title("Unsupervised weight difference:basic VS T-1")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# =========================
# 6. basic 的 Top20 特征在各时期的排名变化
# =========================

# 6.1 取 basic 下的 Top20 特征（按权重从大到小）
basic_weights = weight_wide['basic'].copy()
top20_features = basic_weights.sort_values(ascending=False).head(10).index.tolist()
print("\nbasic Top20 特征：")
print(top20_features)

# 6.2 抽取这些特征在各时期的排名（注意 rank_wide 越小越靠前）
top20_ranks = rank_wide.loc[top20_features, labels]  # labels = ["basic","00-19",...,"00-23"]

# 为了画图直观，可以把“名次轴”反过来：rank_high = max_rank + 1 - rank
max_rank = rank_wide.max().max()
top20_ranks_plot = max_rank + 1 - top20_ranks  # 数值越大代表越靠前

# 6.3 画线图：每条线一条特征，横轴是时期，纵轴是“名次（越高越靠前）”
plt.figure(figsize=(10, 6))

for feat in top20_features:
    plt.plot(labels, top20_ranks_plot.loc[feat, labels],
             marker='o', linewidth=1.5, alpha=0.8, label=feat)

plt.xlabel("Unsupervised training window",fontweight="bold")
plt.ylabel("Relative Rank (Value =  max_rank + 1 - rank) ",fontweight="bold")
plt.title("Top10 Feature",fontweight="bold")

# 如果特征太多，图例会挤，可以放到外面 / 或者只标注前几条
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.show()
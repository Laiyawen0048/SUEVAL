import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# ================= 字体与样式设置 =================
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Times New Roman", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

plt.rcParams["font.size"] = 14           # 基础字号
plt.rcParams["axes.titlesize"] = 18      # 标题字号
plt.rcParams["axes.labelsize"] = 16      # 轴标签字号
plt.rcParams["xtick.labelsize"] = 12
plt.rcParams["ytick.labelsize"] = 12
plt.rcParams["legend.fontsize"] = 12

# ================= 读入数据 =================
data_path = r'C:\Users\沐阳\Desktop\城市综合得分_pro.xlsx'
sheet_name = '全样本目标变量对比'

df = pd.read_excel(data_path, sheet_name=sheet_name)

# 选择需要分析的列
corr_cols = ['GDP', 'Local_exp', 'Post_rev', 'Wastewater', 'Score_A']
df = df[['Year'] + corr_cols]

# Spearman 相关系数
correlation_matrix = df[corr_cols].corr(method='spearman')

# ================= 绘制相关性矩阵 =================
mask = np.tril(np.ones_like(correlation_matrix, dtype=bool))
sns.set(style='white')

fig, ax = plt.subplots(figsize=(8, 6))

heatmap = sns.heatmap(
    correlation_matrix,
    mask=mask,
    annot=True,
    fmt=".2f",                               # 两位小数
    annot_kws={"fontweight": "bold"},        # 数值加粗
    cmap='coolwarm',
    square=True,
    cbar_kws={"shrink": 0.8},
    ax=ax,
    vmin=-1,
    vmax=1
)

# 标题、轴标签加粗
ax.set_title('Spearman Correlation Matrix', fontweight='bold', pad=16)
ax.set_xlabel('Variables', fontweight='bold')
ax.set_ylabel('Variables', fontweight='bold')

# x 轴标签：放上方、90°、加粗
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=90,
    ha='left',
    fontdict={
        'fontsize': 12,
        'fontfamily': 'Times New Roman',
        'fontweight': 'bold'
    }
)
ax.xaxis.tick_top()
ax.xaxis.set_label_position('top')

# y 轴标签：水平、加粗
ax.set_yticklabels(
    ax.get_yticklabels(),
    rotation=0,
    fontdict={
        'fontsize': 12,
        'fontfamily': 'Times New Roman',
        'fontweight': 'bold'
    }
)

# 隐藏顶部刻度线
ax.tick_params(axis='x', which='both', top=False)

# ================= 下三角气泡 =================
cmap = sns.color_palette("coolwarm", as_cmap=True)

for i in range(len(correlation_matrix)):
    for j in range(i):
        value = correlation_matrix.iloc[i, j]
        bubble_size = abs(value) * 1000  # 调整倍数控制气泡大小

        color = cmap(0.5 * (value + 1))  # [-1,1] -> [0,1]

        ax.scatter(
            j + 0.5,
            i + 0.5,
            s=bubble_size,
            color=color,
            alpha=0.6,
            edgecolor='w'
        )

# ================= 对角线直方图（基于轴坐标，解决错位） =================
num_columns = len(corr_cols)
cell_w = 1.0 / num_columns
cell_h = 1.0 / num_columns

for i, column in enumerate(corr_cols):
    # 在 axes 坐标系中放置对角线小图
    x0_ax = i * cell_w
    y0_ax = 1.0 - (i + 1) * cell_h

    inset_ax = ax.inset_axes([x0_ax, y0_ax, cell_w, cell_h])

    hist_color = cmap(0.75)

    sns.histplot(
        df[column],
        kde=True,
        bins=15,
        ax=inset_ax,
        color=hist_color,
        alpha=0.7
    )

    inset_ax.set_xticks([])
    inset_ax.set_yticks([])
    inset_ax.set_title('')
    inset_ax.set_xlabel('')
    inset_ax.set_ylabel('')
    inset_ax.set_facecolor('none')
    for spine in ['top', 'right', 'left', 'bottom']:
        inset_ax.spines[spine].set_visible(False)

# ================= 调整 colorbar =================
heatmap_pos = ax.get_position()
cbar = heatmap.collections[0].colorbar

cbar.ax.set_position([
    heatmap_pos.x1 + 0.02,
    heatmap_pos.y0 + 0.1,
    0.02,
    heatmap_pos.height - 0.2
])

cbar.set_ticks([-1, -0.5, 0, 0.5, 1])
cbar.ax.set_yticklabels(['-1', '-0.5', '0', '0.5', '1'], fontweight='bold')

plt.tight_layout()
plt.show()
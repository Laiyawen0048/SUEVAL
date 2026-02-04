import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

# ========= Style settings =========
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Times New Roman", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 13
plt.rcParams["axes.titlesize"] = 15
plt.rcParams["axes.labelsize"] = 13
plt.rcParams["xtick.labelsize"] = 11
plt.rcParams["ytick.labelsize"] = 11
plt.rcParams["legend.fontsize"] = 11
plt.rcParams["axes.linewidth"] = 1.0
plt.rcParams["grid.linestyle"] = "--"
plt.rcParams["grid.alpha"] = 0.4
sns.set_style("whitegrid")

# ---------- Read data ----------
data_path = r"C:\Users\沐阳\Desktop\城市得分结果\city_Dscore_chengdu.xlsx"
df = pd.read_excel(data_path)

# Optional inspect
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
print(df.head())

# ---------- Prepare dimension columns ----------
dimension_columns = [c for c in df.columns if c.startswith("D") and c[1:].isdigit()]
dimension_columns.sort(key=lambda x: int(x[1:]))
if not dimension_columns:
    raise ValueError("No dimension columns found. Expect columns like 'D1', 'D2', ...")

# ---------- Prepare years ----------
if 'Year' not in df.columns:
    raise ValueError("No 'Year' column found in the data.")
years = np.unique(df['Year'].dropna().astype(int))
years.sort()

# Aggregate if multiple rows per year exist (use mean)
agg = df.groupby('Year')[dimension_columns].mean().reindex(years)

# ---------- Custom colormap: teal -> yellow -> red (blue-green to red) ----------
from matplotlib.colors import LinearSegmentedColormap
cmap = LinearSegmentedColormap.from_list("teal_yellow_red", ["teal", "lightyellow", "red"])

# Map years to colors across the colormap
year_norm = mpl.colors.Normalize(vmin=years.min(), vmax=years.max())
year_colors = [cmap(year_norm(y)) for y in years]

# ---------- Plot settings ----------
n_dims = len(dimension_columns)
n_years = len(years)
x = np.arange(n_dims)
width = 0.8 / max(n_years, 1)

fig = plt.figure(figsize=(18, 11))
gs = fig.add_gridspec(3, 1, height_ratios=[3, 0.12, 1], hspace=0.35)
ax_main = fig.add_subplot(gs[0, 0])

# Draw bars for each year (side-by-side)
for i, year in enumerate(years):
    values = agg.loc[year].values
    ax_main.bar(x + i * width, values, width=width, label=str(year),
                color=year_colors[i], edgecolor='white', linewidth=0.6)

ax_main.set_xlabel("Dimension")
ax_main.set_ylabel("Score")
ax_main.set_title("City Dimension Scores Over Years")
ax_main.set_xticks(x + width * (n_years - 1) / 2)
ax_main.set_xticklabels(dimension_columns, rotation=45, ha='right')
ax_main.grid(True, axis='y', alpha=0.3)

# Ensure y starts at 0 and add some top margin
ymin, ymax = ax_main.get_ylim()
ax_main.set_ylim(bottom=0, top=max(ymax, agg.max().max() * 1.05))

# Legend
ax_main.legend(title="Year", loc='upper right', frameon=True)

# ---------- Horizontal colorbar showing year gradient ----------
ax_cbar = fig.add_subplot(gs[1, 0])
sm = mpl.cm.ScalarMappable(cmap=cmap, norm=year_norm)
sm.set_array([])
cbar = fig.colorbar(sm, cax=ax_cbar, orientation='horizontal')
cbar.set_label('Year')

# ---------- Table subplot ----------
ax_table = fig.add_subplot(gs[2, 0])
ax_table.axis('off')

table_data = []
for year in years:
    row = [str(year)] + [f"{v:.2f}" for v in agg.loc[year].values]
    table_data.append(row)

col_labels = ["Year"] + dimension_columns
table = ax_table.table(cellText=table_data, colLabels=col_labels, loc='center', cellLoc='center')

# Style table
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.25)

# Color the 'Year' column cells to match year colors for readability
for i, color in enumerate(year_colors):
    cell = table[i+1, 0]  # row i+1 (0 is header), column 0
    cell.set_facecolor(color)
    # choose text color based on luminance
    r, g, b, _ = color
    lum = 0.299*r + 0.587*g + 0.114*b
    cell.get_text().set_color('black' if lum > 0.6 else 'white')

plt.tight_layout()
plt.show()
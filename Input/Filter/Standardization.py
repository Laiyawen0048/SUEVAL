import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# ====== Font / style settings (English fonts) ======
plt.rcParams['font.family'] = ['Times New Roman', 'Arial', 'DejaVu Sans']
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 13
plt.rcParams['axes.titlesize'] = 15
plt.rcParams['axes.labelsize'] = 13
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['grid.linestyle'] = "--"
plt.rcParams['grid.alpha'] = 0.4
sns.set_style("whitegrid")

# ====== 1. Read city data ======
file_path = r"C:\Users\沐阳\PycharmProjects\pythonProject3\SUEVAL\Input\Data\city_data_final.xlsx"
df = pd.read_excel(file_path)

# Assume first 3 columns are meta: Year, City, Province (adjust if needed)
meta_cols = df.columns[:3]
feature_cols = df.columns[3:]
features = df[feature_cols].copy()

records = []               # processing records
standard_params = []       # standardization parameters

# ====== 2. User-specified negative-to-positive variables ======
neg_to_pos_vars = {
    # 'Wastewater',
    # 'SO2_emission',
}

# ====== 3. Indicator type sets ======
ratio_vars = {
    'PI_share_GDP', 'SI_share_GDP', 'TI_share_GDP',
    'Nat_growth_rate', 'Urban_rate',
    'PI_emp_rate', 'SI_emp_rate', 'TI_emp_rate',
    'Sewage_treat_rate', 'Waste_treat_rate', 'Solidwaste_rate',
    'Gen_solidwaste_rate', 'Sewage_plant_rate', 'Good_air_days',
    'GDP_growth'
}

index_vars = set()
score_vars = set()
bool_vars = {'is_forecast'}
no_clip_vars = {'Trade_emp', 'Hotel_emp'}

# ====== 4. Helper: Z-score function that also returns mean/std ======
def zscore_with_params(x: pd.Series):
    x = x.astype(float)
    x = x.replace([np.inf, -np.inf], np.nan)
    if x.notna().sum() == 0:
        return np.zeros(len(x), dtype=float), np.nan, np.nan

    mean_val = x.mean()
    x_filled = x.fillna(mean_val)
    std_val = x_filled.std(ddof=0)

    if std_val == 0 or np.isnan(std_val):
        return np.zeros(len(x), dtype=float), mean_val, std_val

    z = (x_filled - mean_val) / std_val
    return z.values, mean_val, std_val

# ====== 5. Negative-to-positive transformation ======
for col in list(feature_cols):
    if col in neg_to_pos_vars:
        x = features[col]
        if not np.issubdtype(x.dtype, np.number):
            records.append([
                col,
                np.nan,
                np.nan,
                x.nunique(),
                np.nan,
                np.nan,
                "Neg-to-pos requested but non-numeric → kept original"
            ])
            continue

        x_float = pd.to_numeric(x, errors='coerce').astype(float)
        x_float = x_float.replace([np.inf, -np.inf], np.nan)
        valid = x_float.dropna()

        if valid.empty:
            features[col] = x_float.fillna(0.0)
            method = "Neg-to-pos → all NaN → filled 0"
            col_min = np.nan
            col_max = np.nan
        else:
            min_v = valid.min()
            max_v = valid.max()
            x_inv = (max_v + min_v) - x_float
            features[col] = x_inv
            method = f"Neg-to-pos linear inversion (max={max_v:.4g}, min={min_v:.4g})"
            col_min = valid.min()
            col_max = valid.max()

        zero_ratio = (x == 0).mean() if pd.api.types.is_numeric_dtype(x) else np.nan
        skewness = x.skew() if pd.api.types.is_numeric_dtype(x) else np.nan
        unique_vals = x.nunique()
        records.append([
            col,
            zero_ratio,
            skewness,
            unique_vals,
            col_min,
            col_max,
            method
        ])

# ====== 6. Main processing & standardization (no clipping) ======
processed_cols = set([r[0] for r in records])
records_dict = {r[0]: r for r in records}

for col in feature_cols:
    x = features[col]
    col_name = col.strip()

    if not np.issubdtype(x.dtype, np.number):
        if col_name not in records_dict:
            records.append([
                col_name,
                np.nan,
                np.nan,
                x.nunique(),
                np.nan,
                np.nan,
                "Non-numeric (keep original)"
            ])
            records_dict[col_name] = records[-1]

        standard_params.append({
            "Variable_Name": col_name,
            "Transform_Type": "non_numeric_keep",
            "Shift": np.nan,
            "Log1p": False,
            "Mean": np.nan,
            "Std": np.nan
        })
        continue

    x_float = pd.to_numeric(x, errors='coerce').astype(float)
    x_float = x_float.replace([np.inf, -np.inf], np.nan)

    zero_ratio = (x_float == 0).mean()
    skewness = x_float.skew()
    unique_vals = x_float.nunique()
    col_min = x_float.min()
    col_max = x_float.max()

    if col_name in bool_vars:
        features[col] = x_float
        method = "Boolean → keep original"
        standard_params.append({
            "Variable_Name": col_name,
            "Transform_Type": "bool_keep",
            "Shift": 0.0,
            "Log1p": False,
            "Mean": np.nan,
            "Std": np.nan
        })

    elif col_name in ratio_vars:
        z_vals, mu, sigma = zscore_with_params(x_float)
        features[col] = z_vals
        method = "Ratio → Z-score"
        standard_params.append({
            "Variable_Name": col_name,
            "Transform_Type": "zscore",
            "Shift": 0.0,
            "Log1p": False,
            "Mean": mu,
            "Std": sigma
        })

    elif col_name in score_vars:
        z_vals, mu, sigma = zscore_with_params(x_float)
        features[col] = z_vals
        method = "Score → Z-score"
        standard_params.append({
            "Variable_Name": col_name,
            "Transform_Type": "zscore",
            "Shift": 0.0,
            "Log1p": False,
            "Mean": mu,
            "Std": sigma
        })

    elif col_name in index_vars:
        z_vals, mu, sigma = zscore_with_params(x_float)
        features[col] = z_vals
        method = "Index → Z-score"
        standard_params.append({
            "Variable_Name": col_name,
            "Transform_Type": "zscore",
            "Shift": 0.0,
            "Log1p": False,
            "Mean": mu,
            "Std": sigma
        })

    else:
        valid = x_float.dropna()
        if valid.empty:
            features[col] = np.zeros(len(x_float), dtype=float)
            method = "Other numeric → all NaN → filled 0"
            standard_params.append({
                "Variable_Name": col_name,
                "Transform_Type": "all_nan_zero",
                "Shift": 0.0,
                "Log1p": False,
                "Mean": np.nan,
                "Std": np.nan
            })
        else:
            min_val = valid.min()
            skew_val = valid.skew()

            if min_val >= 0 and skew_val > 1:
                x_log = np.log1p(x_float)
                z_vals, mu, sigma = zscore_with_params(x_log)
                features[col] = z_vals
                method = "Other numeric (>=0 & skew>1) → log1p + Z-score"
                standard_params.append({
                    "Variable_Name": col_name,
                    "Transform_Type": "log1p+zscore",
                    "Shift": 0.0,
                    "Log1p": True,
                    "Mean": mu,
                    "Std": sigma
                })

            elif min_val < 0:
                if abs(skew_val) <= 1:
                    z_vals, mu, sigma = zscore_with_params(x_float)
                    features[col] = z_vals
                    method = "Other numeric (with negatives, |skew|<=1) → Z-score"
                    standard_params.append({
                        "Variable_Name": col_name,
                        "Transform_Type": "zscore",
                        "Shift": 0.0,
                        "Log1p": False,
                        "Mean": mu,
                        "Std": sigma
                    })
                else:
                    eps = 1e-6
                    shift = -min_val + eps
                    x_shift = x_float + shift
                    x_log = np.log1p(x_shift)
                    z_vals, mu, sigma = zscore_with_params(x_log)
                    features[col] = z_vals
                    method = "Other numeric (neg & |skew|>1) → shift + log1p + Z-score"
                    standard_params.append({
                        "Variable_Name": col_name,
                        "Transform_Type": "shift+log1p+zscore",
                        "Shift": float(shift),
                        "Log1p": True,
                        "Mean": mu,
                        "Std": sigma
                    })

            else:
                z_vals, mu, sigma = zscore_with_params(x_float)
                features[col] = z_vals
                method = "Other numeric (>=0 & skew<=1) → Z-score"
                standard_params.append({
                    "Variable_Name": col_name,
                    "Transform_Type": "zscore",
                    "Shift": 0.0,
                    "Log1p": False,
                    "Mean": mu,
                    "Std": sigma
                })

    if col_name in records_dict:
        prev = records_dict[col_name]
        prev_method = prev[6] if len(prev) >= 7 else ""
        combined_method = f"{prev_method} → {method}" if prev_method else method
        records_dict[col_name] = [
            col_name,
            zero_ratio,
            skewness,
            unique_vals,
            col_min,
            col_max,
            combined_method
        ]
    else:
        records.append([
            col_name,
            zero_ratio,
            skewness,
            unique_vals,
            col_min,
            col_max,
            method
        ])
        records_dict[col_name] = records[-1]

records = [records_dict[c] for c in records_dict]

# ====== 7. Export standardized data, records & params ======
out_dir = r"C:\Users\沐阳\PycharmProjects\pythonProject3\SUEVAL\Input\Filter"
os.makedirs(out_dir, exist_ok=True)

output_path = os.path.join(out_dir, "City_Data_Standardized_Results.xlsx")
record_path = os.path.join(out_dir, "City_Data_Processing_Records.xlsx")
params_path = os.path.join(out_dir, "City_Standardization_Params.xlsx")

df_std = pd.concat([df[meta_cols].reset_index(drop=True),
                    features.reset_index(drop=True)], axis=1)
df_std.to_excel(output_path, index=False)

records_df = pd.DataFrame(records, columns=[
    "Variable_Name", "Zero_Ratio", "Skewness",
    "Unique_Values", "Min_Value", "Max_Value", "Processing_Method"
])
records_df.to_excel(record_path, index=False)

params_df = pd.DataFrame(standard_params, columns=[
    "Variable_Name", "Transform_Type", "Shift",
    "Log1p", "Mean", "Std"
])
params_df.to_excel(params_path, index=False)

print("✅ Standardization and classification completed (city data)!")
print(f"📑 Standardized data saved to: {output_path}")
print(f"📘 Processing records saved to: {record_path}")
print(f"📐 Standardization parameters saved to: {params_path}")

# ====== 8. Visualization: compare most skewed vars before/after (no numeric labels) ======
sk_df = records_df.copy()
sk_df["Skewness"] = pd.to_numeric(sk_df["Skewness"], errors='coerce')
sk_df = sk_df[sk_df["Skewness"].notna()]
sk_df["Abs_Skewness"] = sk_df["Skewness"].abs()
sk_df = sk_df.sort_values(by="Abs_Skewness", ascending=False)

top_skewed = sk_df["Variable_Name"].head(6).tolist()

if top_skewed:
    n_rows = len(top_skewed)
    fig, axes = plt.subplots(n_rows, 2, figsize=(12, 3 * n_rows))
    plt.subplots_adjust(hspace=1, wspace=0.5)

    if n_rows == 1:
        axes = np.array([axes])

    for i, col in enumerate(top_skewed):
        sns.histplot(df[col].dropna(), kde=True, ax=axes[i, 0],
                     color="steelblue", bins=40)
        axes[i, 0].set_title(f"{col} (Original)", fontsize=10)
        axes[i, 0].set_xlabel("")
        axes[i, 0].set_ylabel("")
        axes[i, 0].tick_params(axis='both', labelsize=8)

        sns.histplot(df_std[col].dropna(), kde=True, ax=axes[i, 1],
                     color="seagreen", bins=40)
        axes[i, 1].set_title(f"{col} (Processed)", fontsize=10)
        axes[i, 1].set_xlabel("")
        axes[i, 1].set_ylabel("")
        axes[i, 1].tick_params(axis='both', labelsize=8)

    plt.suptitle(
        "Most Skewed Variables (City): Before vs After Standardization",
        fontsize=14, fontweight='bold', y=0.99
    )

    plot_path = os.path.join(out_dir, "City_Skewness_Distribution_Comparison.png")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"📊 Distribution plots saved to: {plot_path}")

# ====== 9. Processing method distribution plot ======
method_counts = records_df["Processing_Method"].value_counts()

fig, ax = plt.subplots(figsize=(10, 5))
method_counts.plot(kind='barh', ax=ax, color='skyblue', edgecolor='black')

label_threshold = 3
for i, (method, count) in enumerate(method_counts.items()):
    if count >= label_threshold:
        ax.text(count + 0.1, i, f'{count}', va='center', fontsize=9)

def wrap_label(label, width=30):
    return '\n'.join([label[i:i + width] for i in range(0, len(label), width)])

ax.set_yticklabels([wrap_label(m) for m in method_counts.index], fontsize=9)
ax.set_xlabel('Number of Variables', fontsize=11, fontweight='bold')
ax.set_ylabel('Processing Method', fontsize=11, fontweight='bold')
ax.set_title('Distribution of Variable Processing Methods (City)',
             fontsize=13, fontweight='bold')
ax.grid(axis='x', alpha=0.3)

methods_plot_path = os.path.join(out_dir, "City_Processing_Methods_Distribution.png")
plt.tight_layout()
plt.savefig(methods_plot_path, dpi=300, bbox_inches='tight')
plt.show()
print(f"📊 Processing methods distribution saved to: {methods_plot_path}")

# ====== 10. Skewness comparison plot for top skewed vars ======
if top_skewed:
    skew_comparison = []
    for col in top_skewed:
        if col in df.columns and col in df_std.columns:
            skew_before = df[col].dropna().skew()
            skew_after = df_std[col].dropna().skew()
            skew_comparison.append({
                'Variable': col,
                'Skewness_Before': skew_before,
                'Skewness_After': skew_after,
                'Skewness_Reduction': abs(skew_before) - abs(skew_after)
            })

    skew_df = pd.DataFrame(skew_comparison)
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(skew_df))
    width = 0.35

    ax.bar(x - width / 2, skew_df['Skewness_Before'], width,
           label='Before', color='steelblue', alpha=0.8)
    ax.bar(x + width / 2, skew_df['Skewness_After'], width,
           label='After', color='seagreen', alpha=0.8)

    ax.set_xlabel('Variables', fontsize=11, fontweight='bold')
    ax.set_ylabel('Skewness', fontsize=11, fontweight='bold')
    ax.set_title('Skewness (City): Before vs After Standardization',
                 fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(skew_df['Variable'], rotation=30, ha='right', fontsize=9)
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    skew_plot_path = os.path.join(out_dir, "City_Skewness_Comparison.png")
    plt.tight_layout()
    plt.savefig(skew_plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"📊 Skewness comparison plot saved to: {skew_plot_path}")

print("\n🎯 All city-level processing completed successfully!")
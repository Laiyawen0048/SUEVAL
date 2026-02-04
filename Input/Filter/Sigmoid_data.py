import os
import pandas as pd
import numpy as np

# ====== 1. Path: read standardized reordered file ======
file_path = r"C:\Users\沐阳\PycharmProjects\pythonProject3\SUEVAL\Input\Index_system\City_Data_Standardized_Results_Reordered.xlsx"
df_std = pd.read_excel(file_path)

# use the same directory for outputs
out_dir = os.path.dirname(file_path)
os.makedirs(out_dir, exist_ok=True)

# Assume first 4 columns are meta: Year, City, Province, Region (adjust if needed)
meta_cols = df_std.columns[:4]
feature_cols = df_std.columns[4:]
features = df_std[feature_cols].copy()

# ====== 1.1 Ratio vars to exclude from sigmoid ======
ratio_vars = {
    'PI_share_GDP', 'SI_share_GDP', 'TI_share_GDP',
    'Nat_growth_rate', 'Urban_rate',
    'PI_emp_rate', 'SI_emp_rate', 'TI_emp_rate',
    'Sewage_treat_rate', 'Waste_treat_rate', 'Solidwaste_rate',
    'Gen_solidwaste_rate', 'Sewage_plant_rate', 'Good_air_days',
    'GDP_growth'
}

# ====== 2. Sigmoid function excluding ratio columns ======
def sigmoid_df_exclude_ratio(df: pd.DataFrame, ratio_cols: set) -> pd.DataFrame:
    """
    Apply element-wise sigmoid to numeric columns except those in ratio_cols.
    Non-numeric columns and ratio_cols are kept unchanged.
    """
    df_out = df.copy()
    cols_to_sigmoid = [c for c in df.columns if c not in ratio_cols]
    df_num = df_out[cols_to_sigmoid].apply(pd.to_numeric, errors='coerce')
    df_clip = df_num.clip(lower=-50, upper=50)
    df_sig = 1.0 / (1.0 + np.exp(-df_clip))
    df_out[cols_to_sigmoid] = df_sig
    return df_out

# ====== 3. Apply sigmoid (excluding ratio vars) ======
scaled_features = sigmoid_df_exclude_ratio(features, ratio_vars)

# ====== 4. Merge meta + scaled features and export to same folder as input ======
df_sig = pd.concat(
    [df_std[meta_cols].reset_index(drop=True),
     scaled_features.reset_index(drop=True)],
    axis=1
)

output_path_sig = os.path.join(out_dir, "City_Data_Sigmoid_Scaled.xlsx")
df_sig.to_excel(output_path_sig, index=False)

print("✅ Sigmoid scaling completed (vectorized, excluding ratio vars)!")
print(f"📑 Sigmoid-scaled data saved to: {output_path_sig}")
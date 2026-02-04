"""
Integrated pipeline for:
1) Missing-data diagnostics & basic filtering
2) City-level time-series interpolation (incl. one extra year, no forecast flag)
3) Structure-aware imputation using clustering and GDP-based grouping

"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 字体设置：保留你原来的配置
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# ==============================
# CONFIG
# ==============================
BASE_DIR = r"C:\Users\沐阳\PycharmProjects\pythonProject3\SUEVAL\Input\Data"

# ---- Input / Output paths ----
# 1) raw data (your original file)
raw_input_file = os.path.join(BASE_DIR, "城市原始数据_00-23.xlsx")
# 2) after hard deletion rules
missing_clean_file = os.path.join(BASE_DIR, "city_data_missing_clean.xlsx")
# 3) after interpolation (with one extra year, no flag)
filled_all_file = os.path.join(BASE_DIR, "city_data_filled.xlsx")
# 4) after structured imputation
final_output_file = os.path.join(BASE_DIR, "city_data_final.xlsx")

fig_dir = os.path.join(BASE_DIR, "fig_missing")
os.makedirs(fig_dir, exist_ok=True)

# ---- Missing filter thresholds (stage 1) ----
col_drop_thresh = 0.5        # drop feature columns with missing rate > 50%
gdp_sample_thresh = 0.5      # drop cities with GDP missing rate   > 50%
sample_all_feat_thresh = 0.7 # drop cities with all-feature missing rate > 70%

# ---- Time-series interpolation (stage 2) ----
use_global_year_range = False
global_min_year = 2000
global_max_year = 2023
bool_cols_candidate = ["Province", "Region"]

# ---- Structured imputation / clustering (stage 3) ----
base_features = ['GDP', 'Reg_pop', 'Ind_ent', 'FAI', 'Retail_sales']
min_cities_for_clustering = 5
k_clusters = 3
exclude_cols = ['City', 'Year', 'Province', 'Region']   # no forecast flag here


# ============================================================
# STAGE 1: HARD MISSING FILTERS + BASIC MISSING DIAGNOSTICS
# ============================================================
def stage1_missing_filter(input_path, output_path, fig_dir,
                          col_drop_thresh=0.5,
                          gdp_sample_thresh=0.5,
                          sample_all_feat_thresh=0.7):
    """Apply hard-deletion rules on features and cities, and save diagnostics."""
    df = pd.read_excel(input_path)
    if df.shape[1] < 5:
        raise ValueError("Need at least 5 columns: first 4 as meta, remaining as features.")

    # first 4 columns treated as meta: e.g. Year, City, Province, Region
    feature_cols = df.columns[4:].tolist()

    # 1) Drop feature columns with high missing rate
    col_missing_rate = df[feature_cols].isna().mean()
    cols_to_drop = col_missing_rate[col_missing_rate > col_drop_thresh].index.tolist()
    df = df.drop(columns=cols_to_drop)
    feature_cols = [c for c in feature_cols if c not in cols_to_drop]

    # 2) Build sample id by City
    if 'City' not in df.columns:
        raise KeyError("Column 'City' is required.")
    df['_sample_id'] = df['City'].astype(str)

    # 3) Drop cities with high GDP missing rate
    gdp_candidates = [c for c in df.columns if 'GDP' in str(c).upper()]
    if not gdp_candidates:
        raise KeyError("No GDP column detected (column name should contain 'GDP').")
    gdp_col = gdp_candidates[0]

    gdp_missing_by_city = df.groupby('_sample_id')[gdp_col].apply(lambda s: s.isna().mean())
    # 修正笔误：gdp_missing_by_gdp -> gdp_missing_by_city
    cities_drop_by_gdp = gdp_missing_by_city[gdp_missing_by_city > gdp_sample_thresh].index.tolist()
    if cities_drop_by_gdp:
        df = df[~df['_sample_id'].isin(cities_drop_by_gdp)].copy()

    # 4) Drop cities with high all-feature missing rate
    if feature_cols:
        feat_missing_by_city = df.groupby('_sample_id')[feature_cols].apply(
            lambda g: g.isna().mean(axis=1).mean()
        )
        # 修正笔误：feat_missing_by_allfeat -> feat_missing_by_city
        cities_drop_by_allfeat = feat_missing_by_city[feat_missing_by_city > sample_all_feat_thresh].index.tolist()
        if cities_drop_by_allfeat:
            df = df[~df['_sample_id'].isin(cities_drop_by_allfeat)].copy()

    # 5) Save cleaned data
    df = df.drop(columns=['_sample_id'], errors='ignore')
    df.to_excel(output_path, index=False)

    # 6) Feature-level missing statistics
    if feature_cols:
        final_col_missing_rate = df[feature_cols].isna().mean().sort_values(ascending=False)
        col_missing_df = final_col_missing_rate.reset_index()
        col_missing_df.columns = ['feature', 'missing_rate']
        col_missing_df.to_excel(os.path.join(fig_dir, "feature_missing_rate.xlsx"), index=False)
    else:
        col_missing_df = pd.DataFrame(columns=['feature', 'missing_rate'])

    # 7) City-level missing statistics
    if feature_cols:
        city_feat_missing = df.groupby('City')[feature_cols].apply(
            lambda g: g.isna().mean(axis=1).mean()
        ).sort_values(ascending=False)
        city_missing_df = city_feat_missing.reset_index()
        city_missing_df.columns = ['City', 'missing_rate']
        city_missing_df.to_excel(os.path.join(fig_dir, "city_missing_rate.xlsx"), index=False)
    else:
        city_missing_df = pd.DataFrame(columns=['City', 'missing_rate'])

    # 8) Plots: top-10 most-missing features & cities
    sns.set(style="whitegrid")

    if not col_missing_df.empty:
        top10_feat = col_missing_df.sort_values('missing_rate', ascending=False).head(10)
        plt.figure(figsize=(8, 6))
        sns.barplot(x='missing_rate', y='feature', data=top10_feat, palette='Reds_r')
        plt.xlim(0, 1)
        plt.xlabel('Missing rate')
        plt.title('Top-10 Features by Missing Rate')
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, "top10_feature_missing.png"), dpi=300)
        plt.close()

    if not city_missing_df.empty:
        top10_city = city_missing_df.sort_values('missing_rate', ascending=False).head(10)
        plt.figure(figsize=(8, 6))
        sns.barplot(x='missing_rate', y='City', data=top10_city, palette='Blues_r')
        plt.xlim(0, 1)
        plt.xlabel('Missing rate')
        plt.title('Top-10 Cities by Missing Rate')
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, "top10_city_missing.png"), dpi=300)
        plt.close()

    return df


# ============================================================
# STAGE 2: CITY-LEVEL INTERPOLATION (WITH NEXT YEAR, NO FLAG)
# ============================================================
def build_year_frame(group, use_global_range, g_min_year, g_max_year):
    """Construct year frame per city based on global or local year range."""
    if use_global_range:
        years = np.arange(g_min_year, g_max_year + 1)
    else:
        local_min = int(group['Year'].min())
        local_max = int(group['Year'].max())
        years = np.arange(local_min, local_max + 1)
    return pd.DataFrame({'Year': years})


def stage2_time_fill(input_path, output_path,
                     use_global_year_range=False,
                     global_min_year=2000,
                     global_max_year=2023,
                     bool_cols=None):
    """
    For each city, perform interpolation and add one extra year
    using a simple linear trend, without adding a forecast flag column.
    """
    if bool_cols is None:
        bool_cols = []

    df = pd.read_excel(input_path)
    if 'City' not in df.columns or 'Year' not in df.columns:
        raise ValueError("Columns 'City' and 'Year' are required.")

    # keep only existing categorical / boolean-like meta columns
    bool_cols = [c for c in bool_cols if c in df.columns]
    df = df.sort_values(['City', 'Year']).reset_index(drop=True)

    filled_list = []

    for city_name, city_group in df.groupby('City'):
        group = city_group.sort_values(by='Year').copy()

        # numeric columns for interpolation
        numeric_cols = group.columns.difference(['City', 'Year'] + bool_cols)

        group[numeric_cols] = group[numeric_cols].apply(pd.to_numeric, errors='coerce')
        group[numeric_cols] = group[numeric_cols].replace(0, np.nan)

        # year frame for this city
        year_frame = build_year_frame(group, use_global_year_range, global_min_year, global_max_year)
        merged = pd.merge(year_frame, group, on='Year', how='left')
        merged['City'] = city_name
        years = merged['Year'].values

        # interpolation + simple linear trend extrapolation at tail
        for col in numeric_cols:
            y = merged[col]
            if y.notna().sum() == 0:
                continue

            y_interp = y.interpolate(method='linear', limit_direction='both')

            notnull_mask = y.notna()
            x_known = years[notnull_mask]
            y_known = y[notnull_mask]

            if len(x_known) >= 2:
                k, b = np.polyfit(x_known, y_known, 1)

                first_valid_year = x_known.min()
                first_valid_value = y_known.iloc[0]
                y_interp[years < first_valid_year] = first_valid_value

                last_valid_year = x_known.max()
                y_interp[years > last_valid_year] = k * years[years > last_valid_year] + b

            merged[col] = y_interp

        # categorical / boolean columns: forward & backward fill within city
        for col in bool_cols:
            if col in merged.columns:
                merged[col] = merged[col].ffill().bfill()

        # add one extra year using global linear trend per column
        current_max_year = int(merged['Year'].max())
        next_year = current_max_year + 1
        pred_row = {'City': city_name, 'Year': next_year}

        for col in numeric_cols:
            y_full = merged[col]
            if y_full.notna().sum() == 0:
                pred_row[col] = np.nan
                continue

            x_full = merged['Year'].values
            y_full_vals = y_full.values
            valid_mask = ~np.isnan(y_full_vals)
            x_valid = x_full[valid_mask]
            y_valid = y_full_vals[valid_mask]

            if len(x_valid) >= 2:
                k, b = np.polyfit(x_valid, y_valid, 1)
                pred_row[col] = k * next_year + b
            else:
                pred_row[col] = y_valid[0] if len(y_valid) > 0 else np.nan

        for col in bool_cols:
            if col in merged.columns:
                last_val = merged.loc[merged['Year'] == current_max_year, col]
                last_val = last_val.iloc[0] if not last_val.empty else np.nan
                pred_row[col] = last_val

        merged = pd.concat([merged, pd.DataFrame([pred_row])], ignore_index=True)
        merged = merged.sort_values(by='Year').reset_index(drop=True)

        filled_list.append(merged)

    filled_df = pd.concat(filled_list, ignore_index=True, sort=False)
    filled_df.to_excel(output_path, index=False)
    return filled_df


# ============================================================
# STAGE 3: STRUCTURED IMPUTATION (CLUSTER + GDP LEVEL GROUPING)
# ============================================================
def city_feature_missing_rate(df, feature_cols):
    """City-level mean missing rate across given feature columns."""
    grp = df.groupby('City')[feature_cols].apply(lambda g: g.isna().mean(axis=1).mean())
    return grp.rename('missing_rate').reset_index()


def stage3_structured_imputation(input_path, output_path,
                                 base_features,
                                 exclude_cols,
                                 min_cities_for_clustering=5,
                                 k_clusters=3):
    """Structured imputation using (Year, Province, Region) clusters and GDP-level groups."""
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"File not found: {input_path}")

    df_all = pd.read_excel(input_path)

    for c in ['City', 'Year', 'Province', 'Region']:
        if c not in df_all.columns:
            raise KeyError(f"Missing required column: {c}")

    # numeric cols for filling (exclude ID/meta)
    numeric_cols = [
        c for c in df_all.columns
        if c not in exclude_cols and pd.api.types.is_numeric_dtype(df_all[c])
    ]

    pre_missing = city_feature_missing_rate(df_all, numeric_cols)
    df = df_all.copy()

    # key for year-province-region clusters
    df['ypr_key'] = (df['Year'].astype(str) + '|' +
                     df['Province'].astype(str) + '|' +
                     df['Region'].astype(str))

    # clustering within each (Year, Province, Region)
    base_features_available = [f for f in base_features if f in df.columns]
    if len(base_features_available) == 0:
        raise KeyError("No base_features found in data for clustering.")

    df['year_prov_region_cluster'] = np.nan
    for key in df['ypr_key'].unique():
        sub = df[df['ypr_key'] == key].copy()
        cities_in_sub = sub['City'].nunique()
        idx_sub = sub.index

        if cities_in_sub < min_cities_for_clustering:
            df.loc[idx_sub, 'year_prov_region_cluster'] = key + "|cluster_all"
            continue

        city_vectors = sub.groupby('City')[base_features_available].mean()
        valid_city_vectors = city_vectors.dropna(how='all')
        if valid_city_vectors.shape[0] < min_cities_for_clustering:
            df.loc[idx_sub, 'year_prov_region_cluster'] = key + "|cluster_all"
            continue

        scaler = StandardScaler()
        X = valid_city_vectors.fillna(0).values
        Xs = scaler.fit_transform(X)
        k = min(k_clusters, Xs.shape[0])
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(Xs)
        label_df = pd.DataFrame({'City': valid_city_vectors.index, 'label': labels})

        for _, row in sub.iterrows():
            city = row['City']
            idx = row.name
            if city in label_df['City'].values:
                lab = int(label_df.loc[label_df['City'] == city, 'label'].iloc[0])
                df.at[idx, 'year_prov_region_cluster'] = f"{key}|cluster_{lab}"
            else:
                df.at[idx, 'year_prov_region_cluster'] = key + "|cluster_all"

    # cluster-level mean imputation
    df_struct = df.copy()
    for key, sub in df_struct.groupby('year_prov_region_cluster'):
        cluster_means = sub[numeric_cols].mean(skipna=True)
        if cluster_means.isna().all():
            continue
        for col in numeric_cols:
            if pd.isna(cluster_means.get(col, np.nan)):
                continue
            mask = sub[col].isna()
            if mask.any():
                df_struct.loc[sub.index[mask], col] = cluster_means[col]

    # GDP-level group imputation: (Year, Province, Region, GDP_level)
    gdp_col_candidates = [c for c in df_struct.columns if 'GDP' in c.upper()]
    if not gdp_col_candidates:
        raise KeyError("No GDP column found (name must contain 'GDP').")
    gdp_col = gdp_col_candidates[0]

    df_struct['GDP_level'] = np.nan
    for key, sub in df_struct.groupby('ypr_key'):
        idx = sub.index
        gvals = sub[gdp_col]
        if gvals.dropna().empty:
            df_struct.loc[idx, 'GDP_level'] = 0
            continue

        q1 = gvals.quantile(0.25)
        q2 = gvals.quantile(0.5)
        q3 = gvals.quantile(0.75)

        def level_map(v):
            if pd.isna(v):
                return 0
            if v <= q1:
                return 1
            elif v <= q2:
                return 2
            elif v <= q3:
                return 3
            else:
                return 4

        df_struct.loc[idx, 'GDP_level'] = gvals.map(level_map)

    df_struct['GDP_level'] = df_struct['GDP_level'].astype(int)

    # group means by (ypr_key, GDP_level)
    for (key, lvl), sub in df_struct.groupby(['ypr_key', 'GDP_level']):
        idx = sub.index
        group_means = sub[numeric_cols].mean(skipna=True)
        if group_means.isna().all():
            continue
        for col in numeric_cols:
            if pd.isna(group_means.get(col, np.nan)):
                continue
            mask = sub[col].isna()
            if mask.any():
                df_struct.loc[idx[mask], col] = group_means[col]

    # fallback: (Year, Province, Region) mean
    for key, sub in df_struct.groupby('ypr_key'):
        idx = sub.index
        means = sub[numeric_cols].mean(skipna=True)
        if means.isna().all():
            continue
        for col in numeric_cols:
            mask = sub[col].isna()
            if mask.any():
                df_struct.loc[idx[mask], col] = means[col]

    post_missing = city_feature_missing_rate(df_struct, numeric_cols)
    _ = pre_missing.merge(post_missing, on='City', suffixes=('_pre', '_post'))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    cols_to_drop_final = ["ypr_key", "year_prov_region_cluster", "GDP_level"]
    df_export = df_struct.drop(columns=cols_to_drop_final, errors="ignore")
    df_export.to_excel(output_path, index=False)
    return df_struct


# ============================================================
# MAIN PIPELINE
# ============================================================
if __name__ == "__main__":
    # Stage 1: Hard missing filtering + diagnostics
    stage1_missing_filter(
        input_path=raw_input_file,
        output_path=missing_clean_file,
        fig_dir=fig_dir,
        col_drop_thresh=col_drop_thresh,
        gdp_sample_thresh=gdp_sample_thresh,
        sample_all_feat_thresh=sample_all_feat_thresh
    )

    # Stage 2: Interpolation + extra year
    stage2_time_fill(
        input_path=missing_clean_file,
        output_path=filled_all_file,
        use_global_year_range=use_global_year_range,
        global_min_year=global_min_year,
        global_max_year=global_max_year,
        bool_cols=bool_cols_candidate
    )

    # Stage 3: Structured imputation with clustering + GDP-level groups
    stage3_structured_imputation(
        input_path=filled_all_file,
        output_path=final_output_file,
        base_features=base_features,
        exclude_cols=exclude_cols,
        min_cities_for_clustering=min_cities_for_clustering,
        k_clusters=k_clusters
    )
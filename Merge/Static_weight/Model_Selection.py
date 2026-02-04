import os
import datetime
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, make_scorer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from scipy.stats import rankdata
from pandas.plotting import parallel_coordinates

warnings.filterwarnings("ignore")

# =============================================================================
# Paths and output folder
# =============================================================================
data_path = r"C:\Users\沐阳\Desktop\城市综合指数_pro\City_Data_Standardized_Results.xlsx"
output_dir_root = r"C:\Users\沐阳\Desktop"

time_tag = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = os.path.join(output_dir_root, f"Model_Comparison_TopConf_{time_tag}")
os.makedirs(output_dir, exist_ok=True)

# =============================================================================
# Plot style
# =============================================================================
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Times New Roman", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

plt.rcParams["font.size"] = 14
plt.rcParams["axes.titlesize"] = 18
plt.rcParams["axes.labelsize"] = 16
plt.rcParams["xtick.labelsize"] = 12
plt.rcParams["ytick.labelsize"] = 12
plt.rcParams["legend.fontsize"] = 12

# =============================================================================
# Random seed
# =============================================================================
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# =============================================================================
# Load data and basic preprocessing
# =============================================================================
data = pd.read_excel(data_path)

# shuffle rows to avoid ordering bias
rng = np.random.default_rng(RANDOM_STATE)
shuffled_idx = rng.permutation(len(data))
data = data.iloc[shuffled_idx].reset_index(drop=True)

# feature columns start index (adjust if necessary)
start_col = 8
feature_cols = data.columns[start_col:]

# targets to model
targets = ['GDP', 'Local_exp', 'Post_rev', 'Wastewater']

# encode non-numeric features if present
non_numeric = data[feature_cols].select_dtypes(exclude=[np.number]).columns.tolist()
if len(non_numeric) > 0:
    print(f"Warning: non-numeric feature columns detected, applying categorical encoding: {non_numeric}")
    for c in non_numeric:
        data[c] = data[c].astype("category").cat.codes

# =============================================================================
# Model definitions
# =============================================================================
base_models = {
    'Random Forest': RandomForestRegressor(
        n_estimators=200, max_depth=8, min_samples_split=10, min_samples_leaf=5,
        random_state=RANDOM_STATE, bootstrap=True
    ),
    'SVM': SVR(C=0.5, epsilon=0.2),
    'KNN': KNeighborsRegressor(n_neighbors=8, weights='distance'),
    'BP Neural Network': MLPRegressor(max_iter=1000, random_state=RANDOM_STATE),
    'Decision Tree': DecisionTreeRegressor(
        max_depth=4, min_samples_split=10, min_samples_leaf=5, random_state=RANDOM_STATE
    ),
    'XGBoost': XGBRegressor(
        max_depth=4, learning_rate=0.05, n_estimators=200,
        subsample=0.7, colsample_bytree=0.7, reg_alpha=0.05, reg_lambda=1.5,
        random_state=RANDOM_STATE
    ),
    'CatBoost': CatBoostRegressor(
        iterations=200, learning_rate=0.05, depth=4,
        l2_leaf_reg=3, random_state=RANDOM_STATE, verbose=0
    ),
}

def make_pipeline(model, scale=True):
    """Return a sklearn Pipeline with optional StandardScaler."""
    if scale:
        return Pipeline([('scaler', StandardScaler()), ('model', model)])
    else:
        return Pipeline([('model', model)])

# models that require scaling if present
scale_sensitive = {'Linear Regression', 'SVM', 'KNN', 'BP Neural Network'}

# =============================================================================
# Cross-validation and scoring
# =============================================================================
cv_kfold = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

rmse_scorer = make_scorer(mean_squared_error, squared=False)
mae_scorer  = make_scorer(mean_absolute_error)
r2_scorer   = make_scorer(r2_score)

summary_rows = []
all_results = []

# =============================================================================
# Main loop: train, CV, validate for each target
# =============================================================================
for target in targets:
    print("\n" + "=" * 80)
    print(f"Target: {target}")
    print("=" * 80)

    X = data[feature_cols].copy()
    y = data[target].copy()

    # drop rows where target is missing
    valid_idx = y.notna()
    X = X.loc[valid_idx]
    y = y.loc[valid_idx]

    # fill feature missing values with median
    X = X.fillna(X.median(numeric_only=True))

    # train/validation split (hold-out validation)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_STATE
    )

    results = {}
    fitted_models = {}

    for name, model in base_models.items():
        use_scale = name in scale_sensitive
        pipe = make_pipeline(model, scale=use_scale)

        # 5-fold CV on training set
        try:
            rmse_scores = cross_val_score(pipe, X_train, y_train, scoring=rmse_scorer, cv=cv_kfold)
            mae_scores = cross_val_score(pipe, X_train, y_train, scoring=mae_scorer, cv=cv_kfold)
            r2_scores  = cross_val_score(pipe, X_train, y_train, scoring=r2_scorer, cv=cv_kfold)

            rmse_cv_mean = rmse_scores.mean()
            rmse_cv_std  = rmse_scores.std(ddof=1)
            mae_cv_mean  = mae_scores.mean()
            r2_cv_mean   = r2_scores.mean()
        except Exception as e:
            print(f"[WARN][{name}] Cross-validation failed, fallback evaluation: {e}")
            split = int(0.8 * len(X_train))
            X_sub_tr, X_sub_va = X_train.iloc[:split], X_train.iloc[split:]
            y_sub_tr, y_sub_va = y_train.iloc[:split], y_train.iloc[split:]
            try:
                pipe.fit(X_sub_tr, y_sub_tr)
                y_sub_pred = pipe.predict(X_sub_va)
                rmse_cv_mean = mean_squared_error(y_sub_va, y_sub_pred, squared=False)
                rmse_cv_std  = np.nan
                mae_cv_mean  = mean_absolute_error(y_sub_va, y_sub_pred)
                r2_cv_mean   = r2_score(y_sub_va, y_sub_pred)
            except Exception as e2:
                print(f"[ERROR][{name}] Fallback failed: {e2}")
                rmse_cv_mean = np.nan
                rmse_cv_std  = np.nan
                mae_cv_mean  = np.nan
                r2_cv_mean   = np.nan

        # fit on full training set and evaluate on validation set
        pipe.fit(X_train, y_train)
        y_val_pred = pipe.predict(X_val)

        val_r2   = r2_score(y_val, y_val_pred)
        val_mae  = mean_absolute_error(y_val, y_val_pred)
        val_rmse = mean_squared_error(y_val, y_val_pred, squared=False)

        results[name] = {
            'CV_RMSE_mean': rmse_cv_mean,
            'CV_RMSE_std' : rmse_cv_std,
            'CV_MAE_mean' : mae_cv_mean,
            'CV_R2_mean'  : r2_cv_mean,
            'Val_R2'      : val_r2,
            'Val_RMSE'    : val_rmse,
            'Val_MAE'     : val_mae,
        }

        fitted_models[name] = pipe

        all_results.append({
            'Target': target,
            'Model': name,
            'CV_RMSE_mean': rmse_cv_mean,
            'CV_RMSE_std' : rmse_cv_std,
            'CV_MAE_mean' : mae_cv_mean,
            'CV_R2_mean'  : r2_cv_mean,
            'Val_R2'      : val_r2,
            'Val_RMSE'    : val_rmse,
            'Val_MAE'     : val_mae,
        })

    # print CV + validation metrics
    print("Model evaluation (5-fold CV on train + validation set):")
    for model_name, m in results.items():
        print(f"{model_name}:")
        print(f"  CV:    RMSE = {m['CV_RMSE_mean']:.4f} ± {m['CV_RMSE_std']:.4f}, "
              f"MAE = {m['CV_MAE_mean']:.4f}, R² = {m['CV_R2_mean']:.4f}")
        print(f"  Valid: R² = {m['Val_R2']:.4f}, RMSE = {m['Val_RMSE']:.4f}, MAE = {m['Val_MAE']:.4f}")

    # choose best model by CV_RMSE_mean
    best_model_name = min(
        results.keys(),
        key=lambda k: results[k]['CV_RMSE_mean'] if not np.isnan(results[k]['CV_RMSE_mean']) else np.inf
    )
    best_metrics = results[best_model_name]

    print(f"\nBest model for {target}: {best_model_name} | "
          f"CV RMSE = {best_metrics['CV_RMSE_mean']:.4f} ± {best_metrics['CV_RMSE_std']:.4f} | "
          f"Valid R² = {best_metrics['Val_R2']:.4f}, "
          f"Valid RMSE = {best_metrics['Val_RMSE']:.4f}, "
          f"Valid MAE = {best_metrics['Val_MAE']:.4f}")

    summary_rows.append({
        'Target': target,
        'Best Model (by CV_RMSE)': best_model_name,
        'CV_RMSE_mean': best_metrics['CV_RMSE_mean'],
        'CV_RMSE_std' : best_metrics['CV_RMSE_std'],
        'Valid_R²'    : best_metrics['Val_R2'],
        'Valid_RMSE'  : best_metrics['Val_RMSE'],
        'Valid_MAE'   : best_metrics['Val_MAE']
    })

    # -------------------------
    # Visualization: validation RMSE/MAE bar plot
    # -------------------------
    model_names_list = list(results.keys())
    rmse_values = [results[name]['Val_RMSE'] for name in model_names_list]
    mae_values  = [results[name]['Val_MAE']  for name in model_names_list]

    metrics_df = pd.DataFrame({
        'Model': model_names_list,
        'RMSE': rmse_values,
        'MAE' : mae_values
    })
    metrics_df = metrics_df.melt(id_vars='Model', var_name='Metric', value_name='Value')

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Model', y='Value', hue='Metric', data=metrics_df, palette='viridis', edgecolor="black")
    plt.title(f'Model comparison on validation set ({target})', fontweight="bold")
    plt.ylabel('Error', fontweight="bold")
    plt.xlabel('Model', fontweight="bold")
    plt.xticks(rotation=30, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.legend(title='Metric')
    plt.tight_layout()

    bar_fig_path = os.path.join(output_dir, f"{target}_Val_bar_RMSE_MAE.png")
    plt.savefig(bar_fig_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()
    print(f"Saved bar plot: {bar_fig_path}")

    # -------------------------
    # Visualization: validation R² radar chart
    # -------------------------
    angles = np.linspace(0, 2 * np.pi, len(model_names_list), endpoint=False).tolist()
    r2_values = [results[name]['Val_R2'] for name in model_names_list]

    r2_values += r2_values[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.plot(angles, r2_values, color='crimson', linewidth=2, label='R² (Validation)')
    ax.fill(angles, r2_values, color='lightcoral', alpha=0.4)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(model_names_list)

    for label, angle in zip(ax.get_xticklabels(), angles[:-1]):
        if 0 <= angle < np.pi/2 or 3*np.pi/2 <= angle <= 2*np.pi:
            label.set_horizontalalignment('left')
        else:
            label.set_horizontalalignment('right')

    ax.set_title(f'Model comparison (R², validation set) - {target}', fontweight="bold", pad=20)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))
    plt.tight_layout()

    radar_fig_path = os.path.join(output_dir, f"{target}_Val_radar_R2.png")
    plt.savefig(radar_fig_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()
    print(f"Saved radar plot: {radar_fig_path}")

    continue

# =============================================================================
# Save summaries
# =============================================================================
summary_df = pd.DataFrame(summary_rows)
print("\nSummary of best models per target (by lowest 5-fold CV RMSE mean; report Validation metrics):")
print(summary_df)

summary_path = os.path.join(output_dir, "best_model_summary.xlsx")
summary_df.to_excel(summary_path, index=False)

all_results_df = pd.DataFrame(all_results)
all_results_path = os.path.join(output_dir, "model_eval_results_detailed.xlsx")
all_results_df.to_excel(all_results_path, index=False)

print(f"\nAll scalar results saved to: {output_dir}")
print(f"  - Summary: {summary_path}")
print(f"  - Detailed: {all_results_path}")

# =============================================================================
# Cross-target visualizations
# =============================================================================

# Heatmap: validation RMSE
pivot_rmse_val = all_results_df.pivot_table(index="Model", columns="Target", values="Val_RMSE")

plt.figure(figsize=(8, 6))
sns.heatmap(pivot_rmse_val, annot=True, fmt=".3f", cmap="viridis_r", cbar_kws={"label": "Validation RMSE"})
plt.title("Validation RMSE across models and targets", fontweight="bold")
plt.xlabel("Target", fontweight="bold")
plt.ylabel("Model", fontweight="bold")
plt.tight_layout()
heatmap_rmse_path = os.path.join(output_dir, "heatmap_Val_RMSE_models_vs_targets.png")
plt.savefig(heatmap_rmse_path, dpi=300, bbox_inches="tight")
plt.show()
plt.close()
print(f"Saved heatmap (Val RMSE): {heatmap_rmse_path}")

# Heatmap: validation R²
pivot_r2_val = all_results_df.pivot_table(index="Model", columns="Target", values="Val_R2")

plt.figure(figsize=(8, 6))
sns.heatmap(pivot_r2_val, annot=True, fmt=".3f", cmap="YlGnBu", vmin=0.0, vmax=1.0, cbar_kws={"label": "Validation R²"})
plt.title("Validation R² across models and targets", fontweight="bold")
plt.xlabel("Target", fontweight="bold")
plt.ylabel("Model", fontweight="bold")
plt.tight_layout()
heatmap_r2_path = os.path.join(output_dir, "heatmap_Val_R2_models_vs_targets.png")
plt.savefig(heatmap_r2_path, dpi=300, bbox_inches="tight")
plt.show()
plt.close()
print(f"Saved heatmap (Val R²): {heatmap_r2_path}")

# Average rank by validation RMSE
df_rank = all_results_df.copy()
df_rank["Rank_Val_RMSE"] = df_rank.groupby("Target")["Val_RMSE"].transform(lambda x: rankdata(x, method="average"))

avg_rank = df_rank.groupby("Model")["Rank_Val_RMSE"].mean().sort_values()

plt.figure(figsize=(8, 5))
plt.barh(avg_rank.index, avg_rank.values, color="#4C78A8", edgecolor="black")
for i, v in enumerate(avg_rank.values):
    plt.text(v, i, f"{v:.2f}", va="center", ha="left", fontsize=12)
plt.gca().invert_yaxis()
plt.xlabel("Average rank (Validation RMSE, lower is better)", fontweight="bold")
plt.title("Average model ranking across targets (validation RMSE)", fontweight="bold")
plt.tight_layout()
avg_rank_path = os.path.join(output_dir, "avg_rank_models_by_Val_RMSE.png")
plt.savefig(avg_rank_path, dpi=300, bbox_inches="tight")
plt.show()
plt.close()
print(f"Saved average rank plot: {avg_rank_path}")

# Multi-line: validation MAE across targets
pivot_mae_val = all_results_df.pivot_table(index="Model", columns="Target", values="Val_MAE")

plt.figure(figsize=(10, 6))
x_positions = np.arange(len(targets))

for model_idx, (model_name, row) in enumerate(pivot_mae_val.iterrows()):
    mae_values = [row[t] for t in targets]
    plt.plot(x_positions, mae_values, marker='o', linewidth=2, alpha=0.8, label=model_name)

plt.xticks(x_positions, targets)
plt.xlabel("Target", fontweight="bold")
plt.ylabel("Validation MAE", fontweight="bold")
plt.title("Validation MAE across targets for each model", fontweight="bold")
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.legend(loc="upper left", fontsize=10, frameon=True, framealpha=0.9, edgecolor="black")
plt.tight_layout()

mae_lines_path = os.path.join(output_dir, "lines_Val_MAE_models_across_targets.png")
plt.savefig(mae_lines_path, dpi=300, bbox_inches="tight")
plt.show()
plt.close()
print(f"Saved multi-line MAE plot: {mae_lines_path}")

# Parallel coordinates: validation R² across targets
r2_for_parallel = pivot_r2_val.reset_index()
r2_for_parallel["ModelName"] = r2_for_parallel["Model"]

plt.figure(figsize=(10, 6))
parallel_coordinates(r2_for_parallel, class_column="ModelName", cols=targets, colormap=plt.get_cmap("tab10"), linewidth=2, alpha=0.7)
plt.title("Parallel coordinates: Validation R² across targets", fontweight="bold")
plt.ylabel("Validation R²", fontweight="bold")
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
parallel_path = os.path.join(output_dir, "parallel_coordinates_Val_R2_models.png")
plt.savefig(parallel_path, dpi=300, bbox_inches="tight")
plt.show()
plt.close()
print(f"Saved parallel coordinates plot: {parallel_path}")

print("\n✅ All training, cross-validation, validation evaluation, and visualizations completed.")
print(f"Output directory: {output_dir}")
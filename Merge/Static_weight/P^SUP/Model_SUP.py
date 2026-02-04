import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, RepeatedKFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from xgboost import XGBRegressor

import warnings
warnings.filterwarnings("ignore")

# ========= Global plotting settings =========
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
plt.rcParams["axes.unicode_minus"] = False

# Font sizes
plt.rcParams["font.size"] = 16          # base font size
plt.rcParams["axes.titlesize"] = 18     # axis title size
plt.rcParams["axes.labelsize"] = 16     # x/y label size
plt.rcParams["xtick.labelsize"] = 14
plt.rcParams["ytick.labelsize"] = 14
plt.rcParams["legend.fontsize"] = 14

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ========= Paths and parameters =========
data_path = r"C:\Users\沐阳\Desktop\城市综合指数_pro\City_Data_Standardized_Results.xlsx"
target = 'Wastewater'                 # target variable: 'GDP'/'Local_exp'/'Post_rev'/'Wastewater'
start_col = 8                          # index of first feature column
corr_thresh = 0.95
inter_corr_thresh = 0.95
save_path = r"C:\Users\沐阳\Desktop\xgb_Wastewater_analysis_results.xlsx"

# ========= Utility functions =========
def compute_target_correlations(df, feature_cols, target):
    """Compute Spearman correlation between each feature and the target."""
    s = df[feature_cols].corrwith(df[target], method='spearman').dropna()
    out = s.to_frame('corr').reset_index().rename(columns={'index': 'Feature'})
    out['abs_corr'] = out['corr'].abs()
    out = out.sort_values('abs_corr', ascending=False)
    return out

def drop_high_corr_with_target(features_df, corr_df, thresh):
    """Interface placeholder: return features to keep and those to drop based on correlation with target."""
    s = corr_df
    to_drop = s.loc[s['abs_corr'] >= thresh, 'Feature'].tolist()
    kept = [c for c in features_df.columns if c not in to_drop]
    return kept, to_drop

def drop_inter_correlated_features(features_df, thresh):
    """Interface placeholder: identify and optionally drop highly inter-correlated features."""
    corr = features_df.corr(method='spearman').abs()
    upper = np.triu(np.ones(corr.shape), k=1).astype(bool)
    corr_vals = corr.where(upper)

    high_corr_pairs = []
    to_drop = []
    for i in range(len(corr.columns)):
        for j in range(i + 1, len(corr.columns)):
            v = corr.iloc[i, j]
            if v >= thresh:
                high_corr_pairs.append((corr.columns[i], corr.columns[j], round(v, 4)))
                if corr.columns[j] not in to_drop:
                    to_drop.append(corr.columns[j])
    kept = [c for c in features_df.columns if c not in to_drop]
    return kept, to_drop, high_corr_pairs

def metrics(y_true, y_pred):
    """Return RMSE, R^2, and MAE between true and predicted values."""
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    return rmse, r2, mae

# ========= Data loading and preprocessing =========
data = pd.read_excel(data_path)
feature_cols = data.columns[start_col:]

# Encode non-numeric features (if any) using integer categorical codes
non_numeric = data[feature_cols].select_dtypes(exclude=[np.number]).columns.tolist()
if non_numeric:
    print(f"Detected non-numeric features; applying categorical encoding: {non_numeric}")
    for c in non_numeric:
        data[c] = data[c].astype('category').cat.codes

# Keep only rows with a defined target; impute feature missing values with median (column-wise)
data = data.loc[data[target].notna()].copy()
data[feature_cols] = data[feature_cols].fillna(data[feature_cols].median(numeric_only=True))

# ========= Feature selection pipeline =========
corr_df = compute_target_correlations(data, feature_cols, target)
feat_A = list(feature_cols)

# 1) Remove features with very weak Spearman correlation with the target
weak_corr_thresh = 0.05
weak_mask = corr_df['abs_corr'] < weak_corr_thresh
weak_features = corr_df.loc[weak_mask, 'Feature'].tolist()
feat_B = [c for c in feat_A if c not in weak_features]

print(f"\nOriginal number of features: {len(feat_A)}")
print(f"Features with |Spearman ρ| < {weak_corr_thresh}: {len(weak_features)}")
print(f"Remaining after weak-correlation filtering: {len(feat_B)}")

# 2) Inspect pairwise inter-feature correlations (Spearman, absolute)
corr_matrix = data[feat_B].corr(method='spearman').abs()
upper = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
corr_vals = corr_matrix.where(upper)

high_corr_pairs = []
for i in range(len(corr_vals.columns)):
    for j in range(i + 1, len(corr_vals.columns)):
        v = corr_vals.iloc[i, j]
        if pd.notnull(v) and v >= inter_corr_thresh:
            high_corr_pairs.append(
                (corr_vals.columns[i], corr_vals.columns[j], round(float(v), 4))
            )

# Current workflow: do not drop features automatically based on target/inter-correlation
dropped_target_corr = []     # placeholder for features dropped due to target correlation
dropped_inter_corr = []      # placeholder for features dropped due to inter-correlation
feat_C = feat_B

selected_features = feat_C
print(f"Final number of features selected for modeling: {len(selected_features)}")

# ========= Shuffle dataset globally and split =========
data_shuffled = data.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
X = data_shuffled[selected_features]
y = data_shuffled[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=RANDOM_STATE)

# ========= XGBoost hyperparameters =========
xgb_params = {
    "max_depth": 4,
    "min_child_weight": 5,
    "gamma": 0.1,
    "subsample": 0.7,
    "colsample_bytree": 0.7,
    "colsample_bylevel": 0.7,
    "learning_rate": 0.05,
    "n_estimators": 800,
    "reg_alpha": 0.25,
    "reg_lambda": 1.5,
    "objective": "reg:squarederror",
    "tree_method": "hist",
    "random_state": RANDOM_STATE,
}

# ========= Repeated cross-validation (5 folds x 3 repeats) =========
cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=RANDOM_STATE)
cv_model = XGBRegressor(**xgb_params)

# Negative MSE is returned by cross_val_score for 'neg_mean_squared_error'
mse_scores = cross_val_score(cv_model, X_train, y_train, scoring='neg_mean_squared_error', cv=cv, n_jobs=-1)
rmse_cv_mean, rmse_cv_std = np.sqrt(-mse_scores).mean(), np.sqrt(-mse_scores).std()

r2_scores = cross_val_score(cv_model, X_train, y_train, scoring='r2', cv=cv, n_jobs=-1)
r2_cv_mean, r2_cv_std = r2_scores.mean(), r2_scores.std()

mae_scores = -cross_val_score(cv_model, X_train, y_train, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
mae_cv_mean, mae_cv_std = mae_scores.mean(), mae_scores.std()

# ========= Train/validation split for early stopping =========
X_tr, X_va, y_tr, y_va = train_test_split(X_train, y_train, test_size=0.3, random_state=RANDOM_STATE)

model = XGBRegressor(**xgb_params)
model.fit(
    X_tr, y_tr,
    eval_set=[(X_tr, y_tr), (X_va, y_va)],
    eval_metric="rmse",
    early_stopping_rounds=80,
    verbose=False
)

evals_result = model.evals_result()
train_curve = evals_result['validation_0']['rmse']
valid_curve = evals_result['validation_1']['rmse']
best_iter = model.best_iteration

# ========= Performance metrics =========
y_tr_pred = model.predict(X_tr)
y_va_pred = model.predict(X_va)
y_te_pred = model.predict(X_test)

rmse_tr, r2_tr, mae_tr = metrics(y_tr, y_tr_pred)
rmse_va, r2_va, mae_va = metrics(y_va, y_va_pred)
rmse_te, r2_te, mae_te = metrics(y_test, y_te_pred)

print("\n===== Repeated 5-Fold x 3 Cross-Validation (on training set) =====")
print(f"CV RMSE: {rmse_cv_mean:.4f} ± {rmse_cv_std:.4f}")
print(f"CV R²  : {r2_cv_mean:.4f} ± {r2_cv_std:.4f}")
print(f"CV MAE : {mae_cv_mean:.4f} ± {mae_cv_std:.4f}")

print("\n===== Training / Validation (with early stopping) =====")
print(f"Best iteration: {best_iter}")
print(f"Train -> RMSE: {rmse_tr:.4f}, R²: {r2_tr:.4f}, MAE: {mae_tr:.4f}")
print(f"Valid -> RMSE: {rmse_va:.4f}, R²: {r2_va:.4f}, MAE: {mae_va:.4f}")

print("\n===== Test (held-out) =====")
print(f"Test  -> RMSE: {rmse_te:.4f}, R²: {r2_te:.4f}, MAE: {mae_te:.4f}")

# ========= Summary table =========
summary = pd.DataFrame({
    'CV_RMSE_mean': [rmse_cv_mean], 'CV_RMSE_std': [rmse_cv_std],
    'CV_R2_mean': [r2_cv_mean],     'CV_R2_std': [r2_cv_std],
    'CV_MAE_mean': [mae_cv_mean],   'CV_MAE_std': [mae_cv_std],
    'Best_Iteration': [best_iter],
    'Train_RMSE': [rmse_tr], 'Train_R2': [r2_tr], 'Train_MAE': [mae_tr],
    'Valid_RMSE': [rmse_va], 'Valid_R2': [r2_va], 'Valid_MAE': [mae_va],
    'Test_RMSE': [rmse_te],  'Test_R2': [r2_te],  'Test_MAE': [mae_te],
    'n_features_used': [len(selected_features)],
    'Dropped_target_corr_cnt': [len(dropped_target_corr)],
    'Dropped_inter_corr_cnt': [len(dropped_inter_corr)]
})

# ========= Feature importance =========
importance_df = (
    pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': model.feature_importances_
    })
    .sort_values('Importance', ascending=False)
)

# ========= Save outputs to an Excel workbook =========
with pd.ExcelWriter(save_path, engine='openpyxl') as writer:
    summary.to_excel(writer, sheet_name='summary', index=False)
    corr_df.to_excel(writer, sheet_name='target_correlation', index=False)
    importance_df.to_excel(writer, sheet_name='feature_importance', index=False)
    if high_corr_pairs:
        pd.DataFrame(high_corr_pairs, columns=['Feature_A', 'Feature_B', 'Spearman_corr']).to_excel(
            writer, sheet_name='high_corr_features', index=False
        )

print(f"\n✅ Results saved to: {save_path}")
print(f"Number of features used: {len(selected_features)}")

# ========= Visualizations =========
# (1) Training and validation RMSE curves
plt.figure(figsize=(12, 6))
plt.plot(train_curve, label='Train RMSE', color='steelblue')
plt.plot(valid_curve, label='Valid RMSE', color='tomato')
plt.title('XGBoost RMSE Iteration Curve')
plt.xlabel('Iterations'); plt.ylabel('RMSE'); plt.legend(); plt.grid(); plt.tight_layout(); plt.show()

# (2) Top-10 feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df.head(10), palette='viridis')
plt.title('Top 10 Feature Importance')
plt.xlabel('Importance Score'); plt.ylabel('Feature'); plt.grid(axis='x'); plt.tight_layout(); plt.show()

# (3) Actual vs. predicted (subset)
sample_size = min(100, len(y_test))
plt.figure(figsize=(12, 6))
plt.plot(y_test.values[:sample_size], label='Actual', color='royalblue', marker='o', ms=4, ls='None')
plt.plot(y_te_pred[:sample_size], label='Predicted', color='tomato', lw=2)
plt.title('Actual vs. Predicted (test subset)'); plt.legend(); plt.grid(); plt.tight_layout(); plt.show()

# (4) Residual distribution
residuals = y_test.values - y_te_pred
plt.figure(figsize=(10, 6))
sns.histplot(residuals, bins=30, kde=True, color='steelblue', edgecolor='black')
plt.axvline(0, color='red', ls='--', lw=2)
plt.title('Residuals Histogram'); plt.grid(); plt.tight_layout(); plt.show()

print("\nAll model training, evaluation, and outputs have been completed.")
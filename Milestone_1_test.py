import pandas as pd
import numpy as np
import pickle
import sys
import re

from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings("ignore")
# ════════════════════════════════════════════════════════
# 1. LOAD TEST CSV
# ════════════════════════════════════════════════════════

df = pd.read_csv("test_data_reg.csv")

# =====================================================
# SAVE ACTUAL LABELS FOR COMPARISON (IF THEY EXIST)
# =====================================================

y_true = None

if "JobSat" in df.columns:

    df = df.dropna(subset=["JobSat"])

    y_true = df["JobSat"].copy()

    # remove target column before preprocessing
    df = df.drop(columns=["JobSat"])

    print(f"Found JobSat column for evaluation. ({len(y_true)} valid rows)")

if "ResponseId" in df.columns:
    df = df.drop(columns=["ResponseId"])


# ════════════════════════════════════════════════════════
# 2. LOAD SAVED ARTIFACTS
# ════════════════════════════════════════════════════════

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


rf_imputer          = load_pickle("saved_pipeline_M1/rf_imputer.pkl")

numerical           = load_pickle("saved_pipeline_M1/numerical_cols.pkl")
numeric_cols        = load_pickle("saved_pipeline_M1/numeric_cols.pkl")

iqr_bounds          = load_pickle("saved_pipeline_M1/iqr_bounds.pkl")
skewed_cols         = load_pickle("saved_pipeline_M1/skewed_cols.pkl")

binary_map          = load_pickle("saved_pipeline_M1/binary_map.pkl")

cols_to_drop        = load_pickle("saved_pipeline_M1/cols_to_drop.pkl")

ohe_columns         = load_pickle("saved_pipeline_M1/ohe_columns.pkl")

freq_maps           = load_pickle("saved_pipeline_M1/freq_maps.pkl")

# Note: Milestone 1 saves mlb_dict under this filename
mlb_encoders        = load_pickle("saved_pipeline_M1/mlb_encoders.pkl")

to_drop_corr        = load_pickle("saved_pipeline_M1/corr_drop_cols.pkl")

selector            = load_pickle("saved_pipeline_M1/variance_selector.pkl")

selector_input_cols = load_pickle("saved_pipeline_M1/selector_input_cols.pkl")

top_features        = load_pickle("saved_pipeline_M1/top_features.pkl")


# ════════════════════════════════════════════════════════
# 3. LOAD MODELS
# ════════════════════════════════════════════════════════

model_paths = {
    "Linear Regression":      "saved_pipeline_M1/linear_regression.pkl",
    "Random Forest":          "saved_pipeline_M1/random_forest.pkl",
    "Gradient Boosting":      "saved_pipeline_M1/gradient_boosting.pkl",
    "Hist Gradient Boosting": "saved_pipeline_M1/hist_gradient_boosting.pkl",
    "Voting Regressor":       "saved_pipeline_M1/voting_regressor.pkl",
}

models = {}

for model_name, model_path in model_paths.items():

    models[model_name] = load_pickle(model_path)

    print(f"Loaded: {model_name}")


# ════════════════════════════════════════════════════════
# 4. DROP HIGH-NULL & ID COLUMNS
# ════════════════════════════════════════════════════════

df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

print(f"Shape after dropping high-null/ID columns: {df.shape}")


# ════════════════════════════════════════════════════════
# 5. IMPUTATION
# ════════════════════════════════════════════════════════

numerical_present = [c for c in numerical if c in df.columns]

df[numerical_present] = df[numerical_present].replace([np.inf, -np.inf], np.nan)
df[numerical_present] = df[numerical_present].clip(lower=-1e12, upper=1e12)
df[numerical_present] = rf_imputer.transform(df[numerical_present].astype(np.float64))

categorical_present = df.select_dtypes(include='object').columns
df[categorical_present] = df[categorical_present].fillna("Unknown")

print("Imputation completed.")


# ════════════════════════════════════════════════════════
# 6. OUTLIER CAPPING
# ════════════════════════════════════════════════════════

for col, (lower, upper) in iqr_bounds.items():
    if col in df.columns:
        df[col] = df[col].clip(lower, upper)

print("Outlier capping completed.")


# ════════════════════════════════════════════════════════
# 7. LOG TRANSFORM
# ════════════════════════════════════════════════════════

for col in skewed_cols:
    if col in df.columns:
        df[col] = np.log1p(df[col])

print("Log transform completed.")


# ════════════════════════════════════════════════════════
# 8. BINARY ENCODING
# ════════════════════════════════════════════════════════

for col in df.select_dtypes(include='object').columns:
    unique_vals = set(df[col].dropna().unique())
    if unique_vals.issubset(set(binary_map.keys())):
        df[col] = df[col].map(binary_map)

print("Binary encoding completed.")


# ════════════════════════════════════════════════════════
# 9. MULTI-VALUE (MLB) ENCODING
# ════════════════════════════════════════════════════════

for col, mlb in mlb_encoders.items():
    if col in df.columns:

        df[col] = df[col].fillna("").apply(lambda x: x.split(";") if x != "" else [])

        known_classes = set(mlb.classes_)
        df[col] = df[col].apply(
            lambda x: [i if i in known_classes else "UNKNOWN" for i in x]
        )

        encoded = mlb.transform(df[col])
        cols    = [f"{col}_{val}" for val in mlb.classes_]

        df = pd.concat(
            [df, pd.DataFrame(encoded, columns=cols, index=df.index).astype(int)],
            axis=1
        ).drop(columns=[col])

print("Multi-value encoding completed.")


# ════════════════════════════════════════════════════════
# 10. FREQUENCY ENCODING
# ════════════════════════════════════════════════════════

for col, freq in freq_maps.items():
    if col in df.columns:
        df[col] = df[col].map(freq).fillna(0)

print("Frequency encoding completed.")


# ════════════════════════════════════════════════════════
# 11. ONE-HOT ENCODING
# ════════════════════════════════════════════════════════

low_card_cols = list(df.select_dtypes(include='str').columns)
df = pd.get_dummies(df, columns=low_card_cols, drop_first=True, dtype=int)

print("One-hot encoding completed.")


# ════════════════════════════════════════════════════════
# 12. SANITIZE COLUMN NAMES
# ════════════════════════════════════════════════════════

df.columns = [re.sub(r'[^A-Za-z0-9_]', '_', col) for col in df.columns]

print("Column name sanitization completed.")


# ════════════════════════════════════════════════════════
# 12b. ALIGN TO TRAINING OHE COLUMNS  ← moved to AFTER sanitize
# ════════════════════════════════════════════════════════

df = df.reindex(columns=ohe_columns, fill_value=0)

print(f"Shape after OHE alignment: {df.shape}")


# ════════════════════════════════════════════════════════
# 13. FEATURE SELECTION
# ════════════════════════════════════════════════════════

df = df.drop(columns=[c for c in to_drop_corr if c in df.columns])

df = df.reindex(columns=selector_input_cols, fill_value=0)

selected_variance_cols = np.array(selector_input_cols)[selector.get_support()]
df = pd.DataFrame(
    selector.transform(df),
    columns=selected_variance_cols,
    index=df.index
)

df = df.reindex(columns=top_features, fill_value=0)

print(f"Final feature shape: {df.shape}")


# ════════════════════════════════════════════════════════
# 14. RUN ALL MODELS
# ════════════════════════════════════════════════════════

results = []

print("\n================ MODEL RESULTS ================\n")

for model_name, model in models.items():

    preds = model.predict(df)
    preds = np.array(preds).flatten()

    if y_true is not None:

        mse  = mean_squared_error(y_true, preds)
        rmse = mse ** 0.5
        r2   = r2_score(y_true, preds)

        results.append({
            "Model": model_name,
            "MSE":   mse,
            "RMSE":  rmse,
            "R2":    r2,
        })

        print(f"{model_name:25s}  MSE = {mse:.4f}  RMSE = {rmse:.4f}  R² = {r2:.4f}")

    else:

        print(f"{model_name:25s}  No labels found in test file.")


# ════════════════════════════════════════════════════════
# 15. PRINT SUMMARY
# ════════════════════════════════════════════════════════

if len(results) > 0:

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(by="R2", ascending=False)

    print("\n================ FINAL METRICS ================\n")
    print(results_df.to_string(index=False))

print("\nDone.")
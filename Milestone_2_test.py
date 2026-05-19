

import pandas as pd
import numpy as np
import pickle
import sys
import re

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings("ignore")

# ════════════════════════════════════════════════════════
# 1. LOAD TEST CSV
# ════════════════════════════════════════════════════════

test_path = "test_data_class.csv"

df = pd.read_csv(test_path)

print(f"\nLoaded test file: {test_path}")
print(f"Shape: {df.shape}")

# =====================================================
# SAVE ACTUAL LABELS FOR COMPARISON (IF THEY EXIST)
# =====================================================

y_true = None

if "JobSatLevel" in df.columns:

    y_true = df["JobSatLevel"].copy()

    # remove target column before preprocessing
    df = df.drop(columns=["JobSatLevel"])

    print("Found JobSatLevel column for evaluation.")


# ════════════════════════════════════════════════════════
# 2. LOAD SAVED ARTIFACTS
# ════════════════════════════════════════════════════════

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


# imputers / preprocessing
rf_imputer          = load_pickle("saved_pipeline_M2/rf_imputer.pkl")

numerical           = load_pickle("saved_pipeline_M2/numerical_cols.pkl")
numeric_cols        = load_pickle("saved_pipeline_M2/numeric_cols.pkl")

iqr_bounds          = load_pickle("saved_pipeline_M2/iqr_bounds.pkl")
# skewed_cols         = load_pickle("saved_pipeline_M2/skewed_cols.pkl")

binary_map          = load_pickle("saved_pipeline_M2/binary_map.pkl")

cols_to_drop        = load_pickle("saved_pipeline_M2/cols_to_drop.pkl")

ohe_columns         = load_pickle("saved_pipeline_M2/ohe_columns.pkl")

freq_maps           = load_pickle("saved_pipeline_M2/freq_maps.pkl")

mlb_encoders        = load_pickle("saved_pipeline_M2/mlb_encoders.pkl")
top_values_map      = load_pickle("saved_pipeline_M2/top_values_map.pkl")

to_drop_corr        = load_pickle("saved_pipeline_M2/corr_drop_cols.pkl")

selector            = load_pickle("saved_pipeline_M2/variance_selector.pkl")

selected            = load_pickle("saved_pipeline_M2/mi_selected_features.pkl")

top_features        = load_pickle("saved_pipeline_M2/top_features.pkl")

label_map           = load_pickle("saved_pipeline_M2/label_map.pkl")

selector_input_cols = load_pickle("saved_pipeline_M2/selector_input_cols.pkl")


# ════════════════════════════════════════════════════════
# 3. LOAD MODEL
# ════════════════════════════════════════════════════════

model_paths = {
    "Logistic Regression": "saved_pipeline_M2/best_logistic_regression.pkl",
    "Random Forest":       "saved_pipeline_M2/best_random_forest.pkl",
    "SVM RBF":             "saved_pipeline_M2/best_svm_rbf.pkl",
    "SVM Linear":          "saved_pipeline_M2/best_svm_linear.pkl",
    "XGBoost":             "saved_pipeline_M2/best_xgboost.pkl",
    "LightGBM":            "saved_pipeline_M2/best_lightgbm.pkl",
    "CatBoost":            "saved_pipeline_M2/best_catboost.pkl",
    "AdaBoost":            "saved_pipeline_M2/best_adaboost.pkl",
}

models = {}

for model_name, model_path in model_paths.items():

    models[model_name] = load_pickle(model_path)

    print(f"Loaded: {model_name}")

# ════════════════════════════════════════════════════════
# 4. DROP HIGH-NULL & ID COLUMNS (FIX 1)
# ════════════════════════════════════════════════════════

df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

print(f"Shape after dropping high-null/ID columns: {df.shape}")


# ════════════════════════════════════════════════════════
# 5. IMPUTATION
# ════════════════════════════════════════════════════════

# Filter numerical to only columns present in df
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

# for col in skewed_cols:
#     if col in df.columns:
#         df[col] = np.log1p(df[col])

# print("Log transform completed.")


# ════════════════════════════════════════════════════════
# 8. BINARY ENCODING
# ════════════════════════════════════════════════════════

for col in df.select_dtypes(include='object').columns:
    unique_vals = set(df[col].dropna().unique())
    if unique_vals.issubset(set(binary_map.keys())):
        df[col] = df[col].map(binary_map)

print("Binary encoding completed.")


# ════════════════════════════════════════════════════════
# 9. MULTI-VALUE (MLB) ENCODING (FIX 2)
# ════════════════════════════════════════════════════════

for col, mlb in mlb_encoders.items():
    if col in df.columns:
        top_values = top_values_map.get(col, set())

        # Split semicolon-separated strings
        df[col] = df[col].fillna("").apply(lambda x: x.split(";") if x != "" else [])

        # Replace unseen values with UNKNOWN
        df[col] = df[col].apply(lambda x: [i if i in top_values else "UNKNOWN" for i in x])

        # Transform using the fitted MLB
        encoded = mlb.transform(df[col])
        cols = [f"{col}_{val}" for val in mlb.classes_]

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

low_card_cols = [
    col for col in df.select_dtypes(include='object').columns
]
df = pd.get_dummies(df, columns=low_card_cols, drop_first=True, dtype=int)

# Align to training columns
df = df.reindex(columns=ohe_columns, fill_value=0)

print("One-hot encoding completed.")


# ════════════════════════════════════════════════════════
# 12. SANITIZE COLUMN NAMES
# ════════════════════════════════════════════════════════

df.columns = [re.sub(r'[^A-Za-z0-9_]', '_', col) for col in df.columns]

print("Column name sanitization completed.")


# ════════════════════════════════════════════════════════
# 13. FEATURE SELECTION
# ════════════════════════════════════════════════════════

# Step 1: Remove correlated columns
df = df.drop(columns=[c for c in to_drop_corr if c in df.columns])

# Step 2: Align to variance selector input, then apply
df = df.reindex(columns=selector_input_cols, fill_value=0)

selected_variance_cols = np.array(selector_input_cols)[selector.get_support()]
df = pd.DataFrame(
    selector.transform(df),
    columns=selected_variance_cols,
    index=df.index
)

# Step 3: MI selected features
df = df.reindex(columns=selected, fill_value=0)

# Step 4: Final top features
df = df.reindex(columns=top_features, fill_value=0)

print(f"Final feature shape: {df.shape}")

# ════════════════════════════════════════════════════════
# 14. ENCODE y_true (WITH NaN FIX)
# ════════════════════════════════════════════════════════

reverse_map = {
    "Low": 0,
    "Average": 1,
    "High": 2
}

y_true_encoded = None
valid_idx = df.index  # default: use all rows

if y_true is not None:

    # Normalize casing and whitespace before mapping
    y_true = y_true.astype(str).str.strip().str.capitalize()

    y_true_encoded = y_true.map(reverse_map)

    # Check for unrecognized labels that became NaN
    nan_mask = y_true_encoded.isna()

    if nan_mask.any():
        print(f"\nWarning: {nan_mask.sum()} rows have unrecognized JobSatLevel values:")
        print(y_true[nan_mask].value_counts())
        print("These rows will be excluded from evaluation.\n")

        # Keep only valid rows for evaluation
        valid_idx = y_true_encoded[~nan_mask].index
        y_true_encoded = y_true_encoded[~nan_mask]
        df_eval = df.loc[valid_idx]

        print(f"Proceeding with {len(y_true_encoded)} valid rows out of {len(y_true)} total.")

    else:
        df_eval = df
        print("All JobSatLevel labels recognized successfully.")


# ════════════════════════════════════════════════════════
# 15. RUN ALL MODELS
# ════════════════════════════════════════════════════════

results = []

print("\n================ MODEL RESULTS ================\n")

for model_name, model in models.items():

    # Use df_eval (rows with valid labels) if evaluating, else full df
    input_df = df_eval if y_true_encoded is not None else df

    preds = model.predict(input_df)

    # Fix CatBoost output shape
    preds = np.array(preds).flatten().astype(int)

    pred_labels = [label_map[p] for p in preds]

    if y_true_encoded is not None:

        acc = accuracy_score(y_true_encoded, preds)

        results.append({
            "Model": model_name,
            "Accuracy": acc
        })

        print(f"{model_name:25s} Accuracy = {acc:.4f}")

        print("\nClassification Report:")
        print(classification_report(y_true_encoded, preds))

        print("\nConfusion Matrix:")
        print(confusion_matrix(y_true_encoded, preds))

        print("\n" + "=" * 60 + "\n")

# ════════════════════════════════════════════════════════
# 16. SAVE SUMMARY
# ════════════════════════════════════════════════════════

if len(results) > 0:

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        by="Accuracy",
        ascending=False
    )

    results_df.to_csv(
        "all_model_accuracies.csv",
        index=False
    )

    print("\n================ FINAL ACCURACIES ================\n")
    print(results_df)

    print("\nSaved: all_model_accuracies.csv")

print("\nDone.")
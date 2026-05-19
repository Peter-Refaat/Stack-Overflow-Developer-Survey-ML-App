import pandas as pd
import numpy as np
import pickle
import re
import os
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
try:
    import xgboost
except ImportError:
    pass
try:    
    import lightgbm
except ImportError:
    pass
try:
    import catboost
except ImportError:
    pass

class BasePredictor:
    def __init__(self, pipeline_dir):
        self.pipeline_dir = pipeline_dir
        self.artifacts = {}
        self.models = {}
        self._load_artifacts()

    def _load_pickle(self, filename):
        path = os.path.join(self.pipeline_dir, filename)
        if not os.path.exists(path):
            return None
        with open(path, "rb") as f:
            return pickle.load(f)

    def _load_artifacts(self):
        raise NotImplementedError

class Milestone1Predictor(BasePredictor):
    def _load_artifacts(self):
        # Preprocessing artifacts
        self.artifacts['rf_imputer']          = self._load_pickle("rf_imputer.pkl")
        self.artifacts['numerical']           = self._load_pickle("numerical_cols.pkl")
        self.artifacts['iqr_bounds']          = self._load_pickle("iqr_bounds.pkl")
        self.artifacts['skewed_cols']         = self._load_pickle("skewed_cols.pkl")
        self.artifacts['binary_map']          = self._load_pickle("binary_map.pkl")
        self.artifacts['cols_to_drop']        = self._load_pickle("cols_to_drop.pkl")
        self.artifacts['ohe_columns']         = self._load_pickle("ohe_columns.pkl")
        self.artifacts['freq_maps']           = self._load_pickle("freq_maps.pkl")
        self.artifacts['mlb_encoders']        = self._load_pickle("mlb_encoders.pkl")
        self.artifacts['corr_drop_cols']      = self._load_pickle("corr_drop_cols.pkl")
        self.artifacts['variance_selector']   = self._load_pickle("variance_selector.pkl")
        self.artifacts['selector_input_cols'] = self._load_pickle("selector_input_cols.pkl")
        self.artifacts['top_features']        = self._load_pickle("top_features.pkl")

        # Models
        model_paths = {
            "Linear Regression":      "linear_regression.pkl",
            "Random Forest":          "random_forest.pkl",
            "Gradient Boosting":      "gradient_boosting.pkl",
            "Hist Gradient Boosting": "hist_gradient_boosting.pkl",
            "Voting Regressor":       "voting_regressor.pkl",
        }
        for name, path in model_paths.items():
            try:
                model = self._load_pickle(path)
                if model:
                    self.models[name] = model
            except Exception as e:
                print(f"Error loading model {name}: {e}")

    def predict(self, df_input):
        df = df_input.copy()
        y_true = None
        if "JobSat" in df.columns:
            df = df.dropna(subset=["JobSat"])
            y_true = df["JobSat"].copy()
            df = df.drop(columns=["JobSat"])
        
        if "ResponseId" in df.columns:
            df = df.drop(columns=["ResponseId"])

        # 4. Drop high-null
        cols_to_drop = self.artifacts['cols_to_drop']
        df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

        # 5. Imputation
        numerical = self.artifacts['numerical']
        rf_imputer = self.artifacts['rf_imputer']
        numerical_present = [c for c in numerical if c in df.columns]
        df[numerical_present] = df[numerical_present].replace([np.inf, -np.inf], np.nan)
        df[numerical_present] = df[numerical_present].clip(lower=-1e12, upper=1e12)
        df[numerical_present] = rf_imputer.transform(df[numerical_present].astype(np.float64))

        categorical_present = df.select_dtypes(include='object').columns
        df[categorical_present] = df[categorical_present].fillna("Unknown")

        # 6. Outlier Capping
        iqr_bounds = self.artifacts['iqr_bounds']
        for col, (lower, upper) in iqr_bounds.items():
            if col in df.columns:
                df[col] = df[col].clip(lower, upper)

        # 7. Log Transform
        skewed_cols = self.artifacts['skewed_cols']
        for col in skewed_cols:
            if col in df.columns:
                df[col] = np.log1p(df[col])

        # 8. Binary Encoding
        binary_map = self.artifacts['binary_map']
        for col in df.select_dtypes(include='object').columns:
            unique_vals = set(df[col].dropna().unique())
            if unique_vals.issubset(set(binary_map.keys())):
                df[col] = df[col].map(binary_map)

        # 9. Multi-value Encoding
        mlb_encoders = self.artifacts['mlb_encoders']
        for col, mlb in mlb_encoders.items():
            if col in df.columns:
                df[col] = df[col].fillna("").apply(lambda x: x.split(";") if x != "" else [])
                known_classes = set(mlb.classes_)
                df[col] = df[col].apply(lambda x: [i if i in known_classes else "UNKNOWN" for i in x])
                encoded = mlb.transform(df[col])
                cols = [f"{col}_{val}" for val in mlb.classes_]
                df = pd.concat([df, pd.DataFrame(encoded, columns=cols, index=df.index).astype(int)], axis=1).drop(columns=[col])

        # 10. Frequency Encoding
        freq_maps = self.artifacts['freq_maps']
        for col, freq in freq_maps.items():
            if col in df.columns:
                df[col] = df[col].map(freq).fillna(0)

        # 11. One-Hot Encoding
        low_card_cols = list(df.select_dtypes(include='str').columns)
        df = pd.get_dummies(df, columns=low_card_cols, drop_first=True, dtype=int)

        # 12. Sanitize and Align
        df.columns = [re.sub(r'[^A-Za-z0-9_]', '_', col) for col in df.columns]
        ohe_columns = self.artifacts['ohe_columns']
        df = df.reindex(columns=ohe_columns, fill_value=0)

        # 13. Feature Selection
        to_drop_corr = self.artifacts['corr_drop_cols']
        df = df.drop(columns=[c for c in to_drop_corr if c in df.columns])
        
        selector_input_cols = self.artifacts['selector_input_cols']
        df = df.reindex(columns=selector_input_cols, fill_value=0)
        
        selector = self.artifacts['variance_selector']
        selected_variance_cols = np.array(selector_input_cols)[selector.get_support()]
        df = pd.DataFrame(selector.transform(df), columns=selected_variance_cols, index=df.index)
        
        top_features = self.artifacts['top_features']
        df = df.reindex(columns=top_features, fill_value=0)

        results = []
        for name, model in self.models.items():
            try:
                preds = model.predict(df)
                preds = np.array(preds).flatten()
                
                res = {"Model": name}
                if y_true is not None:
                    mse = mean_squared_error(y_true, preds)
                    r2 = r2_score(y_true, preds)
                    res.update({"MSE": float(mse), "RMSE": float(mse**0.5), "R2": float(r2)})
                results.append(res)
            except Exception as e:
                print(f"Error predicting with {name}: {e}")
        
        return results

class Milestone2Predictor(BasePredictor):
    def _load_artifacts(self):
        # Preprocessing artifacts
        self.artifacts['rf_imputer']          = self._load_pickle("rf_imputer.pkl")
        self.artifacts['numerical']           = self._load_pickle("numerical_cols.pkl")
        self.artifacts['iqr_bounds']          = self._load_pickle("iqr_bounds.pkl")
        self.artifacts['skewed_cols']         = self._load_pickle("skewed_cols.pkl")
        self.artifacts['binary_map']          = self._load_pickle("binary_map.pkl")
        self.artifacts['cols_to_drop']        = self._load_pickle("cols_to_drop.pkl")
        self.artifacts['ohe_columns']         = self._load_pickle("ohe_columns.pkl")
        self.artifacts['freq_maps']           = self._load_pickle("freq_maps.pkl")
        self.artifacts['mlb_encoders']        = self._load_pickle("mlb_encoders.pkl")
        self.artifacts['top_values_map']      = self._load_pickle("top_values_map.pkl")
        self.artifacts['corr_drop_cols']      = self._load_pickle("corr_drop_cols.pkl")
        self.artifacts['variance_selector']   = self._load_pickle("variance_selector.pkl")
        self.artifacts['mi_selected_features'] = self._load_pickle("mi_selected_features.pkl")
        self.artifacts['top_features']        = self._load_pickle("top_features.pkl")
        self.artifacts['label_map']           = self._load_pickle("label_map.pkl")
        self.artifacts['selector_input_cols'] = self._load_pickle("selector_input_cols.pkl")

        # Models
        model_paths = {
            "Logistic Regression": "best_logistic_regression.pkl",
            "Random Forest":       "best_random_forest.pkl",
            "SVM RBF":             "best_svm_rbf.pkl",
            "SVM Linear":          "best_svm_linear.pkl",
            "XGBoost":             "best_xgboost.pkl",
            "LightGBM":            "best_lightgbm.pkl",
            "CatBoost":            "best_catboost.pkl",
            "AdaBoost":            "best_adaboost.pkl",
        }
        for name, path in model_paths.items():
            try:
                model = self._load_pickle(path)
                if model:
                    self.models[name] = model
            except Exception as e:
                print(f"Error loading model {name}: {e}")

    def predict(self, df_input):
        df = df_input.copy()
        y_true = None
        if "JobSatLevel" in df.columns:
            y_true = df["JobSatLevel"].copy()
            df = df.drop(columns=["JobSatLevel"])

        # 4. Drop high-null
        cols_to_drop = self.artifacts['cols_to_drop']
        df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

        # 5. Imputation
        numerical = self.artifacts['numerical']
        rf_imputer = self.artifacts['rf_imputer']
        numerical_present = [c for c in numerical if c in df.columns]
        df[numerical_present] = df[numerical_present].replace([np.inf, -np.inf], np.nan)
        df[numerical_present] = df[numerical_present].clip(lower=-1e12, upper=1e12)
        df[numerical_present] = rf_imputer.transform(df[numerical_present].astype(np.float64))

        categorical_present = df.select_dtypes(include='object').columns
        df[categorical_present] = df[categorical_present].fillna("Unknown")

        # 6. Outlier Capping
        iqr_bounds = self.artifacts['iqr_bounds']
        for col, (lower, upper) in iqr_bounds.items():
            if col in df.columns:
                df[col] = df[col].clip(lower, upper)

        # 7. Log Transform
        skewed_cols = self.artifacts['skewed_cols']
        for col in skewed_cols:
            if col in df.columns:
                df[col] = np.log1p(df[col])

        # 8. Binary Encoding
        binary_map = self.artifacts['binary_map']
        for col in df.select_dtypes(include='object').columns:
            unique_vals = set(df[col].dropna().unique())
            if unique_vals.issubset(set(binary_map.keys())):
                df[col] = df[col].map(binary_map)

        # 9. Multi-value Encoding
        mlb_encoders = self.artifacts['mlb_encoders']
        top_values_map = self.artifacts['top_values_map']
        for col, mlb in mlb_encoders.items():
            if col in df.columns:
                top_values = top_values_map.get(col, set())
                df[col] = df[col].fillna("").apply(lambda x: x.split(";") if x != "" else [])
                df[col] = df[col].apply(lambda x: [i if i in top_values else "UNKNOWN" for i in x])
                encoded = mlb.transform(df[col])
                cols = [f"{col}_{val}" for val in mlb.classes_]
                df = pd.concat([df, pd.DataFrame(encoded, columns=cols, index=df.index).astype(int)], axis=1).drop(columns=[col])

        # 10. Frequency Encoding
        freq_maps = self.artifacts['freq_maps']
        for col, freq in freq_maps.items():
            if col in df.columns:
                df[col] = df[col].map(freq).fillna(0)

        # 11. One-Hot Encoding
        low_card_cols = [col for col in df.select_dtypes(include='object').columns]
        df = pd.get_dummies(df, columns=low_card_cols, drop_first=True, dtype=int)
        ohe_columns = self.artifacts['ohe_columns']
        df = df.reindex(columns=ohe_columns, fill_value=0)

        # 12. Sanitize
        df.columns = [re.sub(r'[^A-Za-z0-9_]', '_', col) for col in df.columns]

        # 13. Feature Selection
        to_drop_corr = self.artifacts['corr_drop_cols']
        df = df.drop(columns=[c for c in to_drop_corr if c in df.columns])
        
        selector_input_cols = self.artifacts['selector_input_cols']
        df = df.reindex(columns=selector_input_cols, fill_value=0)
        
        selector = self.artifacts['variance_selector']
        selected_variance_cols = np.array(selector_input_cols)[selector.get_support()]
        df = pd.DataFrame(selector.transform(df), columns=selected_variance_cols, index=df.index)
        
        mi_selected = self.artifacts['mi_selected_features']
        df = df.reindex(columns=mi_selected, fill_value=0)
        
        top_features = self.artifacts['top_features']
        df = df.reindex(columns=top_features, fill_value=0)

        reverse_map = {"Low": 0, "Average": 1, "High": 2}
        y_true_encoded = None
        if y_true is not None:
            y_true_encoded = y_true.map(reverse_map)

        results = []
        for name, model in self.models.items():
            try:
                preds = model.predict(df)
                # Fix CatBoost output shape
                preds = np.array(preds).flatten().astype(int)
                
                res = {"Model": name}
                if y_true_encoded is not None:
                    acc = accuracy_score(y_true_encoded, preds)
                    res.update({"Accuracy": float(acc)})
                results.append(res)
            except Exception as e:
                print(f"Error predicting with {name}: {e}")
        
        return results

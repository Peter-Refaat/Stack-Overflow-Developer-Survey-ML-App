# DevSat-ML: Developer Job Satisfaction Prediction Pipeline

An end-to-end machine learning project that analyzes and models software developer job satisfaction using real-world data. This project explores the predictive power of career trajectories, compensation, tooling, and AI sentiments on a developer's self-reported workplace satisfaction.
---

## 📌 Project Overview
This project was developed across two distinct phases to predict developer job satisfaction (`JobSat`) using the comprehensive **2025 Stack Overflow Developer Survey** dataset (44,191 rows, 156 columns).

* **Milestone 1 (Regression Focus):** Modeled `JobSat` as a continuous numeric score ranging from `0` (Not at all satisfied) to `10` (Extremely satisfied).
* **Milestone 2 (Classification & Robustness Focus):** Reframed the target into a 3-class classification problem (Low: 0, Average: 1, High: 2). Introduced strict pipeline fixes to eliminate data leakage and utilized SMOTE to handle class imbalance.

---

## 🛠️ Preprocessing & Data Pipeline
The data engineering process follows a rigorous 13-step pipeline designed to extract strong signals while preserving data integrity:

1. **Target Filtering:** Removed rows missing the target variable (leaving 23,961 active developer records).
2. **Anti-Leakage Split (Fix):** The train/test split (80/20) is executed *immediately* after initial row/column filtering. All downstream imputers, scalers, and encoders are fit *only* on the training set to prevent test-set leakage.
3. **Advanced Imputation:** * *Numerical:* Iterative Imputer with a Random Forest base estimator (captures complex feature relationships).
   * *Categorical:* Handled missing non-responses explicitly by imputing an `'Unknown'` category string.
4. **Outlier & Skew Management:** High-cardinality features are capped via the IQR method, and heavily skewed distributions (e.g., `ConvertedCompYearly`, `YearsCode`) are normalized using a `log1p` transformation.
5. **Categorical Encoding Funnel:**
   * **Binary mapping (1/0)** for pure boolean flags.
   * **MultiLabelBinarizer (MLB)** for semicolon-separated multi-select survey questions (retaining the top 15 choices).
   * **One-Hot Encoding** for nominal/ordinal attributes under a cardinality threshold of 15.
   * **Frequency Encoding** for high-cardinality nominal features (e.g., `DevType`, `Industry`).

---

## 🧬 Feature Selection
To prevent feature explosion and control dimensionality, the project utilizes a strict feature selection pipeline:
* **Regression (M1):** Reduced the dataset to the **top 100 features** using Pearson correlation and Random Forest importance.
* **Classification (M2):** Replaced linear correlation with **Mutual Information (MI)** to capture non-linear relationships with the discrete classes, distilling the data down to the **top 15 features** to combat overfitting.

### Key Predictors Identified:
1. Career change reflections (`NewRole` intentions)
2. Total annual compensation (`ConvertedCompYearly`)
3. Workplace environment complexity (`ToolCountWork`)
4. Professional experience (`YearsCode` / `WorkExp`)

---

## 📊 Modeling & Performance Summary

### Phase 1: Continuous Regression (0-10 Scale)
Evaluated across 5-fold cross-validation and a holdout test set. 

| Model | MSE | RMSE | $R^2$ |
| :--- | :---: | :---: | :---: |
| **Voting Regressor (Ensemble)** | **3.1327** | **1.7699** | **0.2029** |
| Hist Gradient Boosting | 3.1557 | 1.7764 | 0.1970 |
| Gradient Boosting | 3.1600 | 1.7776 | 0.1959 |
| Linear Regression | 3.1693 | 1.7802 | 0.1936 |
| Random Forest | 3.2003 | 1.7889 | 0.1857 |

*Note: A baseline experiment attempting to classify all 11 discrete regression scores using an RBF Support Vector Classifier failed, as it collapsed to predicting the modal class (Score 8), confirming regression as the correct formulation for the continuous target.*

### Phase 2: 3-Class Classification (Low, Avg, High)
Models were trained on SMOTE-balanced training partitions and hyperparameter-tuned via `GridSearchCV` (3-fold CV).

| Model | Test Accuracy | Train Time | Inference Time |
| :--- | :---: | :---: | :---: |
| **LightGBM** | **49.55%** | 1.46s | 17.2ms |
| RBF SVM | 48.88% | 27.61s | 4455.3ms |
| Random Forest | 47.21% | 0.70s | 58.5ms |
| CatBoost | 46.15% | 3.17s | 3.5ms |
| AdaBoost | 45.23% | 1.49s | 36.8ms |
| XGBoost | 42.85% | 1.15s | 9.0ms |
| Logistic Regression | 40.45% | 0.11s | 1.3ms |
| Linear SVM | 36.57% | 11.5s | 1215.6ms |

---

## 🚀 How to Run the Project

You do not need to manually configure environment variables or run multiple terminal commands to start the pipeline. The entire execution workflow has been automated.

### Prerequisites
* Ensure you have Python 3 installed on your system.
* Ensure Git is installed if cloning the repository.

### Launching the Pipeline
1. Clone or download this repository to your local machine.
2. Navigate to the root directory of the project.
3. Double-click the **`run.bat`** file.

**What the batch file does automatically:**
* Creates and activates a local Python virtual environment (`.venv`).
* Automatically upgrades `pip` and installs all necessary dependencies listed in `requirements.txt`.
* Executes the complete preprocessing, feature engineering, and model training pipelines.
* Outputs the evaluation metrics and prints results directly to your terminal window.

---

## 📈 Key Conclusions & Takeaways
* **The Signal Ceiling:** All classification models hovered within a tight accuracy band of 40%–49% (meaningfully above the 33.3% random-chance baseline). This narrow performance cluster indicates a data-signal boundary rather than a model optimization problem.
* **Subjective Challenges:** Predicting job satisfaction is inherently difficult using structured survey data alone. Key driving factors such as individual team dynamics, management quality, and mental well-being are not captured in standard survey fields.

# churn_analysis
subscription based service churn analysis, specifically telecom provider
# 📊 Telco Customer Churn: Advanced Predictive Analytics & Survival Modeling

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Machine Learning](https://img.shields.io/badge/Machine_Learning-FF6F00?style=for-the-badge&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-000000?style=for-the-badge&logo=xgboost&logoColor=white)

## 📌 Project Overview
This project provides a comprehensive data science workflow to analyze and predict customer churn. It integrates traditional machine learning with **Survival Analysis** and **Financial Risk Modeling** to provide actionable business insights.

## 🛠️ Tech Stack & Libraries
* **Core:** `pandas`, `numpy` 🔢
* **Visualization:** `seaborn`, `matplotlib`, `IPython.display` 📈
* **Machine Learning:** `scikit-learn` (Pipelines, ColumnTransformers, Calibration), `XGBoost` 🤖
* **Optimization:** `optuna` (Bayesian Hyperparameter Tuning) 🧪
* **Survival Analysis:** `lifelines` (Kaplan-Meier, Log-Rank testing) ⏳

## 🚀 Key Methodologies

### 1. Robust Engineering & Pipelines ⚙️
* **Feature Engineering:** Automated preprocessing using `ColumnTransformer` for handling `OneHotEncoder` (categorical) and `KBinsDiscretizer` (continuous) variables.
* **Data Cleaning:** Systematic handling of data types (e.g., `TotalCharges` conversion) and missing values.

### 2. Survival Analysis ⏳
We analyze churn as a time-based event:
* **Kaplan-Meier Estimator:** Estimating the survival function to see the probability of retention over time.
* **Log-Rank Test:** Statistical comparison of survival curves across segments (e.g., comparing "Month-to-month" vs "Two-year" contracts).

### 3. Model Development & Calibration 🏆
* **Algorithm Benchmarking:** Evaluated `LogisticRegression` against `XGBClassifier`.
* **Model Calibration:** Used `CalibratedClassifierCV` so that predicted probabilities align with actual churn frequencies, which is vital for financial risk assessment.
* **Hyperparameter Tuning:** Implemented `Optuna` for efficient, automated search of the best model parameters.

### 4. Advanced Risk & Bias Mitigation 🛡️
* **Selection Bias Filtering:** Applied `NearestNeighbors` to identify and mitigate sampling bias.
* **Value at Risk (VaR):** Established an optimal classification threshold based on the business "appetite for risk," balancing revenue loss vs. retention costs.
* **Fairness & Ethics:** Conducted audits to ensure the model does not discriminate against protected groups, such as senior citizens.
* **Drift Detection:** Set distribution thresholds to monitor model stability and performance decay over time.

## 📊 Business Application
The model segments customers into:
* 🟢 **Low-Risk:** High survival probability; focus on upsell.
* 🟡 **Monitor:** Early signs of churn; focus on engagement.
* 🔴 **High-Risk:** Immediate intervention required based on VaR analysis.

---
*Developed with a focus on statistical objectivity and business impact.* 🚀
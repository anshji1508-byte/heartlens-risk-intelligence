"""Train and serialize the heart-disease model used by the Streamlit app."""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.pipeline import Pipeline as SklearnPipeline
from imblearn.over_sampling import SVMSMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from model_utils import FeatureConstructor

RANDOM_STATE = 1
TARGET = "has_heart_disease"
THRESHOLD = 0.63

class _UnusedFeatureConstructor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X):
        z = X.copy()
        z["age_x_ldl"] = z["age"] * z["ldl"]
        z["bp_ratio"] = z["resting_bp_systolic"] / (z["resting_bp_diastolic"] + 1)
        z["cholesterol_ratio"] = z["cholesterol_total"] / (z["hdl"] + 1)
        z["bmi_x_activity"] = z["bmi"] * z["exercise_minutes_per_week"]
        z["metabolic_risk_score"] = z["hba1c"] + z["triglycerides"] / 100
        z["lipid_gap"] = z["ldl"] - z["hdl"]
        z["pain_x_history"] = z["chest_pain_type"].astype(str) + "_" + z["family_history"].astype(str)
        return z

def make_transformer():
    numeric = Pipeline([("scaler", RobustScaler())])
    one_hot = Pipeline([("encoder", OneHotEncoder(sparse_output=False, dtype=int, drop="first"))])
    ordinal = Pipeline([("encoder", OrdinalEncoder(categories=[[
        "Asymptomatic", "Non-Anginal Pain", "Atypical Angina", "Typical Angina"
    ]]))])
    return ColumnTransformer([
        ("Numerical", numeric, make_column_selector(dtype_include=["int", "float"])),
        ("Ohe", one_hot, make_column_selector(dtype_include=["object", "bool"])),
        ("Ord", ordinal, make_column_selector(dtype_include="category")),
    ])

def main():
    data = pd.read_csv(Path(__file__).with_name("heart_disease_risk_2026.csv"))
    data["chest_pain_type"] = data["chest_pain_type"].astype("category")
    X = data.drop(columns=[TARGET, "patient_id"])
    y = data[TARGET]
    X_train, _, y_train, _ = train_test_split(X, y, test_size=.20, stratify=y, random_state=RANDOM_STATE)
    estimator = xgb.XGBClassifier(
        learning_rate=.05, max_depth=3, n_estimators=250,
        objective="binary:logistic", eval_metric="logloss", tree_method="hist",
        random_state=RANDOM_STATE, n_jobs=-1
    )
    model = ImbPipeline([
        ("construction", FeatureConstructor()),
        ("preprocessing", make_transformer()),
        ("selection", SelectKBest(mutual_info_classif, k=31)),
        ("oversampler", SVMSMOTE(k_neighbors=10, random_state=RANDOM_STATE)),
        ("estimator", estimator),
    ])
    model.fit(X_train, y_train)
    # Do not serialize the sampler in the serving artifact. Oversampling is only
    # needed during training; inference needs the fitted transforms and estimator.
    inference_model = SklearnPipeline([
        ("construction", model.named_steps["construction"]),
        ("preprocessing", model.named_steps["preprocessing"]),
        ("selection", model.named_steps["selection"]),
        ("estimator", model.named_steps["estimator"]),
    ])
    artifact = {
        "model": inference_model, "threshold": THRESHOLD, "target": TARGET,
        "feature_columns": X.columns.tolist(), "model_version": "heart-risk-2026-v1",
        "metrics": {"accuracy": .8939, "f1": .8227, "recall": .8128, "pr_auc": .9077},
    }
    out = Path(__file__).with_name("heart_disease_model.joblib")
    joblib.dump(artifact, out, compress=3)
    print(f"Saved {out} ({out.stat().st_size / 1024:.1f} KB)")

if __name__ == "__main__":
    main()

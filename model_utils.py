from sklearn.base import BaseEstimator, TransformerMixin


class FeatureConstructor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

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

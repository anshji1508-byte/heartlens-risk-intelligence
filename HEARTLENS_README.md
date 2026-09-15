# HeartLens

An explainable Streamlit website for the heart-disease risk model in this project.

## Run locally

1. `python train_model.py`
2. `streamlit run heartlens_app.py`

The training script creates `heart_disease_model.joblib`. The app loads that artifact and uses the same feature construction, preprocessing, statistical selection, SMOTE strategy, XGBoost configuration, and operating threshold as the final notebook model.

This is a decision-support prototype, not a medical diagnosis.

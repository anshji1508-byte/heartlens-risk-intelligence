# HeartLens — Explainable Heart-Disease Risk Intelligence

HeartLens is an interactive Streamlit dashboard that estimates heart-disease risk and explains each prediction using SHAP. It is designed to support screening prioritization, preventive-care outreach, and model transparency.

## Features

- Modern light and night mode dashboard
- Patient risk probability with tuned operating threshold
- XGBoost classification model serialized with Joblib
- Statistical feature selection using mutual information and ANOVA/chi-square analysis
- Domain feature construction for lipid, blood-pressure, metabolic, and lifestyle signals
- SHAP-based local explanations and actionable risk contributors
- Before/after model evaluation with Accuracy, F1-score, Recall, and PR-AUC
- Care-team, population-program, and governance guidance
- Privacy-aware medical decision-support disclaimer

## Model performance

Final holdout test results:

| Metric | Score |
|---|---:|
| Accuracy | 0.8939 |
| F1-score | 0.8227 |
| Recall | 0.8128 |
| PR-AUC | 0.9077 |

The threshold selected from training out-of-fold predictions is `0.63`.

## Run locally

```bash
python -m pip install -r requirements_heartlens.txt
python -m streamlit run heartlens_app.py
```

The app loads `heart_disease_model.joblib`. To retrain the model, run:

```bash
python train_model.py
```

## Project files

- `heartlens_app.py` — Streamlit dashboard
- `heart_disease_model.joblib` — serialized serving model
- `train_model.py` — reproducible training and serialization script
- `model_utils.py` — shared feature-construction transformer
- `35.ipynb` — complete analysis, tuning, selection, thresholding, SHAP, and evaluation workflow
- `heart_disease_risk_2026.csv` — supplied dataset

## Important disclaimer

HeartLens is a machine-learning decision-support prototype, not a medical diagnosis. Predictions must be reviewed by qualified healthcare professionals and validated on the intended population before any production or clinical use.

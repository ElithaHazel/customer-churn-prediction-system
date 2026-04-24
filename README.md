# Customer Churn Prediction System

An end-to-end, production-grade machine learning system that predicts which customers are likely to cancel their service, quantifies the revenue at risk, and exposes predictions through an interactive Streamlit app.

---

## 1. The business problem (the "why")

A telecom company loses roughly 1.5% of its customers every month. Acquiring a new customer costs 5–7x more than retaining an existing one. If we can flag the customers most likely to leave *before* they cancel, the retention team can target them with offers, saving millions in lost revenue.

**This project answers three questions a real business would ask:**

1. *Who* is most likely to churn in the next 30 days?
2. *Why* are they likely to churn (what drivers matter)?
3. *What* is it worth to us to prevent it (revenue at risk, ROI of retention offer)?

That framing is what makes this resume-worthy. A model that just outputs `P(churn) = 0.73` is a homework assignment. A system that says "these 1,200 customers represent $340K of at-risk monthly revenue, and here are the top 3 drivers per segment" — that is a project a hiring manager cares about.

---

## 2. Tech stack and why each piece

| Layer | Tool | Why |
|---|---|---|
| Language | Python 3.11 | Industry default for ML |
| Data | pandas, numpy | Standard for tabular manipulation |
| Viz | matplotlib, seaborn, plotly | Static for reports, interactive for app |
| Modeling | scikit-learn, XGBoost, LightGBM | scikit-learn for baselines, gradient boosting for the real model |
| Imbalance | imbalanced-learn (SMOTE, class weights) | Churn is always imbalanced (~25% positive) |
| Explainability | SHAP | Interviewers *will* ask "how do you explain a black-box model" |
| Experiment tracking | MLflow | Tracks every experiment — shows you think like an ML engineer, not a student |
| Packaging | pip + requirements.txt, later Docker | Reproducibility |
| App | Streamlit | Fast to build, easy to deploy, good enough to impress |
| Deployment | Streamlit Community Cloud (free) | Live URL on your resume > a GitHub repo alone |
| Version control | Git + GitHub | Commit history tells the story of the work |
| Testing | pytest | Even a few tests signals production mindset |
| CI | GitHub Actions | Optional but strong signal |

---

## 3. The 10-phase roadmap

Each phase has a clear deliverable. Do not skip phases — in interviews you will be asked about each one.

### Phase 1 — Setup & data acquisition *(Day 1)*
- Create project structure (done)
- Set up virtual environment, install dependencies
- Download the Telco Customer Churn dataset (IBM/Kaggle)
- Load and sanity-check the data

### Phase 2 — Exploratory Data Analysis (EDA) *(Days 2–3)*
- Univariate, bivariate, multivariate exploration
- Target leakage check
- Class imbalance assessment
- Hypothesis-driven analysis ("Do month-to-month customers churn more? By how much?")
- **Deliverable:** `notebooks/01_eda.ipynb` + written findings

### Phase 3 — Data cleaning *(Day 3)*
- Handle missing values, whitespace nulls, dtype issues (`TotalCharges` is notoriously a string)
- Consistent category labels
- Save a clean dataset to `data/processed/`

### Phase 4 — Feature engineering *(Days 4–5)*
- Tenure buckets, contract-risk interactions, service-count features
- Target encoding where appropriate
- One-hot vs ordinal decisions
- Build a `ColumnTransformer` pipeline — no leakage, fully reproducible

### Phase 5 — Baseline modeling *(Day 6)*
- Logistic Regression first — always. It sets the floor and gives you interpretable coefficients
- Decision Tree, Random Forest
- Use stratified k-fold CV
- Track every run with MLflow

### Phase 6 — Advanced modeling *(Days 7–8)*
- XGBoost, LightGBM with early stopping
- Hyperparameter tuning with Optuna (or RandomizedSearchCV to start)
- Handle imbalance: `scale_pos_weight`, SMOTE — compare

### Phase 7 — Evaluation that matters *(Day 9)*
- Accuracy is a trap here. Use ROC-AUC, PR-AUC, F1, recall at fixed precision
- Confusion matrix at the business-optimal threshold (not 0.5)
- **Cost-sensitive threshold:** pick the threshold that maximizes expected retained revenue
- Calibration plot — is a score of 0.8 really an 80% chance?

### Phase 8 — Explainability *(Day 10)*
- Global feature importance (SHAP summary plot)
- Local explanations (per-customer SHAP)
- Partial dependence plots

### Phase 9 — Streamlit app *(Days 11–12)*
- Page 1: batch upload — upload a CSV of customers, get back scored list
- Page 2: single-customer what-if — sliders that show how changing contract or tenure changes churn probability
- Page 3: business dashboard — revenue at risk, top segments, recommended actions
- Use caching, session state, proper layout

### Phase 10 — Deployment + docs *(Days 13–14)*
- Deploy to Streamlit Community Cloud
- Write README with problem framing, approach, results, live demo link
- Record a 60-second Loom walkthrough
- Rewrite your resume bullets based on actual numbers

---

## 4. Folder structure

```
Churn Project/
├── data/
│   ├── raw/            <- original dataset, never edited
│   └── processed/      <- cleaned/feature-engineered outputs
├── notebooks/          <- numbered, one per phase (01_eda.ipynb, 02_cleaning.ipynb, ...)
├── src/
│   ├── data/           <- loading, cleaning modules
│   ├── features/       <- feature engineering pipeline
│   ├── models/         <- train.py, predict.py
│   ├── evaluation/     <- metrics, threshold tuning
│   └── utils/          <- helpers, logging
├── app/                <- Streamlit app (streamlit_app.py)
├── models_store/       <- serialized models (.pkl, .joblib)
├── reports/
│   └── figures/        <- saved plots for README/presentation
├── tests/              <- pytest unit tests
├── config/             <- config.yaml for paths, hyperparameters
├── .github/workflows/  <- CI (later)
├── requirements.txt
└── README.md
```

**Why this structure?** Notebooks are for exploration. `src/` is for reusable code. When you land your first job, you will move code out of notebooks into modules — getting that habit now puts you ahead of 80% of bootcamp grads.

---

## 5. The resume bullet we're writing toward

We will end up with something like this (numbers are placeholders — you'll fill them in with real results):

> **Customer Churn Prediction System** — Built an end-to-end ML system to predict telecom customer churn on 7,000+ customers, achieving ROC-AUC of 0.XX and identifying $YYK of at-risk monthly revenue. Engineered 20+ features, benchmarked 6 models (LogReg → XGBoost), tuned with Optuna, and interpreted predictions with SHAP. Deployed as a 3-page Streamlit app on Streamlit Cloud with batch scoring, what-if analysis, and a business dashboard. Tech: Python, scikit-learn, XGBoost, MLflow, SHAP, Streamlit, Docker.

Every number in that bullet has to come from real work. That is why we are not skipping phases.

# ⚔️ Tabular Arena

**Can Foundation Models Replace Your ML Pipeline?**

A head-to-head benchmark of 7 ML models across 3 tabular datasets — same data, same splits, same metrics.

## Models

| Category | Models |
|---|---|
| 🌲 Gradient Boosting | LightGBM (Default & Tuned), CatBoost (Default & Tuned) |
| 🤖 AutoML | AutoGluon |
| 🧠 Foundation Model | TabPFN |
| 🔥 Deep Learning | FT-Transformer |

## Datasets

- **Telco Customer Churn** — 7,043 rows, 19 features
- **Home Credit Default Risk** — 307,511 rows, 122 features
- **IEEE-CIS Fraud Detection** — 590,540 rows, 434 features

## Live App

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://tabular-arena.streamlit.app)

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Protocol

- 5-fold stratified cross-validation
- 80/20 train/test split (seed 42)
- Wall-clock timing + tracemalloc for memory
- Data scaling curves at 500–5,600 samples
- Optuna (50 trials) for tuned variants

## Built By

Pranav Reveendran

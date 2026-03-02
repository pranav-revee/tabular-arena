"""
Tabular Arena — Material Design 3 Dark Theme Configuration
"""

# ── Material Color Tokens ──────────────────────────────────────────────────
PRIMARY = "#BB86FC"
PRIMARY_VARIANT = "#7C4DFF"
SECONDARY = "#03DAC6"
ERROR = "#CF6679"
ON_BG = "#E8EAED"
ON_SURFACE = "#BDC1C6"
MUTED = "#9AA0A6"

# ── Model Colors ───────────────────────────────────────────────────────────
MODEL_COLORS = {
    "LightGBM (Default)":  "#03DAC6",
    "LightGBM (Tuned)":    "#00BFA5",
    "CatBoost (Default)":  "#80CBC4",
    "CatBoost (Tuned)":    "#4DB6AC",
    "XGBoost (Default)":   "#26A69A",
    "XGBoost (Tuned)":     "#009688",
    "AutoGluon":           "#BB86FC",
    "TabPFN":              "#FF7597",
    "FT-Transformer":      "#FFB74D",
}

MODEL_CATEGORIES = {
    "LightGBM (Default)":  "Gradient Boosting",
    "LightGBM (Tuned)":    "Gradient Boosting",
    "CatBoost (Default)":  "Gradient Boosting",
    "CatBoost (Tuned)":    "Gradient Boosting",
    "XGBoost (Default)":   "Gradient Boosting",
    "XGBoost (Tuned)":     "Gradient Boosting",
    "AutoGluon":           "AutoML",
    "TabPFN":              "Foundation Model",
    "FT-Transformer":      "Deep Learning",
}

CATEGORY_COLORS = {
    "Gradient Boosting": "#03DAC6",
    "AutoML":            "#BB86FC",
    "Foundation Model":  "#FF7597",
    "Deep Learning":     "#FFB74D",
}

CATEGORY_BG = {
    "Gradient Boosting": "rgba(3,218,198,0.12)",
    "AutoML":            "rgba(187,134,252,0.12)",
    "Foundation Model":  "rgba(255,117,151,0.12)",
    "Deep Learning":     "rgba(255,183,77,0.12)",
}

CATEGORY_ICONS = {
    "Gradient Boosting": "",
    "AutoML":            "",
    "Foundation Model":  "",
    "Deep Learning":     "",
}

# ── Dataset Metadata ───────────────────────────────────────────────────────
DATASET_INFO = {
    "telco_churn": {
        "name":        "Telco Customer Churn",
        "short_name":  "Churn",
        "n_samples":   7043,
        "n_features":  19,
        "target_rate": 0.265,
        "task":        "Binary Classification",
        "source":      "Kaggle (IBM Sample)",
        "icon":        "📱",
        "description": "Predict customer churn from account and service features.",
    },
    "home_credit": {
        "name":        "Home Credit Default Risk",
        "short_name":  "Credit Risk",
        "n_samples":   307511,
        "n_features":  "120+",
        "target_rate": 0.081,
        "task":        "Binary Classification",
        "source":      "Kaggle (7 tables)",
        "icon":        "🏦",
        "description": "Predict loan default risk using 7 linked tables — bureau, installments, credit cards, POS cash & more.",
    },
    "ieee_fraud": {
        "name":        "IEEE-CIS Fraud Detection",
        "short_name":  "Fraud",
        "n_samples":   590540,
        "n_features":  434,
        "target_rate": 0.035,
        "task":        "Binary Classification",
        "source":      "Kaggle Competition",
        "icon":        "🕵️",
        "description": "Identify fraudulent transactions from transaction and identity data.",
    },
}

# ── Model Filters ──────────────────────────────────────────────────────────
DEFAULT_MODELS = [
    "LightGBM (Default)",
    "CatBoost (Default)",
    "XGBoost (Default)",
    "AutoGluon",
    "TabPFN",
    "FT-Transformer",
]

TUNED_MODELS = [
    "LightGBM (Tuned)",
    "CatBoost (Tuned)",
    "XGBoost (Tuned)",
    "AutoGluon",
    "TabPFN",
    "FT-Transformer",
]

SCALING_SIZES = [500, 1000, 2000, 3500, 5600]

# ── App Metadata ───────────────────────────────────────────────────────────
APP_TITLE = "Tabular Arena"
APP_SUBTITLE = "Can Foundation Models Replace Your ML Pipeline?"
AUTHOR_NAME = "Pranav Reveendran"
GITHUB_URL = ""
LINKEDIN_URL = ""

# ── Plotly Chart Theme ─────────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, system-ui, sans-serif", color="#9AA0A6", size=12),
    title_font=dict(size=14, color="#E8EAED"),
    legend=dict(
        orientation="h",
        yanchor="bottom", y=1.05,
        xanchor="right", x=1,
        bgcolor="rgba(0,0,0,0)",
        font=dict(size=11, color="#9AA0A6"),
    ),
    xaxis=dict(
        gridcolor="rgba(255,255,255,0.06)",
        zerolinecolor="rgba(255,255,255,0.1)",
        tickfont=dict(size=11, color="#5F6368"),
        linecolor="rgba(255,255,255,0.06)",
    ),
    yaxis=dict(
        gridcolor="rgba(255,255,255,0.06)",
        zerolinecolor="rgba(255,255,255,0.1)",
        tickfont=dict(size=11, color="#5F6368"),
        linecolor="rgba(255,255,255,0.06)",
    ),
    margin=dict(l=60, r=20, t=50, b=50),
    hoverlabel=dict(
        bgcolor="#1F1F1F",
        bordercolor="rgba(255,255,255,0.1)",
        font=dict(size=12, color="#E8EAED"),
    ),
)

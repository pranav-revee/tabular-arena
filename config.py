"""
Tabular Arena — Design System & Configuration
Modern 2026 aesthetic: deep darks, vibrant accents, glass surfaces.
"""

# ── Color Palette ───────────────────────────────────────────────────────────
# Curated via Coolors — cohesive, accessible on white backgrounds.
# Primary hues: Teal (boosting), Indigo (automl), Violet (foundation), Amber (deep learning)

TEAL       = "#0d9488"     # gradient boosting
TEAL_LIGHT = "#ccfbf1"     # teal bg tint
INDIGO       = "#4f46e5"   # automl
INDIGO_LIGHT = "#e0e7ff"   # indigo bg tint
VIOLET       = "#7c3aed"   # foundation model
VIOLET_LIGHT = "#ede9fe"   # violet bg tint
AMBER       = "#d97706"    # deep learning
AMBER_LIGHT = "#fef3c7"    # amber bg tint

# Semantic
GREEN  = "#059669"
RED    = "#dc2626"
GRAY   = "#6b7280"

# ── Model Colors (consistent across all charts) ────────────────────────────
MODEL_COLORS = {
    "LightGBM (Default)":  "#00d4aa",
    "LightGBM (Tuned)":    "#00b894",
    "CatBoost (Default)":  "#00cec9",
    "CatBoost (Tuned)":    "#00b4d8",
    "AutoGluon":           "#6c5ce7",
    "TabPFN":              "#a855f7",
    "FT-Transformer":      "#f59e0b",
}

MODEL_CATEGORIES = {
    "LightGBM (Default)":  "Gradient Boosting",
    "LightGBM (Tuned)":    "Gradient Boosting",
    "CatBoost (Default)":  "Gradient Boosting",
    "CatBoost (Tuned)":    "Gradient Boosting",
    "AutoGluon":           "AutoML",
    "TabPFN":              "Foundation Model",
    "FT-Transformer":      "Deep Learning",
}

CATEGORY_COLORS = {
    "Gradient Boosting": "#00d4aa",
    "AutoML":            "#6c5ce7",
    "Foundation Model":  "#a855f7",
    "Deep Learning":     "#f59e0b",
}

CATEGORY_BG = {
    "Gradient Boosting": "rgba(0,212,170,0.15)",
    "AutoML":            "rgba(108,92,231,0.15)",
    "Foundation Model":  "rgba(168,85,247,0.15)",
    "Deep Learning":     "rgba(245,158,11,0.15)",
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
        "n_features":  122,
        "target_rate": 0.081,
        "task":        "Binary Classification",
        "source":      "Kaggle Competition",
        "icon":        "🏦",
        "description": "Predict loan default risk using alternative data sources.",
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
    "AutoGluon",
    "TabPFN",
    "FT-Transformer",
]

TUNED_MODELS = [
    "LightGBM (Tuned)",
    "CatBoost (Tuned)",
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

# ── Plotly Chart Theme (Dark) ───────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, system-ui, -apple-system, sans-serif", color="#94a3b8", size=12),
    title_font=dict(size=14, color="#f8fafc"),
    legend=dict(
        orientation="h",
        yanchor="bottom", y=1.05,
        xanchor="right", x=1,
        bgcolor="rgba(0,0,0,0)",
        font=dict(size=11, color="#94a3b8"),
    ),
    xaxis=dict(
        gridcolor="rgba(255,255,255,0.06)",
        zerolinecolor="rgba(255,255,255,0.1)",
        tickfont=dict(size=11, color="#64748b"),
        linecolor="rgba(255,255,255,0.1)",
    ),
    yaxis=dict(
        gridcolor="rgba(255,255,255,0.06)",
        zerolinecolor="rgba(255,255,255,0.1)",
        tickfont=dict(size=11, color="#64748b"),
        linecolor="rgba(255,255,255,0.1)",
    ),
    margin=dict(l=60, r=20, t=50, b=50),
    hoverlabel=dict(
        bgcolor="#1e293b",
        bordercolor="rgba(255,255,255,0.1)",
        font=dict(size=12, color="#f8fafc"),
    ),
)

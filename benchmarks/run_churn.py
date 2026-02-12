"""
Tabular Arena — Telco Customer Churn Benchmark
Trains all models on the same data/splits, collects metrics, writes results JSON.
"""

import json
import time
import tracemalloc
import platform
import os
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

# ── Paths ───────────────────────────────────────────────────────────────────
DATA_PATH = Path(__file__).parent.parent.parent / "Dataset" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
OUTPUT_PATH = Path(__file__).parent.parent / "results" / "churn_results.json"

SCALING_SIZES = [500, 1000, 2000, 3500, 5600]
RANDOM_STATE = 42
OPTUNA_TRIALS = 50
CV_FOLDS = 5


# ── Hardware info ───────────────────────────────────────────────────────────
def get_hardware_info():
    import subprocess
    cpu = platform.processor() or "unknown"
    try:
        cpu = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
        ).strip()
    except Exception:
        pass
    ram_gb = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / (1024 ** 3)
    return {
        "cpu": cpu,
        "ram_gb": round(ram_gb, 1),
        "os": f"{platform.system()} {platform.release()}",
        "python": platform.python_version(),
    }


# ── Data loading & preprocessing ────────────────────────────────────────────
def load_data():
    print(f"Loading data from {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)

    # TotalCharges has spaces for empty values
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Drop customerID — not a feature
    df = df.drop(columns=["customerID"])

    # Encode target
    df["Churn"] = (df["Churn"] == "Yes").astype(int)

    target_rate = df["Churn"].mean()
    print(f"  Rows: {len(df)}, Features: {df.shape[1] - 1}, Target rate: {target_rate:.3f}")
    return df, target_rate


def prepare_splits(df):
    """80/20 stratified split. Returns X_train, X_test, y_train, y_test."""
    X = df.drop(columns=["Churn"])
    y = df["Churn"]
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)


def encode_for_lgbm(X_train, X_test):
    """Label-encode categorical columns for LightGBM."""
    X_tr = X_train.copy()
    X_te = X_test.copy()
    encoders = {}
    for col in X_tr.select_dtypes(include="object").columns:
        le = LabelEncoder()
        X_tr[col] = le.fit_transform(X_tr[col].astype(str))
        X_te[col] = le.transform(X_te[col].astype(str))
        encoders[col] = le
    return X_tr, X_te, encoders


def encode_for_numeric(X_train, X_test):
    """Label-encode categoricals and fill NaN for models that need pure numeric input."""
    X_tr, X_te, _ = encode_for_lgbm(X_train, X_test)
    X_tr = X_tr.fillna(X_tr.median())
    X_te = X_te.fillna(X_tr.median())
    return X_tr, X_te


# ── Metric collection helpers ───────────────────────────────────────────────
def measure_model(name, train_fn, predict_fn, X_train, X_test, y_train, y_test):
    """Train a model, measure time/memory, return metrics dict."""
    print(f"\n{'='*60}")
    print(f"  Running: {name}")
    print(f"{'='*60}")

    # Memory tracking
    tracemalloc.start()
    start_time = time.time()

    model = train_fn(X_train, y_train)

    train_time = time.time() - start_time
    _, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_memory_mb = peak_memory / (1024 * 1024)

    # Predictions
    inf_start = time.time()
    y_prob = predict_fn(model, X_test)
    inf_time = time.time() - inf_start
    inf_per_1k = (inf_time / len(X_test)) * 1000 * 1000  # ms per 1000 rows

    auc = roc_auc_score(y_test, y_prob)
    ll = log_loss(y_test, y_prob)

    print(f"  AUC: {auc:.4f} | Log Loss: {ll:.4f}")
    print(f"  Train: {train_time:.1f}s | Memory: {peak_memory_mb:.0f}MB | Inference: {inf_per_1k:.1f}ms/1k")

    return model, {
        "auc_roc": round(auc, 4),
        "log_loss": round(ll, 4),
        "train_time_sec": round(train_time, 2),
        "inference_time_ms_per_1k": round(inf_per_1k, 1),
        "peak_memory_mb": round(peak_memory_mb, 1),
    }


def cv_auc(train_fn, predict_fn, X, y, prep_fn=None):
    """5-fold stratified CV AUC. prep_fn encodes each fold independently."""
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    aucs = []
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
        if prep_fn:
            X_tr, X_val = prep_fn(X_tr, X_val)
        model = train_fn(X_tr, y_tr)
        y_prob = predict_fn(model, X_val)
        aucs.append(roc_auc_score(y_val, y_prob))
    return round(np.mean(aucs), 4)


def scaling_curve(train_fn, predict_fn, X_train, X_test, y_train, y_test, prep_fn=None):
    """AUC at each subsample size."""
    points = []
    for n in SCALING_SIZES:
        if n > len(X_train):
            break
        idx = X_train.sample(n=n, random_state=RANDOM_STATE).index
        X_sub, y_sub = X_train.loc[idx], y_train.loc[idx]
        X_te_local, X_sub_local = X_test.copy(), X_sub.copy()
        if prep_fn:
            X_sub_local, X_te_local = prep_fn(X_sub_local, X_te_local)
        model = train_fn(X_sub_local, y_sub)
        y_prob = predict_fn(model, X_te_local)
        auc = roc_auc_score(y_test, y_prob)
        points.append({"n_samples": n, "auc": round(auc, 4)})
        print(f"    n={n}: AUC={auc:.4f}")
    return points


# ═══════════════════════════════════════════════════════════════════════════
#  MODEL DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════

# ── LightGBM (Default) ─────────────────────────────────────────────────────
def run_lgbm_default(X_train, X_test, y_train, y_test, X_full, y_full):
    import lightgbm as lgb

    def prep(Xtr, Xte):
        return encode_for_lgbm(Xtr, Xte)[:2]

    def train_fn(X, y):
        dtrain = lgb.Dataset(X, label=y)
        return lgb.train({"objective": "binary", "metric": "auc", "verbosity": -1},
                         dtrain, num_boost_round=100)

    def predict_fn(model, X):
        return model.predict(X)

    X_tr_enc, X_te_enc, _ = encode_for_lgbm(X_train, X_test)

    _, metrics = measure_model(
        "LightGBM (Default)", train_fn, predict_fn,
        X_tr_enc, X_te_enc, y_train, y_test
    )

    print("  CV AUC...")
    metrics["auc_roc"] = cv_auc(train_fn, predict_fn, X_full, y_full, prep_fn=prep)
    print(f"  CV AUC: {metrics['auc_roc']}")

    print("  Scaling curve...")
    scaling = scaling_curve(train_fn, predict_fn, X_tr_enc, X_te_enc, y_train, y_test)

    return {
        "name": "LightGBM (Default)",
        "category": "gradient_boosting",
        "tuned": False,
        "metrics": metrics,
        "scaling": scaling,
        "raw_data_handling": {
            "missing_values": "native",
            "categorical_features": "needs_encoding",
            "class_imbalance": "scale_pos_weight",
        },
    }


# ── LightGBM (Tuned) ───────────────────────────────────────────────────────
def run_lgbm_tuned(X_train, X_test, y_train, y_test, X_full, y_full):
    import lightgbm as lgb
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    X_tr_enc, X_te_enc, _ = encode_for_lgbm(X_train, X_test)

    def objective(trial):
        params = {
            "objective": "binary",
            "metric": "auc",
            "verbosity": -1,
            "num_leaves": trial.suggest_int("num_leaves", 20, 150),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }
        n_rounds = trial.suggest_int("n_rounds", 50, 500)

        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
        aucs = []
        for tr_idx, val_idx in skf.split(X_tr_enc, y_train):
            dtrain = lgb.Dataset(X_tr_enc.iloc[tr_idx], label=y_train.iloc[tr_idx])
            dval = lgb.Dataset(X_tr_enc.iloc[val_idx], label=y_train.iloc[val_idx])
            model = lgb.train(params, dtrain, num_boost_round=n_rounds,
                              valid_sets=[dval], callbacks=[lgb.early_stopping(10, verbose=False)])
            pred = model.predict(X_tr_enc.iloc[val_idx])
            aucs.append(roc_auc_score(y_train.iloc[val_idx], pred))
        return np.mean(aucs)

    print(f"\n  Optuna: {OPTUNA_TRIALS} trials...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=OPTUNA_TRIALS)
    best = study.best_params
    n_rounds = best.pop("n_rounds")
    best.update({"objective": "binary", "metric": "auc", "verbosity": -1})

    def train_fn(X, y):
        dtrain = lgb.Dataset(X, label=y)
        return lgb.train(best, dtrain, num_boost_round=n_rounds)

    def predict_fn(model, X):
        return model.predict(X)

    _, metrics = measure_model(
        "LightGBM (Tuned)", train_fn, predict_fn,
        X_tr_enc, X_te_enc, y_train, y_test
    )

    def prep(Xtr, Xte):
        return encode_for_lgbm(Xtr, Xte)[:2]

    print("  CV AUC...")
    metrics["auc_roc"] = cv_auc(train_fn, predict_fn, X_full, y_full, prep_fn=prep)
    print(f"  CV AUC: {metrics['auc_roc']}")

    print("  Scaling curve...")
    scaling = scaling_curve(train_fn, predict_fn, X_tr_enc, X_te_enc, y_train, y_test)

    return {
        "name": "LightGBM (Tuned)",
        "category": "gradient_boosting",
        "tuned": True,
        "metrics": metrics,
        "scaling": scaling,
        "raw_data_handling": {
            "missing_values": "native",
            "categorical_features": "needs_encoding",
            "class_imbalance": "scale_pos_weight",
        },
    }


# ── CatBoost (Default) ─────────────────────────────────────────────────────
def run_catboost_default(X_train, X_test, y_train, y_test, X_full, y_full):
    from catboost import CatBoostClassifier

    cat_cols = list(X_train.select_dtypes(include="object").columns)

    def train_fn(X, y):
        model = CatBoostClassifier(iterations=500, verbose=0, random_seed=RANDOM_STATE,
                                   cat_features=cat_cols, eval_metric="AUC")
        model.fit(X, y)
        return model

    def predict_fn(model, X):
        return model.predict_proba(X)[:, 1]

    _, metrics = measure_model(
        "CatBoost (Default)", train_fn, predict_fn,
        X_train, X_test, y_train, y_test
    )

    print("  CV AUC...")
    metrics["auc_roc"] = cv_auc(train_fn, predict_fn, X_full, y_full)
    print(f"  CV AUC: {metrics['auc_roc']}")

    print("  Scaling curve...")
    scaling = scaling_curve(train_fn, predict_fn, X_train, X_test, y_train, y_test)

    return {
        "name": "CatBoost (Default)",
        "category": "gradient_boosting",
        "tuned": False,
        "metrics": metrics,
        "scaling": scaling,
        "raw_data_handling": {
            "missing_values": "native",
            "categorical_features": "native",
            "class_imbalance": "auto_class_weights",
        },
    }


# ── CatBoost (Tuned) ───────────────────────────────────────────────────────
def run_catboost_tuned(X_train, X_test, y_train, y_test, X_full, y_full):
    from catboost import CatBoostClassifier
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    cat_cols = list(X_train.select_dtypes(include="object").columns)

    def objective(trial):
        params = {
            "iterations": trial.suggest_int("iterations", 100, 1000),
            "depth": trial.suggest_int("depth", 4, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-8, 10.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "random_seed": RANDOM_STATE,
            "verbose": 0,
            "cat_features": cat_cols,
            "eval_metric": "AUC",
        }

        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
        aucs = []
        for tr_idx, val_idx in skf.split(X_train, y_train):
            model = CatBoostClassifier(**params)
            model.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx],
                      eval_set=(X_train.iloc[val_idx], y_train.iloc[val_idx]),
                      early_stopping_rounds=20, verbose=0)
            pred = model.predict_proba(X_train.iloc[val_idx])[:, 1]
            aucs.append(roc_auc_score(y_train.iloc[val_idx], pred))
        return np.mean(aucs)

    print(f"\n  Optuna: {OPTUNA_TRIALS} trials...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=OPTUNA_TRIALS)
    best = study.best_params
    best.update({"verbose": 0, "random_seed": RANDOM_STATE, "cat_features": cat_cols, "eval_metric": "AUC"})

    def train_fn(X, y):
        model = CatBoostClassifier(**best)
        model.fit(X, y, verbose=0)
        return model

    def predict_fn(model, X):
        return model.predict_proba(X)[:, 1]

    _, metrics = measure_model(
        "CatBoost (Tuned)", train_fn, predict_fn,
        X_train, X_test, y_train, y_test
    )

    print("  CV AUC...")
    metrics["auc_roc"] = cv_auc(train_fn, predict_fn, X_full, y_full)
    print(f"  CV AUC: {metrics['auc_roc']}")

    print("  Scaling curve...")
    scaling = scaling_curve(train_fn, predict_fn, X_train, X_test, y_train, y_test)

    return {
        "name": "CatBoost (Tuned)",
        "category": "gradient_boosting",
        "tuned": True,
        "metrics": metrics,
        "scaling": scaling,
        "raw_data_handling": {
            "missing_values": "native",
            "categorical_features": "native",
            "class_imbalance": "auto_class_weights",
        },
    }


# ── AutoGluon ───────────────────────────────────────────────────────────────
def run_autogluon(X_train, X_test, y_train, y_test, X_full, y_full):
    from autogluon.tabular import TabularPredictor
    import tempfile, shutil

    tmpdir = tempfile.mkdtemp()

    # Single fit for timing/memory measurement
    print("  Fitting AutoGluon (single run for metrics)...")
    tracemalloc.start()
    start_time = time.time()

    train_df = X_train.copy()
    train_df["Churn"] = y_train.values
    predictor = TabularPredictor(
        label="Churn", eval_metric="roc_auc", path=tmpdir, verbosity=1
    ).fit(train_df, presets="medium_quality", time_limit=120)

    train_time = time.time() - start_time
    _, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Predictions
    inf_start = time.time()
    y_prob = predictor.predict_proba(X_test)[1].values
    inf_time = time.time() - inf_start

    auc = roc_auc_score(y_test, y_prob)
    ll = log_loss(y_test, y_prob)
    inf_per_1k = (inf_time / len(X_test)) * 1000 * 1000

    print(f"  AUC: {auc:.4f} | Log Loss: {ll:.4f}")
    print(f"  Train: {train_time:.1f}s | Memory: {peak_memory / (1024*1024):.0f}MB | Inference: {inf_per_1k:.1f}ms/1k")

    metrics = {
        "auc_roc": round(auc, 4),
        "log_loss": round(ll, 4),
        "train_time_sec": round(train_time, 2),
        "inference_time_ms_per_1k": round(inf_per_1k, 1),
        "peak_memory_mb": round(peak_memory / (1024 * 1024), 1),
    }
    shutil.rmtree(tmpdir, ignore_errors=True)

    # CV: 5-fold
    print("  CV AUC...")
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    aucs = []
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_full, y_full)):
        fold_dir = tempfile.mkdtemp()
        td = X_full.iloc[tr_idx].copy()
        td["Churn"] = y_full.iloc[tr_idx].values
        pred_fold = TabularPredictor(
            label="Churn", eval_metric="roc_auc", path=fold_dir, verbosity=0
        ).fit(td, presets="medium_quality", time_limit=120)
        pred = pred_fold.predict_proba(X_full.iloc[val_idx])[1].values
        fold_auc = roc_auc_score(y_full.iloc[val_idx], pred)
        aucs.append(fold_auc)
        print(f"    Fold {fold+1}: {fold_auc:.4f}")
        shutil.rmtree(fold_dir, ignore_errors=True)
    metrics["auc_roc"] = round(np.mean(aucs), 4)
    print(f"  CV AUC: {metrics['auc_roc']}")

    # Scaling curve
    print("  Scaling curve...")
    scaling = []
    for n in SCALING_SIZES:
        if n > len(X_train):
            break
        idx = X_train.sample(n=n, random_state=RANDOM_STATE).index
        scale_dir = tempfile.mkdtemp()
        td = X_train.loc[idx].copy()
        td["Churn"] = y_train.loc[idx].values
        pred_scale = TabularPredictor(
            label="Churn", eval_metric="roc_auc", path=scale_dir, verbosity=0
        ).fit(td, presets="medium_quality", time_limit=60)
        pred = pred_scale.predict_proba(X_test)[1].values
        sc_auc = roc_auc_score(y_test, pred)
        scaling.append({"n_samples": n, "auc": round(sc_auc, 4)})
        print(f"    n={n}: AUC={sc_auc:.4f}")
        shutil.rmtree(scale_dir, ignore_errors=True)

    return {
        "name": "AutoGluon",
        "category": "automl",
        "tuned": False,
        "metrics": metrics,
        "scaling": scaling,
        "raw_data_handling": {
            "missing_values": "native",
            "categorical_features": "native",
            "class_imbalance": "native",
        },
    }


# ── TabPFN ──────────────────────────────────────────────────────────────────
def run_tabpfn(X_train, X_test, y_train, y_test, X_full, y_full):
    from tabpfn import TabPFNClassifier

    X_tr_enc, X_te_enc = encode_for_numeric(X_train, X_test)

    def train_fn(X, y):
        model = TabPFNClassifier(device="cpu")
        model.fit(X, y)
        return model

    def predict_fn(model, X):
        return model.predict_proba(X)[:, 1]

    _, metrics = measure_model(
        "TabPFN", train_fn, predict_fn,
        X_tr_enc, X_te_enc, y_train, y_test
    )

    print("  CV AUC...")

    def prep(Xtr, Xte):
        return encode_for_numeric(Xtr, Xte)

    metrics["auc_roc"] = cv_auc(train_fn, predict_fn, X_full, y_full, prep_fn=prep)
    print(f"  CV AUC: {metrics['auc_roc']}")

    print("  Scaling curve...")
    scaling = scaling_curve(train_fn, predict_fn, X_tr_enc, X_te_enc, y_train, y_test)

    return {
        "name": "TabPFN",
        "category": "foundation_model",
        "tuned": False,
        "metrics": metrics,
        "scaling": scaling,
        "raw_data_handling": {
            "missing_values": "native",
            "categorical_features": "needs_encoding",
            "class_imbalance": "none",
        },
    }


# ── FT-Transformer (manual PyTorch) ────────────────────────────────────────
def run_ft_transformer(X_train, X_test, y_train, y_test, X_full, y_full):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    X_tr_enc, X_te_enc = encode_for_numeric(X_train, X_test)

    n_features = X_tr_enc.shape[1]
    d_token = 64
    n_heads = 4
    n_layers = 3
    d_ffn = 128

    class FTTransformer(nn.Module):
        def __init__(self, n_features, d_token, n_heads, n_layers, d_ffn):
            super().__init__()
            # Per-feature linear embeddings
            self.feature_embeddings = nn.ModuleList([
                nn.Linear(1, d_token) for _ in range(n_features)
            ])
            # CLS token
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_token))
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_token, nhead=n_heads, dim_feedforward=d_ffn,
                dropout=0.1, batch_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
            self.head = nn.Linear(d_token, 1)

        def forward(self, x):
            # x: (batch, n_features)
            tokens = []
            for i, emb in enumerate(self.feature_embeddings):
                tokens.append(emb(x[:, i:i+1]))  # (batch, d_token)
            tokens = torch.stack(tokens, dim=1)  # (batch, n_features, d_token)
            cls = self.cls_token.expand(x.size(0), -1, -1)
            tokens = torch.cat([cls, tokens], dim=1)
            out = self.transformer(tokens)
            return self.head(out[:, 0]).squeeze(-1)

    def make_tensors(X, y=None):
        Xt = torch.tensor(X.values, dtype=torch.float32)
        if y is not None:
            yt = torch.tensor(y.values, dtype=torch.float32)
            return Xt, yt
        return Xt

    def train_fn(X, y):
        Xt, yt = make_tensors(X, y)
        ds = TensorDataset(Xt, yt)
        dl = DataLoader(ds, batch_size=256, shuffle=True)

        model = FTTransformer(X.shape[1], d_token, n_heads, n_layers, d_ffn)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
        criterion = nn.BCEWithLogitsLoss()

        model.train()
        for epoch in range(50):
            for xb, yb in dl:
                optimizer.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                optimizer.step()
        model.eval()
        return model

    def predict_fn(model, X):
        Xt = make_tensors(X)
        with torch.no_grad():
            logits = model(Xt)
            return torch.sigmoid(logits).numpy()

    _, metrics = measure_model(
        "FT-Transformer", train_fn, predict_fn,
        X_tr_enc, X_te_enc, y_train, y_test
    )

    print("  CV AUC...")

    def prep(Xtr, Xte):
        return encode_for_numeric(Xtr, Xte)

    metrics["auc_roc"] = cv_auc(train_fn, predict_fn, X_full, y_full, prep_fn=prep)
    print(f"  CV AUC: {metrics['auc_roc']}")

    print("  Scaling curve...")
    scaling = scaling_curve(train_fn, predict_fn, X_tr_enc, X_te_enc, y_train, y_test)

    return {
        "name": "FT-Transformer",
        "category": "deep_learning",
        "tuned": False,
        "metrics": metrics,
        "scaling": scaling,
        "raw_data_handling": {
            "missing_values": "needs_imputation",
            "categorical_features": "needs_encoding",
            "class_imbalance": "none",
        },
    }


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  TABULAR ARENA — Telco Churn Benchmark")
    print("=" * 60)

    df, target_rate = load_data()
    X_train, X_test, y_train, y_test = prepare_splits(df)
    X_full = df.drop(columns=["Churn"])
    y_full = df["Churn"]

    print(f"\n  Train: {len(X_train)} rows | Test: {len(X_test)} rows")

    results = {
        "dataset": "telco_churn",
        "n_samples": len(df),
        "n_features": df.shape[1] - 1,
        "target_rate": round(target_rate, 3),
        "hardware": get_hardware_info(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "models": [],
    }

    runners = [
        ("LightGBM (Default)", run_lgbm_default),
        ("LightGBM (Tuned)", run_lgbm_tuned),
        ("CatBoost (Default)", run_catboost_default),
        ("CatBoost (Tuned)", run_catboost_tuned),
        ("AutoGluon", run_autogluon),
        ("TabPFN", run_tabpfn),
        ("FT-Transformer", run_ft_transformer),
    ]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    for name, runner in runners:
        try:
            result = runner(X_train, X_test, y_train, y_test, X_full, y_full)
            results["models"].append(result)
            # Save incrementally after each model
            with open(OUTPUT_PATH, "w") as f:
                json.dump(results, f, indent=2)
            print(f"  >> Saved ({len(results['models'])} models so far)")
        except Exception as e:
            print(f"\n  FAILED: {name} — {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"  Results saved to {OUTPUT_PATH}")
    print(f"  {len(results['models'])} models completed")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

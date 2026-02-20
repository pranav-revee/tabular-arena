"""
Tabular Arena — Home Credit Default Risk Benchmark
Uses ALL 7 dataset tables with aggregated feature engineering.
Supports GPU (pass --gpu flag) for Colab H100 / A100.
"""

import json
import time
import psutil
import platform
import os
import sys
import warnings
import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

# ── Paths ───────────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent.parent.parent / "Dataset" / "home-credit-default-risk"
OUTPUT_PATH = Path(__file__).parent.parent / "results" / "credit_results.json"

SCALING_SIZES = [1000, 5000, 20000, 50000, 150000, 245000]
TABPFN_MAX_SAMPLES = 10000
RANDOM_STATE = 42
OPTUNA_TRIALS = 25          # 25 is enough for good hyperparams, halves tuning time
CV_FOLDS = 5

# Parsed at runtime
USE_GPU = False
DEVICE = "cpu"


# ── Hardware info ───────────────────────────────────────────────────────────
def get_hardware_info():
    cpu = platform.processor() or "unknown"
    try:
        import subprocess
        cpu = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
        ).strip()
    except Exception:
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if "model name" in line:
                        cpu = line.split(":")[1].strip()
                        break
        except Exception:
            pass
    ram_gb = psutil.virtual_memory().total / (1024 ** 3)
    info = {
        "cpu": cpu,
        "ram_gb": round(ram_gb, 1),
        "os": f"{platform.system()} {platform.release()}",
        "python": platform.python_version(),
    }
    if USE_GPU:
        import torch
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            total = getattr(props, 'total_memory', None) or getattr(props, 'total_mem', 0)
            info["gpu_memory_gb"] = round(total / 1024**3, 1)
    return info


# ═══════════════════════════════════════════════════════════════════════════
#  MULTI-TABLE FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════

def _agg_numeric(df, group_col, prefix):
    """Aggregate all numeric columns by group_col with common stats."""
    num_cols = df.select_dtypes(include="number").columns.drop(group_col, errors="ignore")
    agg = df.groupby(group_col)[num_cols].agg(["mean", "max", "min", "sum"]).reset_index()
    agg.columns = [group_col] + [f"{prefix}_{c[0]}_{c[1]}" for c in agg.columns[1:]]
    # Add count
    counts = df.groupby(group_col).size().reset_index(name=f"{prefix}_count")
    return agg.merge(counts, on=group_col, how="left")


def _agg_categorical(df, group_col, cat_col, prefix):
    """One-hot encode a categorical column and aggregate by group."""
    dummies = pd.get_dummies(df[[group_col, cat_col]], columns=[cat_col], prefix=prefix)
    return dummies.groupby(group_col).mean().reset_index()


def build_features():
    """Load all 7 tables, engineer features, return single merged DataFrame."""
    print("\n  Loading all dataset tables...")
    t0 = time.time()

    app = pd.read_csv(DATA_DIR / "application_train.csv")
    bureau = pd.read_csv(DATA_DIR / "bureau.csv")
    bb = pd.read_csv(DATA_DIR / "bureau_balance.csv")
    prev = pd.read_csv(DATA_DIR / "previous_application.csv")
    ins = pd.read_csv(DATA_DIR / "installments_payments.csv")
    pos = pd.read_csv(DATA_DIR / "POS_CASH_balance.csv")
    cc = pd.read_csv(DATA_DIR / "credit_card_balance.csv")

    print(f"    Loaded 7 tables in {time.time() - t0:.1f}s")

    # ── Application table cleanup ─────────────────────────────────────
    app = app.drop(columns=["SK_ID_CURR"], errors="ignore")
    # DAYS_EMPLOYED anomaly: 365243 means unemployed
    app["DAYS_EMPLOYED"] = app["DAYS_EMPLOYED"].replace(365243, np.nan)
    # Keep SK_ID_CURR for merging — we'll re-add it
    app_raw = pd.read_csv(DATA_DIR / "application_train.csv", usecols=["SK_ID_CURR"])
    app.insert(0, "SK_ID_CURR", app_raw["SK_ID_CURR"])

    # ── Bureau features ───────────────────────────────────────────────
    print("    Engineering bureau features...")

    # Bureau balance → aggregate to bureau level first
    bb_agg = bb.groupby("SK_ID_BUREAU").agg(
        bb_months_count=("MONTHS_BALANCE", "count"),
        bb_months_min=("MONTHS_BALANCE", "min"),
        bb_dpd_status_sum=("STATUS", lambda x: (x.astype(str).str.isdigit().astype(int) * x.astype(str).apply(lambda v: int(v) if v.isdigit() else 0)).sum()),
    ).reset_index()

    bureau = bureau.merge(bb_agg, on="SK_ID_BUREAU", how="left")

    # Bureau → aggregate to applicant level
    bureau_agg = _agg_numeric(bureau, "SK_ID_CURR", "bur")

    # Bureau: count active vs closed credits
    bureau["CREDIT_ACTIVE_BIN"] = (bureau["CREDIT_ACTIVE"] == "Active").astype(int)
    bureau["CREDIT_CLOSED_BIN"] = (bureau["CREDIT_ACTIVE"] == "Closed").astype(int)
    bur_status = bureau.groupby("SK_ID_CURR").agg(
        bur_active_count=("CREDIT_ACTIVE_BIN", "sum"),
        bur_closed_count=("CREDIT_CLOSED_BIN", "sum"),
        bur_credit_day_overdue_max=("CREDIT_DAY_OVERDUE", "max"),
        bur_debt_ratio=("AMT_CREDIT_SUM_DEBT", lambda x: x.sum() / max(x.count(), 1)),
    ).reset_index()
    bureau_agg = bureau_agg.merge(bur_status, on="SK_ID_CURR", how="left")

    # ── Previous application features ─────────────────────────────────
    print("    Engineering previous application features...")
    prev_agg = _agg_numeric(prev, "SK_ID_CURR", "prev")

    # Approval rate
    prev["APPROVED"] = (prev["NAME_CONTRACT_STATUS"] == "Approved").astype(int)
    prev["REFUSED"] = (prev["NAME_CONTRACT_STATUS"] == "Refused").astype(int)
    prev_status = prev.groupby("SK_ID_CURR").agg(
        prev_approved_count=("APPROVED", "sum"),
        prev_refused_count=("REFUSED", "sum"),
        prev_approval_rate=("APPROVED", "mean"),
    ).reset_index()
    prev_agg = prev_agg.merge(prev_status, on="SK_ID_CURR", how="left")

    # ── Installments features ─────────────────────────────────────────
    print("    Engineering installment features...")
    ins["PAYMENT_DELAY"] = ins["DAYS_ENTRY_PAYMENT"] - ins["DAYS_INSTALMENT"]
    ins["PAYMENT_SHORTFALL"] = ins["AMT_INSTALMENT"] - ins["AMT_PAYMENT"]
    ins["PAYMENT_RATIO"] = ins["AMT_PAYMENT"] / ins["AMT_INSTALMENT"].replace(0, np.nan)

    ins_agg = ins.groupby("SK_ID_CURR").agg(
        ins_count=("SK_ID_PREV", "count"),
        ins_delay_mean=("PAYMENT_DELAY", "mean"),
        ins_delay_max=("PAYMENT_DELAY", "max"),
        ins_delay_positive_count=("PAYMENT_DELAY", lambda x: (x > 0).sum()),
        ins_shortfall_mean=("PAYMENT_SHORTFALL", "mean"),
        ins_shortfall_max=("PAYMENT_SHORTFALL", "max"),
        ins_payment_ratio_mean=("PAYMENT_RATIO", "mean"),
        ins_payment_ratio_min=("PAYMENT_RATIO", "min"),
        ins_amt_payment_sum=("AMT_PAYMENT", "sum"),
        ins_amt_instalment_sum=("AMT_INSTALMENT", "sum"),
    ).reset_index()

    # ── POS Cash balance features ─────────────────────────────────────
    print("    Engineering POS cash features...")
    pos_agg = pos.groupby("SK_ID_CURR").agg(
        pos_count=("SK_ID_PREV", "count"),
        pos_dpd_max=("SK_DPD", "max"),
        pos_dpd_mean=("SK_DPD", "mean"),
        pos_dpd_def_max=("SK_DPD_DEF", "max"),
        pos_months_balance_min=("MONTHS_BALANCE", "min"),
        pos_months_balance_max=("MONTHS_BALANCE", "max"),
        pos_instalment_future_mean=("CNT_INSTALMENT_FUTURE", "mean"),
    ).reset_index()

    # POS: late payment ratio
    pos["IS_LATE"] = (pos["SK_DPD"] > 0).astype(int)
    pos_late = pos.groupby("SK_ID_CURR").agg(
        pos_late_ratio=("IS_LATE", "mean"),
    ).reset_index()
    pos_agg = pos_agg.merge(pos_late, on="SK_ID_CURR", how="left")

    # ── Credit card features ──────────────────────────────────────────
    print("    Engineering credit card features...")
    cc["CC_UTILIZATION"] = cc["AMT_BALANCE"] / cc["AMT_CREDIT_LIMIT_ACTUAL"].replace(0, np.nan)

    cc_agg = cc.groupby("SK_ID_CURR").agg(
        cc_count=("SK_ID_PREV", "count"),
        cc_balance_mean=("AMT_BALANCE", "mean"),
        cc_balance_max=("AMT_BALANCE", "max"),
        cc_credit_limit_mean=("AMT_CREDIT_LIMIT_ACTUAL", "mean"),
        cc_utilization_mean=("CC_UTILIZATION", "mean"),
        cc_utilization_max=("CC_UTILIZATION", "max"),
        cc_dpd_max=("SK_DPD", "max"),
        cc_dpd_mean=("SK_DPD", "mean"),
        cc_drawings_atm_mean=("AMT_DRAWINGS_ATM_CURRENT", "mean"),
        cc_payment_current_mean=("AMT_PAYMENT_CURRENT", "mean"),
        cc_months_balance_min=("MONTHS_BALANCE", "min"),
    ).reset_index()

    # ── Merge everything ──────────────────────────────────────────────
    print("    Merging all features...")
    df = app.copy()
    for agg_df in [bureau_agg, prev_agg, ins_agg, pos_agg, cc_agg]:
        df = df.merge(agg_df, on="SK_ID_CURR", how="left")

    # Drop SK_ID_CURR — not a feature
    df = df.drop(columns=["SK_ID_CURR"])

    n_original = app.shape[1] - 2  # minus SK_ID_CURR and TARGET
    n_engineered = df.shape[1] - 1 - n_original  # minus TARGET
    print(f"    Original features: {n_original}")
    print(f"    Engineered features: {n_engineered}")
    print(f"    Total features: {df.shape[1] - 1}")
    print(f"    Feature engineering done in {time.time() - t0:.1f}s\n")

    target_rate = df["TARGET"].mean()
    return df, target_rate


# ── Data helpers ────────────────────────────────────────────────────────────
def prepare_splits(df):
    X = df.drop(columns=["TARGET"])
    y = df["TARGET"]
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)


def encode_for_lgbm(X_train, X_test):
    X_tr = X_train.copy()
    X_te = X_test.copy()
    encoders = {}
    for col in X_tr.select_dtypes(include="object").columns:
        le = LabelEncoder()
        combined = pd.concat([X_tr[col], X_te[col]]).astype(str)
        le.fit(combined)
        X_tr[col] = le.transform(X_tr[col].astype(str))
        X_te[col] = le.transform(X_te[col].astype(str))
        encoders[col] = le
    return X_tr, X_te, encoders


def encode_for_numeric(X_train, X_test):
    X_tr, X_te, _ = encode_for_lgbm(X_train, X_test)
    X_tr = X_tr.fillna(X_tr.median())
    X_te = X_te.fillna(X_tr.median())
    return X_tr, X_te


# ── Metric collection ──────────────────────────────────────────────────────
def measure_model(name, train_fn, predict_fn, X_train, X_test, y_train, y_test):
    print(f"\n{'='*60}")
    print(f"  Running: {name}")
    print(f"{'='*60}")

    process = psutil.Process(os.getpid())
    rss_before = process.memory_info().rss
    start_time = time.time()

    model = train_fn(X_train, y_train)

    train_time = time.time() - start_time
    rss_after = process.memory_info().rss
    peak_memory_mb = max(0.0, (rss_after - rss_before) / (1024 * 1024))

    inf_start = time.time()
    y_prob = predict_fn(model, X_test)
    inf_time = time.time() - inf_start
    inf_per_1k = (inf_time / len(X_test)) * 1000 * 1000

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
        print(f"    Fold {fold+1}: {aucs[-1]:.4f}")
    return round(np.mean(aucs), 4)


def scaling_curve(train_fn, predict_fn, X_train, X_test, y_train, y_test, prep_fn=None):
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

def _lgbm_params():
    base = {"objective": "binary", "metric": "auc", "verbosity": -1}
    if USE_GPU:
        base["device"] = "gpu"
        base["gpu_use_dp"] = False          # FP32 not needed, use FP16 for speed
        base["num_threads"] = os.cpu_count()
    return base


def _xgb_params():
    base = {"objective": "binary:logistic", "eval_metric": "auc", "verbosity": 0}
    if USE_GPU:
        base["device"] = "cuda"
        base["tree_method"] = "hist"        # GPU-accelerated histogram
    base["nthread"] = -1
    return base


def _catboost_kwargs():
    kw = {"verbose": 0, "random_seed": RANDOM_STATE, "eval_metric": "AUC"}
    if USE_GPU:
        kw["task_type"] = "GPU"
        kw["devices"] = "0"
        kw["gpu_ram_part"] = 0.9            # Use 90% of GPU RAM
    return kw


# ── LightGBM (Default) ─────────────────────────────────────────────────────
def run_lgbm_default(X_train, X_test, y_train, y_test, X_full, y_full):
    import lightgbm as lgb

    def prep(Xtr, Xte):
        return encode_for_lgbm(Xtr, Xte)[:2]

    params = _lgbm_params()

    def train_fn(X, y):
        dtrain = lgb.Dataset(X, label=y)
        return lgb.train(params, dtrain, num_boost_round=100)

    def predict_fn(model, X):
        return model.predict(X)

    X_tr_enc, X_te_enc, _ = encode_for_lgbm(X_train, X_test)

    _, metrics = measure_model("LightGBM (Default)", train_fn, predict_fn,
                               X_tr_enc, X_te_enc, y_train, y_test)

    print("  CV AUC...")
    metrics["auc_roc"] = cv_auc(train_fn, predict_fn, X_full, y_full, prep_fn=prep)
    print(f"  CV AUC: {metrics['auc_roc']}")

    print("  Scaling curve...")
    scaling = scaling_curve(train_fn, predict_fn, X_tr_enc, X_te_enc, y_train, y_test)

    return {
        "name": "LightGBM (Default)", "category": "gradient_boosting", "tuned": False,
        "metrics": metrics, "scaling": scaling,
        "raw_data_handling": {"missing_values": "native", "categorical_features": "needs_encoding", "class_imbalance": "scale_pos_weight"},
    }


# ── LightGBM (Tuned) ───────────────────────────────────────────────────────
def run_lgbm_tuned(X_train, X_test, y_train, y_test, X_full, y_full):
    import lightgbm as lgb
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    X_tr_enc, X_te_enc, _ = encode_for_lgbm(X_train, X_test)

    def objective(trial):
        params = _lgbm_params()
        params.update({
            "num_leaves": trial.suggest_int("num_leaves", 20, 150),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        })
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

    search_start = time.time()
    print(f"\n  Optuna: {OPTUNA_TRIALS} trials...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=OPTUNA_TRIALS)
    search_time = time.time() - search_start
    best = study.best_params
    n_rounds = best.pop("n_rounds")
    best.update(_lgbm_params())

    def train_fn(X, y):
        dtrain = lgb.Dataset(X, label=y)
        return lgb.train(best, dtrain, num_boost_round=n_rounds)

    def predict_fn(model, X):
        return model.predict(X)

    _, metrics = measure_model("LightGBM (Tuned)", train_fn, predict_fn,
                               X_tr_enc, X_te_enc, y_train, y_test)
    metrics["search_time_sec"] = round(search_time, 2)
    metrics["fit_time_sec"] = metrics["train_time_sec"]
    metrics["train_time_sec"] = round(search_time + metrics["fit_time_sec"], 2)

    def prep(Xtr, Xte):
        return encode_for_lgbm(Xtr, Xte)[:2]

    print("  CV AUC...")
    metrics["auc_roc"] = cv_auc(train_fn, predict_fn, X_full, y_full, prep_fn=prep)
    print(f"  CV AUC: {metrics['auc_roc']}")

    print("  Scaling curve...")
    scaling = scaling_curve(train_fn, predict_fn, X_tr_enc, X_te_enc, y_train, y_test)

    return {
        "name": "LightGBM (Tuned)", "category": "gradient_boosting", "tuned": True,
        "metrics": metrics, "scaling": scaling,
        "raw_data_handling": {"missing_values": "native", "categorical_features": "needs_encoding", "class_imbalance": "scale_pos_weight"},
    }


# ── XGBoost (Default) ──────────────────────────────────────────────────────
def run_xgb_default(X_train, X_test, y_train, y_test, X_full, y_full):
    import xgboost as xgb

    params = _xgb_params()

    def prep(Xtr, Xte):
        return encode_for_lgbm(Xtr, Xte)[:2]

    def train_fn(X, y):
        dtrain = xgb.DMatrix(X, label=y)
        return xgb.train(params, dtrain, num_boost_round=100)

    def predict_fn(model, X):
        return model.predict(xgb.DMatrix(X))

    X_tr_enc, X_te_enc, _ = encode_for_lgbm(X_train, X_test)

    _, metrics = measure_model("XGBoost (Default)", train_fn, predict_fn,
                               X_tr_enc, X_te_enc, y_train, y_test)

    print("  CV AUC...")
    metrics["auc_roc"] = cv_auc(train_fn, predict_fn, X_full, y_full, prep_fn=prep)
    print(f"  CV AUC: {metrics['auc_roc']}")

    print("  Scaling curve...")
    scaling = scaling_curve(train_fn, predict_fn, X_tr_enc, X_te_enc, y_train, y_test)

    return {
        "name": "XGBoost (Default)", "category": "gradient_boosting", "tuned": False,
        "metrics": metrics, "scaling": scaling,
        "raw_data_handling": {"missing_values": "native", "categorical_features": "needs_encoding", "class_imbalance": "scale_pos_weight"},
    }


# ── XGBoost (Tuned) ────────────────────────────────────────────────────────
def run_xgb_tuned(X_train, X_test, y_train, y_test, X_full, y_full):
    import xgboost as xgb
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    X_tr_enc, X_te_enc, _ = encode_for_lgbm(X_train, X_test)

    def objective(trial):
        params = _xgb_params()
        params.update({
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        })
        n_rounds = trial.suggest_int("n_rounds", 50, 500)
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
        aucs = []
        for tr_idx, val_idx in skf.split(X_tr_enc, y_train):
            dtrain = xgb.DMatrix(X_tr_enc.iloc[tr_idx], label=y_train.iloc[tr_idx])
            dval = xgb.DMatrix(X_tr_enc.iloc[val_idx], label=y_train.iloc[val_idx])
            model = xgb.train(params, dtrain, num_boost_round=n_rounds,
                              evals=[(dval, "val")], early_stopping_rounds=10, verbose_eval=False)
            pred = model.predict(dval)
            aucs.append(roc_auc_score(y_train.iloc[val_idx], pred))
        return np.mean(aucs)

    search_start = time.time()
    print(f"\n  Optuna: {OPTUNA_TRIALS} trials...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=OPTUNA_TRIALS)
    search_time = time.time() - search_start
    best = study.best_params
    n_rounds = best.pop("n_rounds")
    best.update(_xgb_params())

    def train_fn(X, y):
        dtrain = xgb.DMatrix(X, label=y)
        return xgb.train(best, dtrain, num_boost_round=n_rounds)

    def predict_fn(model, X):
        return model.predict(xgb.DMatrix(X))

    _, metrics = measure_model("XGBoost (Tuned)", train_fn, predict_fn,
                               X_tr_enc, X_te_enc, y_train, y_test)
    metrics["search_time_sec"] = round(search_time, 2)
    metrics["fit_time_sec"] = metrics["train_time_sec"]
    metrics["train_time_sec"] = round(search_time + metrics["fit_time_sec"], 2)

    def prep(Xtr, Xte):
        return encode_for_lgbm(Xtr, Xte)[:2]

    print("  CV AUC...")
    metrics["auc_roc"] = cv_auc(train_fn, predict_fn, X_full, y_full, prep_fn=prep)
    print(f"  CV AUC: {metrics['auc_roc']}")

    print("  Scaling curve...")
    scaling = scaling_curve(train_fn, predict_fn, X_tr_enc, X_te_enc, y_train, y_test)

    return {
        "name": "XGBoost (Tuned)", "category": "gradient_boosting", "tuned": True,
        "metrics": metrics, "scaling": scaling,
        "raw_data_handling": {"missing_values": "native", "categorical_features": "needs_encoding", "class_imbalance": "scale_pos_weight"},
    }


# ── CatBoost helpers ────────────────────────────────────────────────────────
def _catboost_fill_nan(df, cat_cols):
    """CatBoost cannot handle NaN in categorical features — fill with 'missing'."""
    df = df.copy()
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].fillna("missing").astype(str)
    return df


# ── CatBoost (Default) ─────────────────────────────────────────────────────
def run_catboost_default(X_train, X_test, y_train, y_test, X_full, y_full):
    from catboost import CatBoostClassifier

    cat_cols = list(X_train.select_dtypes(include="object").columns)
    X_train = _catboost_fill_nan(X_train, cat_cols)
    X_test = _catboost_fill_nan(X_test, cat_cols)
    X_full = _catboost_fill_nan(X_full, cat_cols)

    def train_fn(X, y):
        X = _catboost_fill_nan(X, cat_cols)
        kw = _catboost_kwargs()
        kw["cat_features"] = cat_cols
        kw["iterations"] = 500
        model = CatBoostClassifier(**kw)
        model.fit(X, y)
        return model

    def predict_fn(model, X):
        X = _catboost_fill_nan(X, cat_cols)
        return model.predict_proba(X)[:, 1]

    _, metrics = measure_model("CatBoost (Default)", train_fn, predict_fn,
                               X_train, X_test, y_train, y_test)

    print("  CV AUC...")
    metrics["auc_roc"] = cv_auc(train_fn, predict_fn, X_full, y_full)
    print(f"  CV AUC: {metrics['auc_roc']}")

    print("  Scaling curve...")
    scaling = scaling_curve(train_fn, predict_fn, X_train, X_test, y_train, y_test)

    return {
        "name": "CatBoost (Default)", "category": "gradient_boosting", "tuned": False,
        "metrics": metrics, "scaling": scaling,
        "raw_data_handling": {"missing_values": "native", "categorical_features": "native", "class_imbalance": "auto_class_weights"},
    }


# ── CatBoost (Tuned) ───────────────────────────────────────────────────────
def run_catboost_tuned(X_train, X_test, y_train, y_test, X_full, y_full):
    from catboost import CatBoostClassifier
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    cat_cols = list(X_train.select_dtypes(include="object").columns)
    X_train = _catboost_fill_nan(X_train, cat_cols)
    X_test = _catboost_fill_nan(X_test, cat_cols)
    X_full = _catboost_fill_nan(X_full, cat_cols)

    def objective(trial):
        kw = _catboost_kwargs()
        kw["cat_features"] = cat_cols
        kw.update({
            "iterations": trial.suggest_int("iterations", 100, 1000),
            "depth": trial.suggest_int("depth", 4, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-8, 10.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        })
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
        aucs = []
        for tr_idx, val_idx in skf.split(X_train, y_train):
            model = CatBoostClassifier(**kw)
            model.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx],
                      eval_set=(X_train.iloc[val_idx], y_train.iloc[val_idx]),
                      early_stopping_rounds=20, verbose=0)
            pred = model.predict_proba(X_train.iloc[val_idx])[:, 1]
            aucs.append(roc_auc_score(y_train.iloc[val_idx], pred))
        return np.mean(aucs)

    search_start = time.time()
    print(f"\n  Optuna: {OPTUNA_TRIALS} trials...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=OPTUNA_TRIALS)
    search_time = time.time() - search_start
    best = study.best_params
    kw = _catboost_kwargs()
    kw["cat_features"] = cat_cols
    kw.update(best)

    def train_fn(X, y):
        model = CatBoostClassifier(**kw)
        model.fit(X, y, verbose=0)
        return model

    def predict_fn(model, X):
        return model.predict_proba(X)[:, 1]

    _, metrics = measure_model("CatBoost (Tuned)", train_fn, predict_fn,
                               X_train, X_test, y_train, y_test)
    metrics["search_time_sec"] = round(search_time, 2)
    metrics["fit_time_sec"] = metrics["train_time_sec"]
    metrics["train_time_sec"] = round(search_time + metrics["fit_time_sec"], 2)

    print("  CV AUC...")
    metrics["auc_roc"] = cv_auc(train_fn, predict_fn, X_full, y_full)
    print(f"  CV AUC: {metrics['auc_roc']}")

    print("  Scaling curve...")
    scaling = scaling_curve(train_fn, predict_fn, X_train, X_test, y_train, y_test)

    return {
        "name": "CatBoost (Tuned)", "category": "gradient_boosting", "tuned": True,
        "metrics": metrics, "scaling": scaling,
        "raw_data_handling": {"missing_values": "native", "categorical_features": "native", "class_imbalance": "auto_class_weights"},
    }


# ── AutoGluon ───────────────────────────────────────────────────────────────
def run_autogluon(X_train, X_test, y_train, y_test, X_full, y_full):
    from autogluon.tabular import TabularPredictor
    import tempfile, shutil

    ag_time = 180                              # 3 min is enough for medium_quality
    ag_cv_time = 120                            # CV folds can be shorter
    ag_scale_time = 60                          # Scaling needs even less
    ag_kw = {"num_gpus": 1} if USE_GPU else {}
    tmpdir = tempfile.mkdtemp()

    print("  Fitting AutoGluon...")
    process = psutil.Process(os.getpid())
    rss_before = process.memory_info().rss
    start_time = time.time()

    train_df = X_train.copy()
    train_df["TARGET"] = y_train.values
    predictor = TabularPredictor(
        label="TARGET", eval_metric="roc_auc", path=tmpdir, verbosity=1
    ).fit(train_df, presets="medium_quality", time_limit=ag_time, **ag_kw)

    train_time = time.time() - start_time
    rss_after = process.memory_info().rss
    peak_memory = max(0, rss_after - rss_before)

    inf_start = time.time()
    y_prob = predictor.predict_proba(X_test)[1].values
    inf_time = time.time() - inf_start

    auc = roc_auc_score(y_test, y_prob)
    ll = log_loss(y_test, y_prob)
    inf_per_1k = (inf_time / len(X_test)) * 1000 * 1000

    print(f"  AUC: {auc:.4f} | Log Loss: {ll:.4f}")
    print(f"  Train: {train_time:.1f}s | Memory: {peak_memory / (1024*1024):.0f}MB")

    metrics = {
        "auc_roc": round(auc, 4), "log_loss": round(ll, 4),
        "train_time_sec": round(train_time, 2),
        "inference_time_ms_per_1k": round(inf_per_1k, 1),
        "peak_memory_mb": round(peak_memory / (1024 * 1024), 1),
    }
    shutil.rmtree(tmpdir, ignore_errors=True)

    print("  CV AUC...")
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    aucs = []
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_full, y_full)):
        fold_dir = tempfile.mkdtemp()
        td = X_full.iloc[tr_idx].copy()
        td["TARGET"] = y_full.iloc[tr_idx].values
        pred_fold = TabularPredictor(
            label="TARGET", eval_metric="roc_auc", path=fold_dir, verbosity=0
        ).fit(td, presets="medium_quality", time_limit=ag_cv_time, **ag_kw)
        pred = pred_fold.predict_proba(X_full.iloc[val_idx])[1].values
        fold_auc = roc_auc_score(y_full.iloc[val_idx], pred)
        aucs.append(fold_auc)
        print(f"    Fold {fold+1}: {fold_auc:.4f}")
        shutil.rmtree(fold_dir, ignore_errors=True)
    metrics["auc_roc"] = round(np.mean(aucs), 4)
    print(f"  CV AUC: {metrics['auc_roc']}")

    print("  Scaling curve...")
    scaling = []
    for n in SCALING_SIZES:
        if n > len(X_train):
            break
        idx = X_train.sample(n=n, random_state=RANDOM_STATE).index
        scale_dir = tempfile.mkdtemp()
        td = X_train.loc[idx].copy()
        td["TARGET"] = y_train.loc[idx].values
        pred_scale = TabularPredictor(
            label="TARGET", eval_metric="roc_auc", path=scale_dir, verbosity=0
        ).fit(td, presets="medium_quality", time_limit=ag_scale_time, **ag_kw)
        pred = pred_scale.predict_proba(X_test)[1].values
        sc_auc = roc_auc_score(y_test, pred)
        scaling.append({"n_samples": n, "auc": round(sc_auc, 4)})
        print(f"    n={n}: AUC={sc_auc:.4f}")
        shutil.rmtree(scale_dir, ignore_errors=True)

    return {
        "name": "AutoGluon", "category": "automl", "tuned": False,
        "metrics": metrics, "scaling": scaling,
        "raw_data_handling": {"missing_values": "native", "categorical_features": "native", "class_imbalance": "native"},
    }


# ── TabPFN ──────────────────────────────────────────────────────────────────
def run_tabpfn(X_train, X_test, y_train, y_test, X_full, y_full):
    from tabpfn import TabPFNClassifier

    X_tr_enc, X_te_enc = encode_for_numeric(X_train, X_test)

    if len(X_tr_enc) > TABPFN_MAX_SAMPLES:
        print(f"  Subsampling to {TABPFN_MAX_SAMPLES} rows for TabPFN...")
        idx = X_tr_enc.sample(n=TABPFN_MAX_SAMPLES, random_state=RANDOM_STATE).index
        X_tr_sub = X_tr_enc.loc[idx]
        y_tr_sub = y_train.loc[idx]
    else:
        X_tr_sub = X_tr_enc
        y_tr_sub = y_train

    def train_fn(X, y):
        model = TabPFNClassifier(device=DEVICE, ignore_pretraining_limits=True)
        model.fit(X, y)
        return model

    def predict_fn(model, X):
        return model.predict_proba(X)[:, 1]

    _, metrics = measure_model("TabPFN", train_fn, predict_fn,
                               X_tr_sub, X_te_enc, y_tr_sub, y_test)

    print("  CV AUC...")
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    aucs = []
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_full, y_full)):
        X_tr_fold, X_val_fold = X_full.iloc[tr_idx], X_full.iloc[val_idx]
        y_tr_fold, y_val_fold = y_full.iloc[tr_idx], y_full.iloc[val_idx]
        X_tr_fold_enc, X_val_fold_enc = encode_for_numeric(X_tr_fold, X_val_fold)
        if len(X_tr_fold_enc) > TABPFN_MAX_SAMPLES:
            sub_idx = X_tr_fold_enc.sample(n=TABPFN_MAX_SAMPLES, random_state=RANDOM_STATE).index
            X_tr_fold_enc = X_tr_fold_enc.loc[sub_idx]
            y_tr_fold = y_tr_fold.loc[sub_idx]
        model = train_fn(X_tr_fold_enc, y_tr_fold)
        y_prob = predict_fn(model, X_val_fold_enc)
        fold_auc = roc_auc_score(y_val_fold, y_prob)
        aucs.append(fold_auc)
        print(f"    Fold {fold+1}: {fold_auc:.4f}")
    metrics["auc_roc"] = round(np.mean(aucs), 4)
    print(f"  CV AUC: {metrics['auc_roc']}")

    print("  Scaling curve...")
    tabpfn_sizes = [s for s in SCALING_SIZES if s <= TABPFN_MAX_SAMPLES]
    scaling = []
    for n in tabpfn_sizes:
        sub_idx = X_tr_enc.sample(n=n, random_state=RANDOM_STATE).index
        model = train_fn(X_tr_enc.loc[sub_idx], y_train.loc[sub_idx])
        y_prob = predict_fn(model, X_te_enc)
        auc = roc_auc_score(y_test, y_prob)
        scaling.append({"n_samples": n, "auc": round(auc, 4)})
        print(f"    n={n}: AUC={auc:.4f}")

    return {
        "name": "TabPFN", "category": "foundation_model", "tuned": False,
        "metrics": metrics, "scaling": scaling,
        "raw_data_handling": {"missing_values": "native", "categorical_features": "needs_encoding", "class_imbalance": "none"},
    }


# ── FT-Transformer ─────────────────────────────────────────────────────────
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
    batch_size = 4096 if USE_GPU else 512   # 96GB GPU can handle much larger batches
    epochs = 30

    # H100 optimizations
    if USE_GPU and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    class FTTransformer(nn.Module):
        def __init__(self, n_feat, d_tok, n_h, n_l, d_ff):
            super().__init__()
            self.feature_embeddings = nn.ModuleList([nn.Linear(1, d_tok) for _ in range(n_feat)])
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_tok))
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_tok, nhead=n_h, dim_feedforward=d_ff, dropout=0.1, batch_first=True)
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_l)
            self.head = nn.Linear(d_tok, 1)

        def forward(self, x):
            tokens = torch.stack([emb(x[:, i:i+1]) for i, emb in enumerate(self.feature_embeddings)], dim=1)
            cls = self.cls_token.expand(x.size(0), -1, -1)
            tokens = torch.cat([cls, tokens], dim=1)
            return self.head(self.transformer(tokens)[:, 0]).squeeze(-1)

    dev = torch.device(DEVICE)
    use_amp = USE_GPU and torch.cuda.is_available()

    def train_fn(X, y):
        Xt = torch.tensor(X.values, dtype=torch.float32, device=dev)
        yt = torch.tensor(y.values, dtype=torch.float32, device=dev)
        dl = DataLoader(TensorDataset(Xt, yt), batch_size=batch_size, shuffle=True)

        model = FTTransformer(X.shape[1], d_token, n_heads, n_layers, d_ffn).to(dev)
        if use_amp:
            model = torch.compile(model, mode="reduce-overhead")

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
        criterion = nn.BCEWithLogitsLoss()
        scaler = torch.amp.GradScaler("cuda") if use_amp else None

        model.train()
        for epoch in range(epochs):
            for xb, yb in dl:
                optimizer.zero_grad(set_to_none=True)
                if use_amp:
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        loss = criterion(model(xb), yb)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss = criterion(model(xb), yb)
                    loss.backward()
                    optimizer.step()
        model.eval()
        return model

    def predict_fn(model, X):
        Xt = torch.tensor(X.values, dtype=torch.float32, device=dev)
        with torch.no_grad():
            if use_amp:
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(Xt)
            else:
                logits = model(Xt)
            return torch.sigmoid(logits.float()).cpu().numpy()

    _, metrics = measure_model("FT-Transformer", train_fn, predict_fn,
                               X_tr_enc, X_te_enc, y_train, y_test)

    print("  CV AUC...")
    def prep(Xtr, Xte):
        return encode_for_numeric(Xtr, Xte)
    metrics["auc_roc"] = cv_auc(train_fn, predict_fn, X_full, y_full, prep_fn=prep)
    print(f"  CV AUC: {metrics['auc_roc']}")

    print("  Scaling curve...")
    scaling = scaling_curve(train_fn, predict_fn, X_tr_enc, X_te_enc, y_train, y_test)

    if USE_GPU:
        torch.cuda.empty_cache()

    return {
        "name": "FT-Transformer", "category": "deep_learning", "tuned": False,
        "metrics": metrics, "scaling": scaling,
        "raw_data_handling": {"missing_values": "needs_imputation", "categorical_features": "needs_encoding", "class_imbalance": "none"},
    }


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    global USE_GPU, DEVICE, DATA_DIR, OUTPUT_PATH

    parser = argparse.ArgumentParser(description="Home Credit Default Risk Benchmark")
    parser.add_argument("--gpu", action="store_true", help="Enable GPU acceleration")
    parser.add_argument("--data-dir", type=str, default=None, help="Override dataset directory")
    parser.add_argument("--output", type=str, default=None, help="Override output JSON path")
    args = parser.parse_args()

    USE_GPU = args.gpu
    if args.data_dir:
        DATA_DIR = Path(args.data_dir)
    if args.output:
        OUTPUT_PATH = Path(args.output)

    if USE_GPU:
        import torch
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        if DEVICE == "cuda":
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("  WARNING: --gpu flag set but CUDA not available, falling back to CPU")
            USE_GPU = False

    print("=" * 60)
    print("  TABULAR ARENA — Home Credit Default Risk Benchmark")
    print(f"  Device: {DEVICE.upper()} | GPU: {USE_GPU}")
    print(f"  Data: {DATA_DIR}")
    print("=" * 60)

    df, target_rate = build_features()
    X_train, X_test, y_train, y_test = prepare_splits(df)
    X_full = df.drop(columns=["TARGET"])
    y_full = df["TARGET"]

    print(f"  Train: {len(X_train)} rows | Test: {len(X_test)} rows")
    print(f"  Features: {X_train.shape[1]}")

    results = {
        "dataset": "home_credit",
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
        ("XGBoost (Default)", run_xgb_default),
        ("XGBoost (Tuned)", run_xgb_tuned),
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
            with open(OUTPUT_PATH, "w") as f:
                json.dump(results, f, indent=2)
            print(f"  >> Saved ({len(results['models'])} models so far)")
        except Exception as e:
            print(f"\n  FAILED: {name} — {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"  BENCHMARK COMPLETE")
    print(f"  {len(results['models'])} models → {OUTPUT_PATH}")
    print(f"{'='*60}")
    for m in sorted(results["models"], key=lambda x: -x["metrics"]["auc_roc"]):
        print(f"  {m['name']:<25s}  AUC: {m['metrics']['auc_roc']:.4f}  Time: {m['metrics']['train_time_sec']:.1f}s")


if __name__ == "__main__":
    main()

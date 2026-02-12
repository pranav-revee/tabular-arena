"""
Run the remaining models (AutoGluon, TabPFN, FT-Transformer) and append to existing results.
"""
import json
import os
import time
import psutil
import warnings
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

DATA_PATH = Path(__file__).parent.parent.parent / "Dataset" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
OUTPUT_PATH = Path(__file__).parent.parent / "results" / "churn_results.json"
SCALING_SIZES = [500, 1000, 2000, 3500, 5600]
RANDOM_STATE = 42
CV_FOLDS = 5


def load_data():
    df = pd.read_csv(DATA_PATH)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.drop(columns=["customerID"])
    df["Churn"] = (df["Churn"] == "Yes").astype(int)
    return df


def encode_for_numeric(X_train, X_test):
    X_tr, X_te = X_train.copy(), X_test.copy()
    for col in X_tr.select_dtypes(include="object").columns:
        le = LabelEncoder()
        X_tr[col] = le.fit_transform(X_tr[col].astype(str))
        X_te[col] = le.transform(X_te[col].astype(str))
    X_tr = X_tr.fillna(X_tr.median())
    X_te = X_te.fillna(X_tr.median())
    return X_tr, X_te


# ═════════════════════════════════════════════════════════════════════════
#  AutoGluon — single process to avoid macOS fork issues
# ═════════════════════════════════════════════════════════════════════════
def run_autogluon(X_train, X_test, y_train, y_test, X_full, y_full):
    from autogluon.tabular import TabularPredictor
    import tempfile, shutil

    tmpdir = tempfile.mkdtemp()
    print("  AutoGluon: fitting (single run)...")

    process = psutil.Process(os.getpid())
    rss_before = process.memory_info().rss
    t0 = time.time()

    train_df = X_train.copy()
    train_df["Churn"] = y_train.values
    predictor = TabularPredictor(
        label="Churn", eval_metric="roc_auc", path=tmpdir, verbosity=2
    ).fit(
        train_df,
        presets="medium_quality",
        time_limit=120,
        num_cpus=1,  # avoid multiprocessing fork issues on macOS
    )

    train_time = time.time() - t0
    rss_after = process.memory_info().rss
    peak_mem_mb = max(0.0, (rss_after - rss_before) / (1024 ** 2))

    y_prob = predictor.predict_proba(X_test)[1].values
    inf_start = time.time()
    y_prob = predictor.predict_proba(X_test)[1].values
    inf_time = time.time() - inf_start

    auc = roc_auc_score(y_test, y_prob)
    ll = log_loss(y_test, y_prob)
    inf_per_1k = (inf_time / len(X_test)) * 1000 * 1000

    print(f"  AUC: {auc:.4f} | Log Loss: {ll:.4f}")
    print(f"  Train: {train_time:.1f}s | Memory: {peak_mem_mb:.0f}MB | Infer: {inf_per_1k:.1f}ms/1k")

    metrics = {
        "auc_roc": round(auc, 4),
        "log_loss": round(ll, 4),
        "train_time_sec": round(train_time, 2),
        "inference_time_ms_per_1k": round(inf_per_1k, 1),
        "peak_memory_mb": round(peak_mem_mb, 1),
    }
    shutil.rmtree(tmpdir, ignore_errors=True)

    # CV
    print("  CV AUC...")
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    aucs = []
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_full, y_full)):
        fold_dir = tempfile.mkdtemp()
        td = X_full.iloc[tr_idx].copy()
        td["Churn"] = y_full.iloc[tr_idx].values
        p = TabularPredictor(
            label="Churn", eval_metric="roc_auc", path=fold_dir, verbosity=0
        ).fit(td, presets="medium_quality", time_limit=120, num_cpus=1)
        pred = p.predict_proba(X_full.iloc[val_idx])[1].values
        fold_auc = roc_auc_score(y_full.iloc[val_idx], pred)
        aucs.append(fold_auc)
        print(f"    Fold {fold+1}: {fold_auc:.4f}")
        shutil.rmtree(fold_dir, ignore_errors=True)
    metrics["auc_roc"] = round(np.mean(aucs), 4)
    print(f"  CV AUC: {metrics['auc_roc']}")

    # Scaling
    print("  Scaling curve...")
    scaling = []
    for n in SCALING_SIZES:
        if n > len(X_train):
            break
        idx = X_train.sample(n=n, random_state=RANDOM_STATE).index
        sd = tempfile.mkdtemp()
        td = X_train.loc[idx].copy()
        td["Churn"] = y_train.loc[idx].values
        p = TabularPredictor(
            label="Churn", eval_metric="roc_auc", path=sd, verbosity=0
        ).fit(td, presets="medium_quality", time_limit=60, num_cpus=1)
        pred = p.predict_proba(X_test)[1].values
        sc_auc = roc_auc_score(y_test, pred)
        scaling.append({"n_samples": n, "auc": round(sc_auc, 4)})
        print(f"    n={n}: AUC={sc_auc:.4f}")
        shutil.rmtree(sd, ignore_errors=True)

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


# ═════════════════════════════════════════════════════════════════════════
#  TabPFN
# ═════════════════════════════════════════════════════════════════════════
def run_tabpfn(X_train, X_test, y_train, y_test, X_full, y_full):
    from tabpfn import TabPFNClassifier

    X_tr_enc, X_te_enc = encode_for_numeric(X_train, X_test)

    print("  TabPFN: fitting...")
    process = psutil.Process(os.getpid())
    rss_before = process.memory_info().rss
    t0 = time.time()
    model = TabPFNClassifier(device="cpu", ignore_pretraining_limits=True)
    model.fit(X_tr_enc, y_train)
    train_time = time.time() - t0
    rss_after = process.memory_info().rss
    peak_mem_mb = max(0.0, (rss_after - rss_before) / (1024 ** 2))

    inf_start = time.time()
    y_prob = model.predict_proba(X_te_enc)[:, 1]
    inf_time = time.time() - inf_start
    inf_per_1k = (inf_time / len(X_te_enc)) * 1000 * 1000

    auc = roc_auc_score(y_test, y_prob)
    ll = log_loss(y_test, y_prob)

    print(f"  AUC: {auc:.4f} | Log Loss: {ll:.4f}")
    print(f"  Train: {train_time:.1f}s | Memory: {peak_mem_mb:.0f}MB | Infer: {inf_per_1k:.1f}ms/1k")

    metrics = {
        "auc_roc": round(auc, 4),
        "log_loss": round(ll, 4),
        "train_time_sec": round(train_time, 2),
        "inference_time_ms_per_1k": round(inf_per_1k, 1),
        "peak_memory_mb": round(peak_mem_mb, 1),
    }

    # CV
    print("  CV AUC...")
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    aucs = []
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_full, y_full)):
        Xtr, Xval = X_full.iloc[tr_idx], X_full.iloc[val_idx]
        Xtr_e, Xval_e = encode_for_numeric(Xtr, Xval)
        m = TabPFNClassifier(device="cpu", ignore_pretraining_limits=True)
        m.fit(Xtr_e, y_full.iloc[tr_idx])
        pred = m.predict_proba(Xval_e)[:, 1]
        fold_auc = roc_auc_score(y_full.iloc[val_idx], pred)
        aucs.append(fold_auc)
        print(f"    Fold {fold+1}: {fold_auc:.4f}")
    metrics["auc_roc"] = round(np.mean(aucs), 4)
    print(f"  CV AUC: {metrics['auc_roc']}")

    # Scaling
    print("  Scaling curve...")
    scaling = []
    for n in SCALING_SIZES:
        if n > len(X_tr_enc):
            break
        idx = X_tr_enc.sample(n=n, random_state=RANDOM_STATE).index
        m = TabPFNClassifier(device="cpu", ignore_pretraining_limits=True)
        m.fit(X_tr_enc.loc[idx], y_train.loc[idx])
        pred = m.predict_proba(X_te_enc)[:, 1]
        sc_auc = roc_auc_score(y_test, pred)
        scaling.append({"n_samples": n, "auc": round(sc_auc, 4)})
        print(f"    n={n}: AUC={sc_auc:.4f}")

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


# ═════════════════════════════════════════════════════════════════════════
#  FT-Transformer (manual PyTorch)
# ═════════════════════════════════════════════════════════════════════════
def run_ft_transformer(X_train, X_test, y_train, y_test, X_full, y_full):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    X_tr_enc, X_te_enc = encode_for_numeric(X_train, X_test)
    n_features = X_tr_enc.shape[1]

    class FTTransformer(nn.Module):
        def __init__(self, n_feat, d_tok=64, n_heads=4, n_layers=3, d_ffn=128):
            super().__init__()
            self.embeddings = nn.ModuleList([nn.Linear(1, d_tok) for _ in range(n_feat)])
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_tok))
            layer = nn.TransformerEncoderLayer(
                d_model=d_tok, nhead=n_heads, dim_feedforward=d_ffn,
                dropout=0.1, batch_first=True
            )
            self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
            self.head = nn.Linear(d_tok, 1)

        def forward(self, x):
            tokens = torch.stack([emb(x[:, i:i+1]) for i, emb in enumerate(self.embeddings)], dim=1)
            cls = self.cls_token.expand(x.size(0), -1, -1)
            tokens = torch.cat([cls, tokens], dim=1)
            out = self.transformer(tokens)
            return self.head(out[:, 0]).squeeze(-1)

    def train_model(X, y):
        Xt = torch.tensor(X.values, dtype=torch.float32)
        yt = torch.tensor(y.values, dtype=torch.float32)
        dl = DataLoader(TensorDataset(Xt, yt), batch_size=256, shuffle=True)

        model = FTTransformer(X.shape[1])
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
        loss_fn = nn.BCEWithLogitsLoss()

        model.train()
        for epoch in range(50):
            for xb, yb in dl:
                opt.zero_grad()
                loss_fn(model(xb), yb).backward()
                opt.step()
        model.eval()
        return model

    def predict(model, X):
        Xt = torch.tensor(X.values, dtype=torch.float32)
        with torch.no_grad():
            return torch.sigmoid(model(Xt)).numpy()

    print("  FT-Transformer: fitting...")
    process = psutil.Process(os.getpid())
    rss_before = process.memory_info().rss
    t0 = time.time()
    model = train_model(X_tr_enc, y_train)
    train_time = time.time() - t0
    rss_after = process.memory_info().rss
    peak_mem_mb = max(0.0, (rss_after - rss_before) / (1024 ** 2))

    inf_start = time.time()
    y_prob = predict(model, X_te_enc)
    inf_time = time.time() - inf_start
    inf_per_1k = (inf_time / len(X_te_enc)) * 1000 * 1000

    auc = roc_auc_score(y_test, y_prob)
    ll = log_loss(y_test, y_prob)

    print(f"  AUC: {auc:.4f} | Log Loss: {ll:.4f}")
    print(f"  Train: {train_time:.1f}s | Memory: {peak_mem_mb:.0f}MB | Infer: {inf_per_1k:.1f}ms/1k")

    metrics = {
        "auc_roc": round(auc, 4),
        "log_loss": round(ll, 4),
        "train_time_sec": round(train_time, 2),
        "inference_time_ms_per_1k": round(inf_per_1k, 1),
        "peak_memory_mb": round(peak_mem_mb, 1),
    }

    # CV
    print("  CV AUC...")
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    aucs = []
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_full, y_full)):
        Xtr, Xval = X_full.iloc[tr_idx], X_full.iloc[val_idx]
        Xtr_e, Xval_e = encode_for_numeric(Xtr, Xval)
        m = train_model(Xtr_e, y_full.iloc[tr_idx])
        pred = predict(m, Xval_e)
        fold_auc = roc_auc_score(y_full.iloc[val_idx], pred)
        aucs.append(fold_auc)
        print(f"    Fold {fold+1}: {fold_auc:.4f}")
    metrics["auc_roc"] = round(np.mean(aucs), 4)
    print(f"  CV AUC: {metrics['auc_roc']}")

    # Scaling
    print("  Scaling curve...")
    scaling = []
    for n in SCALING_SIZES:
        if n > len(X_tr_enc):
            break
        idx = X_tr_enc.sample(n=n, random_state=RANDOM_STATE).index
        m = train_model(X_tr_enc.loc[idx], y_train.loc[idx])
        pred = predict(m, X_te_enc)
        sc_auc = roc_auc_score(y_test, pred)
        scaling.append({"n_samples": n, "auc": round(sc_auc, 4)})
        print(f"    n={n}: AUC={sc_auc:.4f}")

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


# ═════════════════════════════════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════════════════════════════════
def main():
    # Which models to run (pass names as args, or run all remaining)
    to_run = sys.argv[1:] if len(sys.argv) > 1 else ["autogluon", "tabpfn", "ft-transformer"]

    df = load_data()
    X = df.drop(columns=["Churn"])
    y = df["Churn"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
    print(f"Train: {len(X_train)} | Test: {len(X_test)}")

    # Load existing results
    with open(OUTPUT_PATH) as f:
        results = json.load(f)
    existing_names = {m["name"] for m in results["models"]}
    print(f"Existing models: {existing_names}")

    runners = {
        "autogluon": ("AutoGluon", run_autogluon),
        "tabpfn": ("TabPFN", run_tabpfn),
        "ft-transformer": ("FT-Transformer", run_ft_transformer),
    }

    for key in to_run:
        name, runner = runners[key]
        if name in existing_names:
            print(f"\n  Skipping {name} (already in results)")
            continue
        print(f"\n{'='*60}")
        print(f"  Running: {name}")
        print(f"{'='*60}")
        try:
            result = runner(X_train, X_test, y_train, y_test, X, y)
            results["models"].append(result)
            with open(OUTPUT_PATH, "w") as f:
                json.dump(results, f, indent=2)
            print(f"  >> Saved ({len(results['models'])} models total)")
        except Exception as e:
            print(f"\n  FAILED: {name} — {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()

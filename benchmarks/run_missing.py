"""
Run only the models NOT yet in churn_results.json, appending incrementally.
"""
import json
import sys
from pathlib import Path

# Reuse everything from run_churn
from run_churn import (
    load_data, prepare_splits, OUTPUT_PATH,
    run_xgb_default, run_xgb_tuned,
    run_autogluon, run_tabpfn, run_ft_transformer,
)

ALL_RUNNERS = [
    ("XGBoost (Default)", run_xgb_default),
    ("XGBoost (Tuned)", run_xgb_tuned),
    ("AutoGluon", run_autogluon),
    ("TabPFN", run_tabpfn),
    ("FT-Transformer", run_ft_transformer),
]


def main():
    df, _ = load_data()
    X_train, X_test, y_train, y_test = prepare_splits(df)
    X_full = df.drop(columns=["Churn"])
    y_full = df["Churn"]

    with open(OUTPUT_PATH) as f:
        results = json.load(f)

    existing = {m["name"] for m in results["models"]}
    print(f"Already have: {existing}")

    for name, runner in ALL_RUNNERS:
        if name in existing:
            print(f"\n  Skipping {name} (already in results)")
            continue
        print(f"\n{'='*60}")
        print(f"  Running: {name}")
        print(f"{'='*60}")
        try:
            result = runner(X_train, X_test, y_train, y_test, X_full, y_full)
            results["models"].append(result)
            with open(OUTPUT_PATH, "w") as f:
                json.dump(results, f, indent=2)
            print(f"  >> Saved ({len(results['models'])} models total)")
        except Exception as e:
            print(f"\n  FAILED: {name} — {e}")
            import traceback
            traceback.print_exc()

    print(f"\nDone — {len(results['models'])} models in {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

# Benchmark Strategy: "Can LLM-style models read tabular data?"

This project compares classic tabular ML vs foundation/deep tabular models under the same protocol.

## Core Question

Can foundation-model-style approaches (TabPFN) and deep tabular models (FT-Transformer) match or beat boosted trees on:

- small mixed-feature business data (Churn),
- medium-large multi-table credit risk data (Home Credit),
- large highly-imbalanced fraud data (IEEE-CIS)?

## Model Use-Case Map

| Model | Typical Strength | Typical Weakness |
|---|---|---|
| LightGBM | Fast on large tabular, strong with missing values | Needs categorical encoding |
| CatBoost | Best with many categorical features, strong defaults | Slower than LightGBM/XGBoost at scale |
| XGBoost | Very strong tuned performance, strong for fraud/risk tasks | Requires careful tuning/encoding |
| AutoGluon | Strong out-of-box ensembles on tabular | High memory/time cost |
| TabPFN | Strong on small/medium classification, minimal tuning | Scaling/inference constraints on very large data |
| FT-Transformer | Can improve with lots of data + GPU tuning | Can underperform trees without heavy tuning |

## Dataset-Specific Benchmark Expectations

| Dataset | Data Regime | Expected Top Tier | LLM/Foundation Expectation |
|---|---|---|---|
| Churn (7k rows, mixed categorical) | Small tabular | AutoGluon / CatBoost / strong trees | TabPFN should be competitive, near top but not always #1 |
| Home Credit (307k rows, 300+ features) | Large engineered tabular | Tuned XGBoost/LightGBM + AutoGluon | TabPFN should be respectable but usually behind tuned trees |
| IEEE Fraud (590k rows, high imbalance) | Very large + imbalanced | Tuned XGBoost/LightGBM/CatBoost | Foundation model should lag top tuned GBDTs at this scale |

## Benchmarking Rules Used

- Same dataset split policy and metrics across models.
- Primary metric: AUC-ROC.
- Secondary metrics: log loss, train time, inference latency, memory.
- Scaling curves are required to show small-data vs large-data behavior.
- Final interpretation focuses on *accuracy vs effort* (not accuracy alone).

## How to Read Results

- If TabPFN wins or is near-tied on Churn: strong evidence for "LLM-style tabular reading" in low-data settings.
- If tuned trees dominate Home Credit/Fraud: evidence that classic GBDTs still lead at scale.
- If AutoGluon wins but at high cost: evidence that ensembling can buy accuracy with runtime/memory tradeoffs.

"""Tab 5 — Final Verdict."""

import json
from pathlib import Path

import streamlit as st


def _load_results():
    results_dir = Path(__file__).parent.parent / "results"
    out = {}
    for f in results_dir.glob("*_results.json"):
        with open(f) as fh:
            d = json.load(fh)
            out[d["dataset"]] = d
    return out


def _best_model(ds):
    return sorted(ds["models"], key=lambda m: m["metrics"]["auc_roc"], reverse=True)[0]


def _fastest_model(results):
    all_models = [m for ds in results.values() for m in ds["models"]]
    return sorted(all_models, key=lambda m: m["metrics"]["train_time_sec"])[0]


def _best_overall(results):
    all_models = [m for ds in results.values() for m in ds["models"]]
    return sorted(all_models, key=lambda m: m["metrics"]["auc_roc"], reverse=True)[0]


def _avg_auc(ds, cat):
    rows = [m["metrics"]["auc_roc"] for m in ds["models"] if m.get("category") == cat]
    if not rows:
        return 0.0
    return sum(rows) / len(rows)


def render():
    results = _load_results()
    if not results:
        st.markdown(
            '<div style="text-align:center;padding:3rem;color:#5F6368;">No results available yet.</div>',
            unsafe_allow_html=True,
        )
        return

    if not {"telco_churn", "home_credit", "ieee_fraud"}.issubset(set(results.keys())):
        st.markdown(
            '<div style="text-align:center;padding:3rem;color:#5F6368;">Verdict unlocks once churn, credit, and fraud results are present.</div>',
            unsafe_allow_html=True,
        )
        return

    churn = results["telco_churn"]
    credit = results["home_credit"]
    fraud = results["ieee_fraud"]

    churn_best = _best_model(churn)
    credit_best = _best_model(credit)
    fraud_best = _best_model(fraud)
    fastest = _fastest_model(results)
    best_overall = _best_overall(results)

    st.markdown(
        '<div class="card" style="margin-bottom:12px;">'
        '<div style="font-size:1.15rem;font-weight:700;color:#E8EAED;margin-bottom:8px;">Final Verdict</div>'
        '<p style="font-size:0.9rem;color:#BDC1C6;line-height:1.7;margin:0;">'
        'Foundation-style tabular models are competitive, especially on small and medium settings. '
        'But across larger, imbalanced production-like data, tuned gradient boosting still provides the '
        'most consistent accuracy-to-latency tradeoff.'
        '</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    def mc(lbl, val, detail):
        return (
            f'<div class="metric-card">'
            f'<div class="metric-label">{lbl}</div>'
            f'<div class="metric-value">{val}</div>'
            f'<div class="metric-detail">{detail}</div>'
            f"</div>"
        )

    st.markdown(
        '<div class="bento bento-3" style="margin:1rem 0 1.8rem 0;">'
        + mc("Highest AUC observed", f'{best_overall["metrics"]["auc_roc"]:.4f}', best_overall["name"])
        + mc("Fastest Training", f'{fastest["metrics"]["train_time_sec"]:.2f}s', fastest["name"])
        + mc("Core Answer", "Partial Yes", "LLM-style models compete, trees still dominate at scale")
        + "</div>",
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-header">Dataset Winners</div>', unsafe_allow_html=True)
    winners_rows = (
        f'<tr><td class="model-name">Churn</td><td>{churn_best["name"]}</td><td class="r">{churn_best["metrics"]["auc_roc"]:.4f}</td></tr>'
        f'<tr><td class="model-name">Credit</td><td>{credit_best["name"]}</td><td class="r">{credit_best["metrics"]["auc_roc"]:.4f}</td></tr>'
        f'<tr><td class="model-name">Fraud</td><td>{fraud_best["name"]}</td><td class="r">{fraud_best["metrics"]["auc_roc"]:.4f}</td></tr>'
    )
    st.markdown(
        '<div class="card" style="padding:0;overflow:hidden;">'
        '<table class="lb"><thead><tr>'
        '<th>Dataset</th><th>Top Model</th><th class="r">AUC-ROC</th>'
        f'</tr></thead><tbody>{winners_rows}</tbody></table></div>',
        unsafe_allow_html=True,
    )

    st.markdown('<hr class="soft-divider">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">Family-Level Summary</div>', unsafe_allow_html=True)

    cats = [
        ("gradient_boosting", "Gradient Boosting", "Most robust across all scales"),
        ("automl", "AutoML", "Strong AUC, highest runtime/memory"),
        ("foundation_model", "Foundation Model", "Competitive on smaller data, weaker at scale"),
        ("deep_learning", "Deep Learning", "Improves with scale/tuning, less consistent here"),
    ]
    rows = ""
    for key, label, note in cats:
        rows += (
            "<tr>"
            + f'<td class="model-name">{label}</td>'
            + f'<td class="r">{_avg_auc(churn, key):.4f}</td>'
            + f'<td class="r">{_avg_auc(credit, key):.4f}</td>'
            + f'<td class="r">{_avg_auc(fraud, key):.4f}</td>'
            + f"<td>{note}</td>"
            + "</tr>"
        )

    st.markdown(
        '<div class="card" style="padding:0;overflow:hidden;">'
        '<table class="lb"><thead><tr>'
        '<th>Family</th><th class="r">Churn</th><th class="r">Credit</th><th class="r">Fraud</th><th>Verdict</th>'
        f'</tr></thead><tbody>{rows}</tbody></table></div>',
        unsafe_allow_html=True,
    )

"""Tab 1 — Overview."""

import json
import streamlit as st
from pathlib import Path
from config import DATASET_INFO, CATEGORY_COLORS, CATEGORY_BG, MODEL_COLORS


def _load_all_results():
    results_dir = Path(__file__).parent.parent / "results"
    results = {}
    for f in results_dir.glob("*_results.json"):
        with open(f) as fh:
            data = json.load(fh)
            results[data["dataset"]] = data
    return results


def _best(results, metric, minimize=False):
    best_val = float("inf") if minimize else float("-inf")
    best_name = "---"
    for ds in results.values():
        for m in ds["models"]:
            v = m["metrics"].get(metric)
            if v is not None and ((minimize and v < best_val) or (not minimize and v > best_val)):
                best_val, best_name = v, m["name"]
    return best_name, best_val


def render():
    # Hero
    st.markdown(
        '<div style="padding:0.8rem 0 1.4rem 0;">'
        '<div class="hero-title">Tabular Arena ⚔️</div>'
        '<div class="hero-sub">Can foundation-style tabular models really replace a tuned classical ML stack? We evaluate that claim on three datasets that scale from small business tabular to large imbalanced fraud.</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    # The Question
    st.markdown(
        '<div class="card" style="margin-bottom:24px;">'
        '<p style="font-size:0.88rem;line-height:1.75;color:#BDC1C6;margin:0;">'
        'The central test is not just raw AUC. It is whether an <strong style="color:#FF7597">LLM-style tabular model</strong> can stay competitive '
        'as data regime changes from <strong style="color:#E8EAED">small</strong> to <strong style="color:#E8EAED">very large + imbalanced</strong>. '
        'We compare open alternatives: '
        '<strong style="color:#FF7597">TabPFN</strong> (foundation model) and '
        '<strong style="color:#FFB74D">FT-Transformer</strong> (deep learning) against '
        '<strong style="color:#03DAC6">LightGBM</strong>, '
        '<strong style="color:#03DAC6">CatBoost</strong>, '
        '<strong style="color:#03DAC6">XGBoost</strong>, and '
        '<strong style="color:#BB86FC">AutoGluon</strong>. '
        'Same data, same splits, same metrics. Two modes: '
        '<strong style="color:#E8EAED">zero-effort</strong> (defaults) and '
        '<strong style="color:#E8EAED">tuned</strong> (50 Optuna trials).'
        '</p></div>',
        unsafe_allow_html=True,
    )

    # Datasets
    ds_items = ""
    for key, info in DATASET_INFO.items():
        ds_items += (
            f'<div style="margin-bottom:12px;">'
            f'<div style="font-weight:600;color:#E8EAED;font-size:0.85rem;">{info["icon"]} {info["short_name"]}</div>'
            f'<div style="color:#9AA0A6;font-size:0.78rem;margin-top:2px;">'
            f'{info["n_samples"]:,} rows · {info["n_features"]} features · {info["target_rate"]:.0%} positive · {info["source"]}'
            f'</div></div>'
        )

    # Models
    cat_models = {
        "Gradient Boosting": "LightGBM, CatBoost",
        "AutoML": "AutoGluon",
        "Foundation Model": "TabPFN",
        "Deep Learning": "FT-Transformer",
    }
    model_items = ""
    for cat, names in cat_models.items():
        fg = CATEGORY_COLORS[cat]
        bg = CATEGORY_BG[cat]
        model_items += (
            f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:10px;">'
            f'<span class="cat-tag" style="background:{bg};color:{fg};">{cat}</span>'
            f'<span style="color:#BDC1C6;font-size:0.8rem;">{names}</span>'
            f'</div>'
        )

    lbl = "font-size:0.68rem;font-weight:600;text-transform:uppercase;letter-spacing:0.08em;color:#9AA0A6;margin-bottom:14px;"
    st.markdown(
        f'<div class="bento bento-2" style="margin-bottom:10px;">'
        f'<div class="card"><div style="{lbl}">Datasets</div>{ds_items}</div>'
        f'<div class="card"><div style="{lbl}">Models</div>{model_items}</div>'
        f'</div>'
        f'<div class="card" style="padding:14px 24px;">'
        f'<span style="font-size:0.68rem;font-weight:600;text-transform:uppercase;letter-spacing:0.08em;color:#9AA0A6;margin-right:12px;">Protocol</span>'
        f'<span style="color:#BDC1C6;font-size:0.82rem;">5-fold stratified CV · 80/20 split (seed 42) · Wall-clock timing · tracemalloc · Scaling at 500–5,600 samples</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Use-case matrix
    st.markdown(
        '<div class="card" style="margin-top:12px;">'
        '<div style="font-size:0.68rem;font-weight:600;text-transform:uppercase;letter-spacing:0.08em;color:#9AA0A6;margin-bottom:14px;">Model Use-Case Lens</div>'
        '<table class="lb"><thead><tr>'
        '<th>Model Family</th><th>Best Fit</th><th>Risk</th>'
        '</tr></thead><tbody>'
        '<tr><td class="model-name">Gradient Boosting</td><td>Large/structured tabular, strong production baselines</td><td class="worst">Needs careful tuning/encoding</td></tr>'
        '<tr><td class="model-name">AutoML (AutoGluon)</td><td>Fast path to high accuracy</td><td class="worst">High memory/time overhead</td></tr>'
        '<tr><td class="model-name">Foundation Model (TabPFN)</td><td>Strong low-data signal extraction</td><td class="worst">Scaling/inference constraints on very large data</td></tr>'
        '<tr><td class="model-name">Deep Learning (FT-Transformer)</td><td>Can improve with data + tuning</td><td class="worst">Often less reliable than tuned trees on tabular</td></tr>'
        '</tbody></table>'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<hr class="soft-divider">', unsafe_allow_html=True)

    # Results
    results = _load_all_results()
    if not results:
        st.markdown('<div style="text-align:center;padding:2rem;color:#5F6368;">No results yet — run benchmarks first.</div>', unsafe_allow_html=True)
        return

    st.markdown('<div class="section-header">Results at a Glance</div>', unsafe_allow_html=True)

    best_auc_name, best_auc_val = _best(results, "auc_roc")
    best_speed_name, best_speed_val = _best(results, "train_time_sec", minimize=True)
    best_mem_name, best_mem_val = _best(results, "peak_memory_mb", minimize=True)

    best_zs_name, best_zs_val = "---", 0.0
    for ds in results.values():
        for m in ds["models"]:
            if not m.get("tuned", False) and m["metrics"]["auc_roc"] > best_zs_val:
                best_zs_val, best_zs_name = m["metrics"]["auc_roc"], m["name"]

    def mc(label, value, detail):
        return (
            f'<div class="metric-card">'
            f'<div class="metric-label">{label}</div>'
            f'<div class="metric-value">{value}</div>'
            f'<div class="metric-detail">{detail}</div>'
            f'</div>'
        )

    st.markdown(
        '<div class="bento bento-4" style="margin-top:12px;">'
        + mc("Best AUC", f"{best_auc_val:.4f}", best_auc_name)
        + mc("Fastest", f"{best_speed_val:.1f}s", best_speed_name)
        + mc("Lightest", f'{best_mem_val:.0f}<span style="font-size:0.9rem;color:#5F6368;">MB</span>', best_mem_name)
        + mc("Best Zero-Shot", f"{best_zs_val:.4f}", best_zs_name)
        + '</div>',
        unsafe_allow_html=True,
    )

    hw = next(iter(results.values())).get("hardware", {})
    if hw:
        items = " · ".join(str(v) for v in hw.values())
        st.markdown(
            f'<div style="margin-top:16px;padding:10px 18px;background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.06);border-radius:8px;color:#5F6368;font-size:0.78rem;text-align:center;">{items}</div>',
            unsafe_allow_html=True,
        )

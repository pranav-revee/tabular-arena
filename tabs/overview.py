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
    # ── Hero ────────────────────────────────────────────────────────────
    st.markdown('<div style="padding:1.5rem 0 2rem 0;"><div class="hero-title">Tabular Arena</div><div class="hero-sub">Foundation models claim to replace traditional ML on tabular data.<br>We benchmarked the open-source alternatives to find out.</div></div>', unsafe_allow_html=True)

    # ── The Question ────────────────────────────────────────────────────
    st.markdown('<div class="card" style="margin-bottom:24px;"><p style="font-size:0.88rem;line-height:1.75;color:#94a3b8;margin:0;">Google\'s <strong style="color:#e2e8f0">NEXUS</strong> promises zero-shot tabular prediction &mdash; but it\'s closed-source. We test what\'s actually available: <strong style="color:#e2e8f0">TabPFN</strong> and <strong style="color:#e2e8f0">FT-Transformer</strong> against <strong style="color:#e2e8f0">LightGBM</strong>, <strong style="color:#e2e8f0">CatBoost</strong>, and <strong style="color:#e2e8f0">AutoGluon</strong>. Same data, same splits, same metrics. Two modes: <strong style="color:#e2e8f0">zero-effort</strong> (defaults) and <strong style="color:#e2e8f0">tuned</strong> (50 Optuna trials).</p></div>', unsafe_allow_html=True)

    # ── Datasets + Models ───────────────────────────────────────────────
    ds_items = []
    for info in DATASET_INFO.values():
        ds_items.append(
            f'<div style="margin-bottom:12px;">'
            f'<div style="font-weight:600;color:#e2e8f0;font-size:0.85rem;">{info["short_name"]}</div>'
            f'<div style="color:#64748b;font-size:0.78rem;margin-top:2px;">'
            f'{info["n_samples"]:,} rows · {info["n_features"]} features · {info["target_rate"]:.0%} positive'
            f'</div></div>'
        )
    ds_html = "".join(ds_items)

    model_items = []
    cat_models = {
        "Gradient Boosting": "LightGBM, CatBoost",
        "AutoML": "AutoGluon",
        "Foundation Model": "TabPFN",
        "Deep Learning": "FT-Transformer",
    }
    for cat, names in cat_models.items():
        fg = CATEGORY_COLORS[cat]
        bg = CATEGORY_BG[cat]
        model_items.append(
            f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:10px;">'
            f'<span class="cat-tag" style="background:{bg};color:{fg};">{cat}</span>'
            f'<span style="color:#94a3b8;font-size:0.8rem;">{names}</span>'
            f'</div>'
        )
    model_html = "".join(model_items)

    label_style = "font-size:0.68rem;font-weight:600;text-transform:uppercase;letter-spacing:0.08em;color:#64748b;margin-bottom:14px;"

    html = (
        f'<div class="bento bento-2" style="margin-bottom:10px;">'
        f'<div class="card"><div style="{label_style}">Datasets</div>{ds_html}</div>'
        f'<div class="card"><div style="{label_style}">Models</div>{model_html}</div>'
        f'</div>'
        f'<div class="card" style="padding:14px 24px;">'
        f'<span style="font-size:0.68rem;font-weight:600;text-transform:uppercase;letter-spacing:0.08em;color:#64748b;margin-right:12px;">Protocol</span>'
        f'<span style="color:#94a3b8;font-size:0.82rem;">5-fold stratified CV · 80/20 split (seed 42) · Wall-clock timing · tracemalloc · Scaling at 500–5,600 samples</span>'
        f'</div>'
    )
    st.markdown(html, unsafe_allow_html=True)

    st.markdown('<hr class="soft-divider">', unsafe_allow_html=True)

    # ── Results ─────────────────────────────────────────────────────────
    results = _load_all_results()
    if not results:
        st.markdown('<div style="text-align:center;padding:2rem;color:#64748b;">No results yet.</div>', unsafe_allow_html=True)
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

    def metric_card(label, value, detail):
        return (
            f'<div class="metric-card">'
            f'<div class="metric-label">{label}</div>'
            f'<div class="metric-value">{value}</div>'
            f'<div class="metric-detail">{detail}</div>'
            f'</div>'
        )

    cards = (
        f'<div class="bento bento-4" style="margin-top:12px;">'
        + metric_card("Best AUC", f"{best_auc_val:.4f}", best_auc_name)
        + metric_card("Fastest", f"{best_speed_val:.1f}s", best_speed_name)
        + metric_card("Lightest", f'{best_mem_val:.0f}<span style="font-size:0.9rem;color:#64748b;">MB</span>', best_mem_name)
        + metric_card("Best Zero-Shot", f"{best_zs_val:.4f}", best_zs_name)
        + '</div>'
    )
    st.markdown(cards, unsafe_allow_html=True)

    hw = next(iter(results.values())).get("hardware", {})
    if hw:
        items = " · ".join(str(v) for v in hw.values())
        st.markdown(
            f'<div style="margin-top:16px;padding:10px 18px;background:rgba(255,255,255,0.04);'
            f'border:1px solid rgba(255,255,255,0.08);border-radius:8px;'
            f'color:#64748b;font-size:0.78rem;text-align:center;">{items}</div>',
            unsafe_allow_html=True,
        )

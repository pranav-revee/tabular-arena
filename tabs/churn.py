"""Tab 2 — Churn Prediction Benchmarks."""

import json
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from config import MODEL_COLORS, CATEGORY_COLORS, CATEGORY_BG, MODEL_CATEGORIES, DEFAULT_MODELS, TUNED_MODELS, PLOTLY_LAYOUT

RESULTS_PATH = Path(__file__).parent.parent / "results" / "churn_results.json"


@st.cache_data
def load_results():
    if not RESULTS_PATH.exists():
        return None
    with open(RESULTS_PATH) as f:
        return json.load(f)


def _to_df(models):
    rows = []
    for m in models:
        rows.append({
            "Model":             m["name"],
            "Category":          MODEL_CATEGORIES.get(m["name"], ""),
            "AUC-ROC":           m["metrics"]["auc_roc"],
            "Log Loss":          m["metrics"]["log_loss"],
            "Train Time (s)":    m["metrics"]["train_time_sec"],
            "Memory (MB)":       m["metrics"]["peak_memory_mb"],
            "Inference (ms/1k)": m["metrics"]["inference_time_ms_per_1k"],
        })
    return pd.DataFrame(rows).sort_values("AUC-ROC", ascending=False).reset_index(drop=True)


def _filter(data, tuned):
    names = TUNED_MODELS if tuned else DEFAULT_MODELS
    return [m for m in data["models"] if m["name"] in names]


def _leaderboard(df):
    cols = {
        "AUC-ROC": ("{:.4f}", False),
        "Log Loss": ("{:.4f}", True),
        "Train Time (s)": ("{:.2f}s", True),
        "Memory (MB)": ("{:.0f}", True),
        "Inference (ms/1k)": ("{:.1f}", True),
    }
    best, worst = {}, {}
    for c, (_, lower) in cols.items():
        best[c] = df[c].min() if lower else df[c].max()
        worst[c] = df[c].max() if lower else df[c].min()

    rows = ""
    for i, r in df.iterrows():
        rank = i + 1
        rcls = {1: "rk1", 2: "rk2", 3: "rk3"}.get(rank, "rkn")
        color = MODEL_COLORS.get(r["Model"], "#999")
        cat_color = CATEGORY_COLORS.get(r["Category"], "#999")

        cells = ""
        for c, (fmt, _) in cols.items():
            cls = "best" if r[c] == best[c] else ("worst" if r[c] == worst[c] else "")
            wcol = ' style="color:#475569;"' if cls == "worst" else ""
            cells += f'<td class="r {cls}"{wcol}>{fmt.format(r[c])}</td>'

        rows += f"""<tr>
<td><span class="rank-badge {rcls}">{rank}</span></td>
<td><div style="display:flex;align-items:center;gap:10px;">
    <div style="width:3px;height:22px;border-radius:2px;background:{color};"></div>
    <div><div class="model-name" style="color:#f8fafc;">{r['Model']}</div>
    <div style="font-size:0.7rem;color:{cat_color};">{r['Category']}</div></div>
</div></td>
{cells}
</tr>"""

    return f"""<div class="card" style="padding:0;overflow:hidden;">
<table class="lb"><thead><tr>
    <th style="width:44px">#</th><th>Model</th>
    <th class="r">AUC-ROC</th><th class="r">Log Loss</th>
    <th class="r">Time</th><th class="r">Memory</th><th class="r">Infer</th>
</tr></thead><tbody>{rows}</tbody></table></div>"""


def _handling(models, df):
    badge = {
        "native":            ("Native", "pill-green"),
        "needs_encoding":    ("Needs Encoding", "pill-yellow"),
        "needs_imputation":  ("Needs Imputation", "pill-red"),
        "scale_pos_weight":  ("scale_pos_weight", "pill-blue"),
        "auto_class_weights": ("Auto Weights", "pill-blue"),
        "none":              ("None", "pill-gray"),
    }
    order = {n: i for i, n in enumerate(df["Model"])}
    models = sorted(models, key=lambda m: order.get(m["name"], 99))

    rows = ""
    for m in models:
        h = m.get("raw_data_handling", {})
        color = MODEL_COLORS.get(m["name"], "#999")

        def pill(k):
            lbl, cls = badge.get(h.get(k, ""), ("?", "pill-gray"))
            return f'<span class="pill {cls}">{lbl}</span>'

        rows += f"""<tr>
<td><div style="display:flex;align-items:center;gap:8px;">
    <div style="width:3px;height:18px;border-radius:2px;background:{color};"></div>
    <span class="model-name" style="font-size:0.83rem;color:#e2e8f0;">{m['name']}</span>
</div></td>
<td>{pill('missing_values')}</td>
<td>{pill('categorical_features')}</td>
<td>{pill('class_imbalance')}</td>
</tr>"""

    return f"""<div class="card" style="padding:0;overflow:hidden;">
<table class="lb"><thead><tr>
    <th>Model</th><th>Missing Values</th><th>Categoricals</th><th>Class Imbalance</th>
</tr></thead><tbody>{rows}</tbody></table></div>"""


def render():
    data = load_results()
    if data is None:
        st.markdown('<div style="text-align:center;padding:4rem;color:#64748b;">No results yet.</div>', unsafe_allow_html=True)
        return

    # Toggle
    c1, c2 = st.columns([1, 4])
    with c1:
        mode = st.toggle("Tuned", value=False, help="Zero-effort defaults vs Optuna-tuned")
    models = _filter(data, tuned=mode)
    df = _to_df(models)
    label = "Tuned" if mode else "Zero Effort"

    # Metrics
    top = df.iloc[0]
    fast = df.loc[df["Train Time (s)"].idxmin()]
    light = df.loc[df["Memory (MB)"].idxmin()]

    st.markdown(f"""
<div class="bento bento-3" style="margin:1rem 0 2rem 0;">
    <div class="metric-card">
        <div class="metric-label">Best AUC · {label}</div>
        <div class="metric-value">{top['AUC-ROC']:.4f}</div>
        <div class="metric-detail">{top['Model']}</div>
    </div>
    <div class="metric-card">
        <div class="metric-label">Fastest</div>
        <div class="metric-value">{fast['Train Time (s)']:.1f}s</div>
        <div class="metric-detail">{fast['Model']}</div>
    </div>
    <div class="metric-card">
        <div class="metric-label">Lightest</div>
        <div class="metric-value">{light['Memory (MB)']:.0f}<span style="font-size:0.9rem;color:#9ca3af;">MB</span></div>
        <div class="metric-detail">{light['Model']}</div>
    </div>
</div>
""", unsafe_allow_html=True)

    # Leaderboard
    st.markdown('<div class="section-header">Leaderboard</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="section-caption">Ranked by AUC-ROC · {label} mode</div>', unsafe_allow_html=True)
    st.markdown(_leaderboard(df), unsafe_allow_html=True)

    st.markdown('<hr class="soft-divider">', unsafe_allow_html=True)

    # Money Chart
    st.markdown('<div class="section-header">Accuracy vs Speed</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-caption">What\'s accurate AND fast? Bottom-right is the sweet spot.</div>', unsafe_allow_html=True)

    fig = px.scatter(
        df, x="Train Time (s)", y="AUC-ROC", size="Memory (MB)",
        color="Category", color_discrete_map=CATEGORY_COLORS,
        hover_name="Model",
        hover_data={"Train Time (s)": ":.2f", "AUC-ROC": ":.4f", "Memory (MB)": ":.0f", "Category": False},
        log_x=True, size_max=50,
    )
    fig.update_layout(**PLOTLY_LAYOUT, height=460,
                      xaxis_title="Training Time (log scale)",
                      yaxis_title="AUC-ROC")
    fig.update_traces(marker=dict(line=dict(width=1.5, color="#fff")))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<hr class="soft-divider">', unsafe_allow_html=True)

    # Scaling
    st.markdown('<div class="section-header">Data Efficiency</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-caption">AUC vs training set size &mdash; who learns faster from less data?</div>', unsafe_allow_html=True)

    sc_rows = []
    for m in models:
        for pt in m.get("scaling", []):
            sc_rows.append({"Model": m["name"], "Samples": pt["n_samples"], "AUC": pt["auc"]})

    if sc_rows:
        fig2 = go.Figure()
        for name in df["Model"]:
            md = [r for r in sc_rows if r["Model"] == name]
            if not md:
                continue
            color = MODEL_COLORS.get(name, "#999")
            fig2.add_trace(go.Scatter(
                x=[r["Samples"] for r in md], y=[r["AUC"] for r in md],
                name=name, mode="lines+markers",
                line=dict(color=color, width=2.5),
                marker=dict(size=6, color=color, line=dict(width=1.5, color="#fff")),
                hovertemplate=f"<b>{name}</b><br>%{{x:,}} samples<br>AUC: %{{y:.4f}}<extra></extra>",
            ))
        fig2.update_layout(**PLOTLY_LAYOUT, height=420,
                           xaxis_title="Training Samples", yaxis_title="AUC-ROC")
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown('<hr class="soft-divider">', unsafe_allow_html=True)

    # Data handling
    st.markdown('<div class="section-header">Raw Data Handling</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-caption">Can it handle messy data out of the box?</div>', unsafe_allow_html=True)
    st.markdown(_handling(models, df), unsafe_allow_html=True)

    st.markdown('<hr class="soft-divider">', unsafe_allow_html=True)

    # Takeaways
    st.markdown('<div class="section-header">Takeaways</div>', unsafe_allow_html=True)

    best_name = df.iloc[0]["Model"]
    best_auc = df.iloc[0]["AUC-ROC"]
    fast_name = fast["Model"]
    fast_time = fast["Train Time (s)"]

    fm = df[df["Category"] == "Foundation Model"]
    gb = df[df["Category"] == "Gradient Boosting"]
    fm_auc = fm["AUC-ROC"].max() if len(fm) else 0
    gb_auc = gb["AUC-ROC"].max() if len(gb) else 0

    accent = ["#a855f7", "#3b82f6", "#00d4aa", "#f59e0b"]
    points = [
        f"<strong>{best_name}</strong> leads at <strong>{best_auc:.4f}</strong> AUC.",
        f"<strong>{fast_name}</strong> trains in <strong>{fast_time:.1f}s</strong>.",
    ]
    if fm_auc >= gb_auc:
        points.append("Foundation models <strong>match or beat</strong> gradient boosting at zero effort.")
    else:
        points.append(f"Gradient boosting still leads by <strong>{gb_auc - fm_auc:.4f}</strong> AUC.")

    if sc_rows:
        df_small = pd.DataFrame(sc_rows)
        smallest = df_small[df_small["Samples"] == df_small["Samples"].min()]
        if len(smallest):
            bs = smallest.loc[smallest["AUC"].idxmax()]
            points.append(f"<strong>{bs['Model']}</strong> hits {bs['AUC']:.4f} AUC with just {int(bs['Samples'])} samples.")

    html = '<style>.takeaway strong{color:#e2e8f0;}</style>'
    for i, p in enumerate(points):
        html += f'<div class="takeaway" style="background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.08);border-left:3px solid {accent[i % 4]};color:#94a3b8;">{p}</div>'
    st.markdown(html, unsafe_allow_html=True)

    # Download
    st.markdown('<div style="margin-top:1.5rem;">', unsafe_allow_html=True)
    with open(RESULTS_PATH) as f:
        st.download_button("Download Results JSON", f.read(), file_name="churn_results.json", mime="application/json")
    st.markdown('</div>', unsafe_allow_html=True)

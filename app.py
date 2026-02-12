import streamlit as st

st.set_page_config(
    page_title="Tabular Arena",
    page_icon="⚔️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Vibrant Dark Design System ──────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
    }

    /* ── Dark gradient background ── */
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #1a1a3e, #0f1624);
        color: #e2e8f0;
    }
    .block-container {
        padding: 2rem 3rem 3rem 3rem;
        max-width: 1200px;
    }

    #MainMenu, footer, header { visibility: hidden; }
    .stDeployButton { display: none; }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 4px;
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
    }
    .stTabs [data-baseweb="tab"] {
        padding: 9px 22px;
        font-weight: 500;
        font-size: 0.84rem;
        color: rgba(226, 232, 240, 0.5);
        border-radius: 8px;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        border-bottom: 2px solid transparent;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #e2e8f0;
        background: rgba(255, 255, 255, 0.06);
    }
    .stTabs [aria-selected="true"] {
        background: rgba(255, 255, 255, 0.08) !important;
        color: #f8fafc !important;
        font-weight: 600;
        border-bottom: 2px solid transparent !important;
        border-image: linear-gradient(90deg, #3b82f6, #ec4899, #06b6d4) 1 !important;
    }
    .stTabs [data-baseweb="tab-highlight"],
    .stTabs [data-baseweb="tab-border"] { display: none; }

    /* ── Glass Cards ── */
    .card {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 24px;
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .card:hover {
        background: rgba(255, 255, 255, 0.06);
        border-color: rgba(255, 255, 255, 0.12);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }

    /* ── Metric Cards with vibrant left borders ── */
    .metric-card {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 22px 24px;
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border-left: 3px solid;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .metric-card:hover {
        background: rgba(255, 255, 255, 0.07);
        transform: translateY(-2px);
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.3);
    }
    .metric-card:nth-child(4n+1) { border-left-color: #3b82f6; }
    .metric-card:nth-child(4n+2) { border-left-color: #ec4899; }
    .metric-card:nth-child(4n+3) { border-left-color: #06b6d4; }
    .metric-card:nth-child(4n)   { border-left-color: #f59e0b; }
    .metric-label {
        font-size: 0.72rem;
        font-weight: 600;
        color: rgba(226, 232, 240, 0.5);
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 6px;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #f8fafc;
        font-family: 'JetBrains Mono', monospace;
        line-height: 1.2;
    }
    .metric-detail {
        font-size: 0.78rem;
        color: rgba(226, 232, 240, 0.45);
        margin-top: 4px;
        font-weight: 500;
    }

    /* ── Sections ── */
    .section-header {
        font-size: 1.05rem;
        font-weight: 700;
        color: #f8fafc;
        margin: 2.5rem 0 0.3rem 0;
        letter-spacing: -0.02em;
    }
    .section-caption {
        font-size: 0.84rem;
        color: rgba(226, 232, 240, 0.45);
        margin-bottom: 1.2rem;
    }
    .soft-divider {
        border: none;
        border-top: 1px solid rgba(255, 255, 255, 0.06);
        margin: 2.5rem 0;
    }

    /* ── Gradient Hero Title ── */
    .hero-title {
        font-size: 2.8rem;
        font-weight: 800;
        letter-spacing: -0.04em;
        line-height: 1.1;
        background: linear-gradient(135deg, #3b82f6 0%, #a855f7 40%, #ec4899 70%, #06b6d4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 10px;
    }
    .hero-sub {
        font-size: 1.1rem;
        color: rgba(226, 232, 240, 0.6);
        line-height: 1.6;
        max-width: 580px;
    }

    /* ── Vibrant Pills ── */
    .pill {
        display: inline-block;
        padding: 3px 12px;
        border-radius: 100px;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.02em;
        transition: all 0.3s ease;
    }
    .pill:hover { transform: scale(1.05); }
    .pill-green  { background: rgba(16, 185, 129, 0.15); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.25); }
    .pill-yellow { background: rgba(245, 158, 11, 0.15); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.25); }
    .pill-red    { background: rgba(239, 68, 68, 0.15);  color: #f87171; border: 1px solid rgba(239, 68, 68, 0.25);  }
    .pill-blue   { background: rgba(59, 130, 246, 0.15); color: #60a5fa; border: 1px solid rgba(59, 130, 246, 0.25); }
    .pill-gray   { background: rgba(148, 163, 184, 0.1); color: #94a3b8; border: 1px solid rgba(148, 163, 184, 0.2); }

    /* ── Leaderboard ── */
    .lb { width: 100%; border-collapse: separate; border-spacing: 0; }
    .lb th {
        font-size: 0.68rem;
        font-weight: 600;
        color: rgba(226, 232, 240, 0.5);
        text-transform: uppercase;
        letter-spacing: 0.07em;
        padding: 12px 14px;
        text-align: left;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        background: rgba(255, 255, 255, 0.03);
    }
    .lb th:first-child { border-radius: 14px 0 0 0; }
    .lb th:last-child  { border-radius: 0 14px 0 0; }
    .lb th.r { text-align: right; }
    .lb td {
        padding: 14px;
        font-size: 0.85rem;
        color: #cbd5e1;
        border-bottom: 1px solid rgba(255, 255, 255, 0.04);
        transition: all 0.2s ease;
    }
    .lb td.r {
        text-align: right;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.82rem;
    }
    .lb tr:last-child td { border-bottom: none; }
    /* Alternating row tints */
    .lb tr:nth-child(even) td { background: rgba(255, 255, 255, 0.02); }
    .lb tr:nth-child(odd) td  { background: rgba(255, 255, 255, 0.00); }
    .lb tr:hover td { background: rgba(59, 130, 246, 0.08) !important; }
    .lb .best  { color: #34d399; font-weight: 600; }
    .lb .worst { color: rgba(148, 163, 184, 0.4); }
    .lb .model-name { font-weight: 600; color: #f8fafc; }

    /* ── Glowing Rank Badges ── */
    .rank-badge {
        width: 28px; height: 28px; border-radius: 8px;
        display: inline-flex; align-items: center; justify-content: center;
        font-weight: 700; font-size: 0.75rem;
        font-family: 'JetBrains Mono', monospace;
        transition: all 0.3s ease;
    }
    .rank-badge:hover { transform: scale(1.15); }
    .rk1 {
        background: rgba(255, 215, 0, 0.15);
        color: #FFD700;
        border: 1px solid rgba(255, 215, 0, 0.35);
        box-shadow: 0 0 12px rgba(255, 215, 0, 0.25), 0 0 4px rgba(255, 215, 0, 0.15);
    }
    .rk2 {
        background: rgba(192, 192, 192, 0.12);
        color: #C0C0C0;
        border: 1px solid rgba(192, 192, 192, 0.3);
        box-shadow: 0 0 10px rgba(192, 192, 192, 0.2), 0 0 4px rgba(192, 192, 192, 0.1);
    }
    .rk3 {
        background: rgba(205, 127, 50, 0.15);
        color: #CD7F32;
        border: 1px solid rgba(205, 127, 50, 0.35);
        box-shadow: 0 0 10px rgba(205, 127, 50, 0.2), 0 0 4px rgba(205, 127, 50, 0.1);
    }
    .rkn {
        background: rgba(148, 163, 184, 0.08);
        color: rgba(148, 163, 184, 0.4);
        border: 1px solid rgba(148, 163, 184, 0.1);
    }

    /* ── Takeaway ── */
    .takeaway {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-left: 3px solid;
        border-radius: 0 12px 12px 0;
        padding: 14px 18px;
        margin-bottom: 8px;
        font-size: 0.88rem;
        line-height: 1.6;
        color: rgba(226, 232, 240, 0.7);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        transition: all 0.3s ease;
    }
    .takeaway:hover {
        background: rgba(255, 255, 255, 0.06);
        border-left-width: 4px;
    }
    .takeaway strong { color: #f8fafc; }

    /* ── Cat badge ── */
    .cat-tag {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 100px;
        font-size: 0.7rem;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    .cat-tag:hover { transform: scale(1.05); }

    /* ── Bento Grid ── */
    .bento { display: grid; gap: 10px; }
    .bento-2 { grid-template-columns: 1fr 1fr; }
    .bento-4 { grid-template-columns: 1fr 1fr 1fr 1fr; }
    .bento-3 { grid-template-columns: 1fr 1fr 1fr; }

    /* ── Footer ── */
    .site-footer {
        text-align: center;
        padding: 2.5rem 0 1.5rem 0;
        margin-top: 3rem;
        border-top: 1px solid rgba(255, 255, 255, 0.06);
        font-size: 0.82rem;
        color: rgba(226, 232, 240, 0.35);
    }
    .site-footer a {
        color: rgba(226, 232, 240, 0.5);
        text-decoration: none;
        transition: all 0.3s ease;
    }
    .site-footer a:hover {
        color: #60a5fa;
        text-shadow: 0 0 8px rgba(96, 165, 250, 0.3);
    }

    /* ── Streamlit overrides ── */
    [data-testid="stMetric"] { display: none; }
    .stAlert {
        border-radius: 12px;
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.08);
        color: #e2e8f0;
    }
    div[data-testid="stExpander"] {
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        background: rgba(255, 255, 255, 0.03);
    }
    div[data-testid="stExpander"] summary {
        color: #e2e8f0;
    }
    .stDownloadButton button {
        background: rgba(255, 255, 255, 0.06);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 10px;
        color: #e2e8f0;
        font-weight: 500;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .stDownloadButton button:hover {
        background: rgba(59, 130, 246, 0.15);
        border-color: rgba(59, 130, 246, 0.3);
        color: #60a5fa;
        box-shadow: 0 4px 20px rgba(59, 130, 246, 0.2);
    }
    h1 a, h2 a, h3 a { display: none !important; }
    h1, h2, h3, h4, h5, h6 { color: #f8fafc !important; }

    /* ── Streamlit text & widget overrides for dark ── */
    .stMarkdown, .stText, p, span, li { color: #e2e8f0; }
    label, .stSelectbox label, .stMultiSelect label { color: #cbd5e1 !important; }
    .stSelectbox > div > div,
    .stMultiSelect > div > div {
        background: rgba(255, 255, 255, 0.05) !important;
        border-color: rgba(255, 255, 255, 0.1) !important;
        color: #e2e8f0 !important;
    }

    /* toggle */
    .stToggle > div { background: transparent !important; }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: rgba(255, 255, 255, 0.02); }
    ::-webkit-scrollbar-thumb {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 3px;
    }
    ::-webkit-scrollbar-thumb:hover { background: rgba(255, 255, 255, 0.2); }

    /* ── Ambient glow on page ── */
    .stApp::before {
        content: '';
        position: fixed;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle at 20% 20%, rgba(59, 130, 246, 0.04) 0%, transparent 50%),
                    radial-gradient(circle at 80% 80%, rgba(168, 85, 247, 0.04) 0%, transparent 50%),
                    radial-gradient(circle at 50% 50%, rgba(236, 72, 153, 0.02) 0%, transparent 50%);
        pointer-events: none;
        z-index: 0;
    }
</style>
""", unsafe_allow_html=True)

# ── Tabs ────────────────────────────────────────────────────────────────────
from tabs import overview, churn
from config import AUTHOR_NAME, GITHUB_URL, LINKEDIN_URL

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Overview",
    "Churn",
    "Credit Risk",
    "Fraud",
    "Verdict",
])

with tab1:
    overview.render()
with tab2:
    churn.render()
with tab3:
    st.markdown('<div style="padding: 4rem 0; text-align: center; color: rgba(226,232,240,0.4); font-size: 0.9rem;">Coming soon</div>', unsafe_allow_html=True)
with tab4:
    st.markdown('<div style="padding: 4rem 0; text-align: center; color: rgba(226,232,240,0.4); font-size: 0.9rem;">Coming soon</div>', unsafe_allow_html=True)
with tab5:
    st.markdown('<div style="padding: 4rem 0; text-align: center; color: rgba(226,232,240,0.4); font-size: 0.9rem;">Available after all datasets are benchmarked</div>', unsafe_allow_html=True)

# ── Footer ──────────────────────────────────────────────────────────────────
links = []
if GITHUB_URL:
    links.append(f'<a href="{GITHUB_URL}" target="_blank">GitHub</a>')
if LINKEDIN_URL:
    links.append(f'<a href="{LINKEDIN_URL}" target="_blank">LinkedIn</a>')
sep = " · ".join(links)

st.markdown(f"""
<div class="site-footer">
    Built by {AUTHOR_NAME}{f' · {sep}' if sep else ''}
</div>
""", unsafe_allow_html=True)

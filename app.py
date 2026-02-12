import streamlit as st

st.set_page_config(
    page_title="Tabular Arena",
    page_icon="⚔️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown('<style>'
    '@import url("https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap");'
    '@import url("https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&display=swap");'
    'html,body,[class*="css"]{font-family:"Inter",system-ui,-apple-system,sans-serif;}'
    '.stApp{background:#0F1117;color:#E8EAED;}'
    '.block-container{padding:2rem 3rem 3rem 3rem;max-width:1200px;}'
    '#MainMenu,footer,header{visibility:hidden;}'
    '.stDeployButton{display:none;}'
    # Tabs
    '.stTabs [data-baseweb="tab-list"]{gap:0;background:rgba(255,255,255,0.05);border:1px solid rgba(255,255,255,0.08);border-radius:12px;padding:4px;backdrop-filter:blur(12px);-webkit-backdrop-filter:blur(12px);}'
    '.stTabs [data-baseweb="tab"]{padding:9px 22px;font-weight:500;font-size:0.84rem;color:#9AA0A6;border-radius:8px;border-bottom:none;}'
    '.stTabs [data-baseweb="tab"]:hover{color:#E8EAED;background:rgba(255,255,255,0.04);}'
    '.stTabs [aria-selected="true"]{background:rgba(187,134,252,0.15) !important;color:#BB86FC !important;font-weight:600;border-bottom:none !important;}'
    '.stTabs [data-baseweb="tab-highlight"],.stTabs [data-baseweb="tab-border"]{display:none;}'
    # Cards
    '.card{background:rgba(255,255,255,0.05);border:1px solid rgba(255,255,255,0.08);border-radius:16px;padding:24px;backdrop-filter:blur(12px);-webkit-backdrop-filter:blur(12px);}'
    # Metric cards
    '.metric-card{background:rgba(255,255,255,0.05);border:1px solid rgba(255,255,255,0.08);border-radius:16px;padding:22px 24px;backdrop-filter:blur(12px);-webkit-backdrop-filter:blur(12px);border-left:3px solid #BB86FC;}'
    '.metric-label{font-size:0.72rem;font-weight:600;color:#9AA0A6;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:6px;}'
    '.metric-value{font-size:1.8rem;font-weight:700;color:#E8EAED;font-family:"JetBrains Mono",monospace;line-height:1.2;}'
    '.metric-detail{font-size:0.78rem;color:#9AA0A6;margin-top:4px;font-weight:500;}'
    # Sections
    '.section-header{font-size:1.05rem;font-weight:700;color:#E8EAED;margin:2.5rem 0 0.3rem 0;letter-spacing:-0.02em;}'
    '.section-caption{font-size:0.84rem;color:#9AA0A6;margin-bottom:1.2rem;}'
    '.soft-divider{border:none;border-top:1px solid rgba(255,255,255,0.06);margin:2.5rem 0;}'
    # Hero
    '.hero-title{font-size:2.6rem;font-weight:800;letter-spacing:-0.04em;line-height:1.1;color:#E8EAED;margin-bottom:10px;}'
    '.hero-sub{font-size:1.1rem;color:#9AA0A6;line-height:1.6;max-width:580px;}'
    # Pills
    '.pill{display:inline-block;padding:3px 12px;border-radius:100px;font-size:0.72rem;font-weight:600;letter-spacing:0.02em;}'
    '.pill-green{background:rgba(3,218,198,0.12);color:#03DAC6;border:1px solid rgba(3,218,198,0.2);}'
    '.pill-yellow{background:rgba(255,183,77,0.12);color:#FFB74D;border:1px solid rgba(255,183,77,0.2);}'
    '.pill-red{background:rgba(207,102,121,0.12);color:#CF6679;border:1px solid rgba(207,102,121,0.2);}'
    '.pill-blue{background:rgba(187,134,252,0.12);color:#BB86FC;border:1px solid rgba(187,134,252,0.2);}'
    '.pill-gray{background:rgba(154,160,166,0.1);color:#9AA0A6;border:1px solid rgba(154,160,166,0.15);}'
    # Leaderboard
    '.lb{width:100%;border-collapse:separate;border-spacing:0;}'
    '.lb th{font-size:0.68rem;font-weight:600;color:#9AA0A6;text-transform:uppercase;letter-spacing:0.07em;padding:12px 14px;text-align:left;border-bottom:1px solid rgba(255,255,255,0.08);background:rgba(255,255,255,0.03);}'
    '.lb th:first-child{border-radius:14px 0 0 0;}'
    '.lb th:last-child{border-radius:0 14px 0 0;}'
    '.lb th.r{text-align:right;}'
    '.lb td{padding:14px;font-size:0.85rem;color:#BDC1C6;border-bottom:1px solid rgba(255,255,255,0.04);}'
    '.lb td.r{text-align:right;font-family:"JetBrains Mono",monospace;font-size:0.82rem;}'
    '.lb tr:last-child td{border-bottom:none;}'
    '.lb tr:nth-child(even) td{background:rgba(255,255,255,0.02);}'
    '.lb .best{color:#03DAC6;font-weight:600;}'
    '.lb .worst{color:#5F6368;}'
    '.lb .model-name{font-weight:600;color:#E8EAED;}'
    # Rank badges
    '.rank-badge{width:28px;height:28px;border-radius:8px;display:inline-flex;align-items:center;justify-content:center;font-weight:700;font-size:0.75rem;font-family:"JetBrains Mono",monospace;}'
    '.rk1{background:rgba(3,218,198,0.15);color:#03DAC6;border:1px solid rgba(3,218,198,0.3);}'
    '.rk2{background:rgba(154,160,166,0.1);color:#9AA0A6;border:1px solid rgba(154,160,166,0.15);}'
    '.rk3{background:rgba(187,134,252,0.1);color:#BB86FC;border:1px solid rgba(187,134,252,0.2);}'
    '.rkn{background:rgba(154,160,166,0.06);color:#5F6368;border:1px solid rgba(154,160,166,0.08);}'
    # Takeaway
    '.takeaway{background:rgba(255,255,255,0.05);border:1px solid rgba(255,255,255,0.08);border-left:3px solid;border-radius:0 12px 12px 0;padding:14px 18px;margin-bottom:8px;font-size:0.88rem;line-height:1.6;color:#BDC1C6;}'
    '.takeaway strong{color:#E8EAED;}'
    # Cat tag
    '.cat-tag{display:inline-block;padding:2px 10px;border-radius:100px;font-size:0.7rem;font-weight:500;}'
    # Dataset stat bar
    '.ds-stats{display:flex;gap:24px;margin:12px 0 20px 0;flex-wrap:wrap;}'
    '.ds-stat{display:flex;flex-direction:column;}'
    '.ds-stat-val{font-size:1.1rem;font-weight:700;color:#E8EAED;font-family:"JetBrains Mono",monospace;}'
    '.ds-stat-lbl{font-size:0.68rem;font-weight:600;color:#9AA0A6;text-transform:uppercase;letter-spacing:0.06em;}'
    # Bento
    '.bento{display:grid;gap:10px;}'
    '.bento-2{grid-template-columns:1fr 1fr;}'
    '.bento-4{grid-template-columns:1fr 1fr 1fr 1fr;}'
    '.bento-3{grid-template-columns:1fr 1fr 1fr;}'
    # Footer
    '.site-footer{text-align:center;padding:2.5rem 0 1.5rem 0;margin-top:3rem;border-top:1px solid rgba(255,255,255,0.06);font-size:0.82rem;color:#9AA0A6;}'
    '.site-footer a{color:#BDC1C6;text-decoration:none;}'
    '.site-footer a:hover{color:#BB86FC;}'
    # Streamlit overrides
    '[data-testid="stMetric"]{display:none;}'
    '.stAlert{border-radius:12px;background:rgba(255,255,255,0.05);border:1px solid rgba(255,255,255,0.08);color:#E8EAED;}'
    'div[data-testid="stExpander"]{border:1px solid rgba(255,255,255,0.08);border-radius:12px;background:rgba(255,255,255,0.03);}'
    '.stDownloadButton button{background:rgba(255,255,255,0.06);border:1px solid rgba(255,255,255,0.12);border-radius:10px;color:#E8EAED;font-weight:500;}'
    '.stDownloadButton button:hover{background:rgba(187,134,252,0.12);border-color:rgba(187,134,252,0.25);color:#BB86FC;}'
    'h1 a,h2 a,h3 a{display:none !important;}'
    'h1,h2,h3,h4,h5,h6{color:#E8EAED !important;}'
    '.stMarkdown,.stText,p,span,li{color:#E8EAED;}'
    'label,.stSelectbox label,.stMultiSelect label{color:#BDC1C6 !important;}'
    '.stToggle > div{background:transparent !important;}'
    '::-webkit-scrollbar{width:6px;}'
    '::-webkit-scrollbar-track{background:rgba(255,255,255,0.02);}'
    '::-webkit-scrollbar-thumb{background:rgba(255,255,255,0.1);border-radius:3px;}'
'</style>', unsafe_allow_html=True)

from tabs import overview, churn
from config import AUTHOR_NAME, GITHUB_URL, LINKEDIN_URL

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview", "Churn", "Credit Risk", "Fraud", "Verdict"])

with tab1:
    overview.render()
with tab2:
    churn.render()
with tab3:
    st.markdown('<div style="padding:4rem 0;text-align:center;color:#5F6368;font-size:0.9rem;">Coming soon</div>', unsafe_allow_html=True)
with tab4:
    st.markdown('<div style="padding:4rem 0;text-align:center;color:#5F6368;font-size:0.9rem;">Coming soon</div>', unsafe_allow_html=True)
with tab5:
    st.markdown('<div style="padding:4rem 0;text-align:center;color:#5F6368;font-size:0.9rem;">Available after all datasets are benchmarked</div>', unsafe_allow_html=True)

links = []
if GITHUB_URL:
    links.append(f'<a href="{GITHUB_URL}" target="_blank">GitHub</a>')
if LINKEDIN_URL:
    links.append(f'<a href="{LINKEDIN_URL}" target="_blank">LinkedIn</a>')
sep = " · ".join(links)
st.markdown(f'<div class="site-footer">Built by {AUTHOR_NAME}{f" · {sep}" if sep else ""}</div>', unsafe_allow_html=True)

# ============================================================
# CLUSTERFORGE PRO — Full ML Pipeline Platform
# Beginner → Pro Learning Journey
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler,
    LabelEncoder, OneHotEncoder
)
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.cluster import (
    KMeans, DBSCAN, AgglomerativeClustering,
    SpectralClustering, Birch, MeanShift
)
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score, calinski_harabasz_score
)
from sklearn.feature_selection import VarianceThreshold
from scipy.stats import zscore
import scipy.cluster.hierarchy as sch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="ClusterForge Pro",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# THEME & CSS
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=IBM+Plex+Mono:wght@400;500;600&family=Outfit:wght@300;400;500&display=swap');

:root {
    --bg: #07080f;
    --bg2: #0d0e1a;
    --card: #111225;
    --card2: #181930;
    --border: #1e2035;
    --border2: #2a2c4a;
    --cyan: #22d3ee;
    --violet: #a78bfa;
    --emerald: #34d399;
    --amber: #fbbf24;
    --rose: #fb7185;
    --text: #e2e4f0;
    --muted: #6b7090;
    --dim: #2e3050;
    --font-head: 'Syne', sans-serif;
    --font-mono: 'IBM Plex Mono', monospace;
    --font-body: 'Outfit', sans-serif;
}

html, body, [class*="css"] {
    font-family: var(--font-body) !important;
    background: var(--bg) !important;
    color: var(--text) !important;
}
.main .block-container { padding: 1.5rem 2rem 3rem; max-width: 1500px; }

/* ── PIPELINE STEPPER ── */
.pipeline-nav {
    display: flex;
    align-items: center;
    gap: 0;
    background: var(--bg2);
    border: 1px solid var(--border);
    border-radius: 10px;
    overflow: hidden;
    margin-bottom: 2rem;
}
.step-btn {
    flex: 1;
    padding: 0.85rem 0.5rem;
    text-align: center;
    cursor: pointer;
    border-right: 1px solid var(--border);
    transition: all 0.2s;
    position: relative;
}
.step-btn:last-child { border-right: none; }
.step-btn:hover { background: var(--card2); }
.step-btn.active {
    background: linear-gradient(135deg, rgba(34,211,238,0.12), rgba(167,139,250,0.08));
    border-bottom: 2px solid var(--cyan);
}
.step-icon {
    font-size: 1.3rem;
    display: block;
    margin-bottom: 0.2rem;
}
.step-label {
    font-family: var(--font-mono);
    font-size: 0.58rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--muted);
}
.step-btn.active .step-label { color: var(--cyan); }
.step-num {
    position: absolute;
    top: 6px; left: 8px;
    font-family: var(--font-mono);
    font-size: 0.55rem;
    color: var(--dim);
}
.step-btn.done .step-icon::after {
    content: ' ✓';
    font-size: 0.7rem;
    color: var(--emerald);
}

/* ── CARDS ── */
.card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1.2rem;
    position: relative;
    overflow: hidden;
}
.card::after {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(34,211,238,0.4), transparent);
}
.card-title {
    font-family: var(--font-head);
    font-size: 1rem;
    font-weight: 700;
    color: var(--text);
    margin-bottom: 0.6rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.card-desc {
    font-size: 0.82rem;
    color: var(--muted);
    line-height: 1.6;
    margin-bottom: 0.8rem;
}

/* ── INSIGHT BOX ── */
.insight {
    background: linear-gradient(135deg, rgba(34,211,238,0.05), rgba(167,139,250,0.05));
    border: 1px solid rgba(34,211,238,0.2);
    border-left: 3px solid var(--cyan);
    border-radius: 6px;
    padding: 0.9rem 1.1rem;
    margin: 0.8rem 0;
    font-size: 0.82rem;
    color: var(--muted);
    line-height: 1.7;
}
.insight strong { color: var(--cyan); }
.warn-box {
    background: rgba(251,191,36,0.05);
    border: 1px solid rgba(251,191,36,0.2);
    border-left: 3px solid var(--amber);
    border-radius: 6px;
    padding: 0.8rem 1rem;
    margin: 0.6rem 0;
    font-size: 0.82rem;
    color: var(--muted);
}
.warn-box strong { color: var(--amber); }
.success-box {
    background: rgba(52,211,153,0.05);
    border: 1px solid rgba(52,211,153,0.2);
    border-left: 3px solid var(--emerald);
    border-radius: 6px;
    padding: 0.8rem 1rem;
    margin: 0.6rem 0;
    font-size: 0.82rem;
    color: var(--muted);
}
.success-box strong { color: var(--emerald); }

/* ── LEARN CALLOUT ── */
.learn-box {
    background: rgba(167,139,250,0.06);
    border: 1px solid rgba(167,139,250,0.2);
    border-radius: 8px;
    padding: 1rem 1.2rem;
    margin: 1rem 0;
}
.learn-title {
    font-family: var(--font-mono);
    font-size: 0.65rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--violet);
    margin-bottom: 0.5rem;
}
.learn-body {
    font-size: 0.82rem;
    color: var(--muted);
    line-height: 1.7;
}
.learn-body strong { color: var(--text); }

/* ── METRIC TILES ── */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
    gap: 0.8rem;
    margin: 1rem 0;
}
.metric-tile {
    background: var(--card2);
    border: 1px solid var(--border2);
    border-radius: 8px;
    padding: 1rem;
    text-align: center;
}
.metric-tile .val {
    font-family: var(--font-mono);
    font-size: 1.6rem;
    font-weight: 600;
    display: block;
}
.metric-tile .lbl {
    font-family: var(--font-mono);
    font-size: 0.58rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--muted);
    margin-top: 0.3rem;
    display: block;
}

/* ── SECTION DIVIDER ── */
.sec {
    font-family: var(--font-mono);
    font-size: 0.62rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--muted);
    border-bottom: 1px solid var(--border);
    padding-bottom: 0.4rem;
    margin: 1.8rem 0 1.2rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.sec::before {
    content: '';
    width: 3px; height: 3px;
    border-radius: 50%;
    background: var(--cyan);
    flex-shrink: 0;
}

/* ── HERO ── */
.hero {
    padding: 1.5rem 0 1rem;
    margin-bottom: 1.5rem;
}
.hero-title {
    font-family: var(--font-head);
    font-size: 2.4rem;
    font-weight: 800;
    background: linear-gradient(135deg, var(--cyan) 0%, var(--violet) 60%, var(--rose) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0; line-height: 1.1;
}
.hero-sub {
    font-size: 0.88rem;
    color: var(--muted);
    margin-top: 0.5rem;
    font-family: var(--font-mono);
    letter-spacing: 0.05em;
}

/* ── PROGRESS BADGE ── */
.progress-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: var(--card2);
    border: 1px solid var(--border2);
    border-radius: 20px;
    padding: 0.3rem 0.8rem;
    font-family: var(--font-mono);
    font-size: 0.65rem;
    color: var(--muted);
    margin-left: 1rem;
}
.progress-pill span { color: var(--cyan); }

/* ── SIDEBAR ── */
.stSidebar { background: var(--bg2) !important; border-right: 1px solid var(--border) !important; }
.stSidebar label, .stSidebar .stRadio > label {
    font-family: var(--font-mono) !important;
    font-size: 0.72rem !important;
    color: var(--muted) !important;
    letter-spacing: 0.05em !important;
}

/* ── BUTTONS ── */
.stButton > button {
    font-family: var(--font-mono) !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    border-radius: 6px !important;
    padding: 0.55rem 1.5rem !important;
    transition: all 0.2s !important;
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, var(--cyan), #0ea5e9) !important;
    color: #07080f !important;
    border: none !important;
    font-weight: 600 !important;
}
.stButton > button[kind="primary"]:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 24px rgba(34,211,238,0.25) !important;
}
.stButton > button[kind="secondary"] {
    background: var(--card) !important;
    color: var(--text) !important;
    border: 1px solid var(--border2) !important;
}

/* ── TABS ── */
.stTabs [data-baseweb="tab-list"] {
    background: var(--bg2) !important;
    border-bottom: 1px solid var(--border) !important;
    gap: 0 !important;
}
.stTabs [data-baseweb="tab"] {
    font-family: var(--font-mono) !important;
    font-size: 0.68rem !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    color: var(--muted) !important;
    padding: 0.75rem 1.2rem !important;
}
.stTabs [aria-selected="true"] {
    color: var(--cyan) !important;
    border-bottom: 2px solid var(--cyan) !important;
}

/* ── DATAFRAME ── */
.stDataFrame {
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    overflow: hidden !important;
}

/* ── EXPANDER ── */
.stExpander {
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    background: var(--card) !important;
}

/* ── MISC ── */
div[data-testid="stMetricValue"] { font-family: var(--font-mono) !important; }
div[data-testid="stMetricLabel"] {
    font-family: var(--font-mono) !important;
    font-size: 0.6rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
}
[data-testid="stFileUploadDropzone"] {
    background: var(--card) !important;
    border: 1px dashed var(--border2) !important;
    border-radius: 10px !important;
}
[data-testid="stFileUploadDropzone"]:hover {
    border-color: var(--cyan) !important;
}
hr { border-color: var(--border) !important; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# PLOTLY DEFAULTS
# ============================================================
PT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="IBM Plex Mono, monospace", color="#6b7090", size=11),
    colorway=["#22d3ee","#a78bfa","#34d399","#fbbf24","#fb7185","#38bdf8","#c084fc","#6ee7b7"],
)
COLORS = ["#22d3ee","#a78bfa","#34d399","#fbbf24","#fb7185","#38bdf8","#c084fc","#6ee7b7"]

# ============================================================
# SESSION STATE
# ============================================================
DEFAULTS = {
    "step": 0,
    "df_raw": None,
    "df_clean": None,
    "df_engineered": None,
    "X_processed": None,
    "labels": None,
    "model": None,
    "model_name": "",
    "metrics": {},
    "preprocessing_done": False,
    "eda_done": False,
    "engineering_done": False,
    "clustering_done": False,
    "outlier_method": "none",
    "scaler": "StandardScaler",
    "imputer": "Mean",
    "removed_cols": [],
    "feature_notes": {},
    "automl_results": [],
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("""
    <div style="font-family:IBM Plex Mono;font-size:0.6rem;letter-spacing:0.18em;
    text-transform:uppercase;color:#2e3050;padding:0.5rem 0 0.8rem;">▸ ClusterForge Pro</div>
    """, unsafe_allow_html=True)

    st.markdown("**Upload Dataset**")
    uploaded = st.file_uploader("CSV file", type=["csv"], label_visibility="collapsed")

    if uploaded:
        st.markdown(f"""
        <div style="background:#111225;border:1px solid #1e2035;border-radius:6px;
        padding:0.6rem 0.8rem;margin:0.5rem 0;font-family:IBM Plex Mono;font-size:0.7rem;color:#22d3ee;">
        ✓ {uploaded.name}
        </div>""", unsafe_allow_html=True)

    st.divider()
    st.markdown("**Experience Level**")
    xp = st.radio("", ["🟢 Beginner", "🟡 Intermediate", "🔴 Advanced"], label_visibility="collapsed")
    st.session_state["xp"] = xp

    st.divider()
    st.markdown("**Quick Jump**")
    steps = ["📥 Load", "🔍 EDA", "🧹 Clean", "⚙️ Features", "🤖 Model", "📈 Results", "🎓 Learn"]
    for i, s in enumerate(steps):
        if st.button(s, key=f"nav_{i}", use_container_width=True):
            st.session_state.step = i

    st.divider()
    st.caption("ClusterForge Pro · ML Pipeline for Everyone")

# ============================================================
# HERO
# ============================================================
col_hero, col_prog = st.columns([3,1])
with col_hero:
    st.markdown("""
    <div class="hero">
      <div class="hero-title">ClusterForge Pro</div>
      <div class="hero-sub">End-to-End ML Clustering Pipeline · From Raw Data to Expert Insights</div>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# PIPELINE STEPPER
# ============================================================
PIPELINE_STEPS = [
    ("📥", "Load Data"),
    ("🔍", "EDA"),
    ("🧹", "Clean"),
    ("⚙️", "Features"),
    ("🤖", "Cluster"),
    ("📈", "Results"),
    ("🎓", "Learn"),
]

step_html = '<div class="pipeline-nav">'
for i, (icon, label) in enumerate(PIPELINE_STEPS):
    active = "active" if st.session_state.step == i else ""
    done = "done" if st.session_state.step > i else ""
    step_html += f"""
    <div class="step-btn {active} {done}">
      <span class="step-num">{i+1}</span>
      <span class="step-icon">{icon}</span>
      <span class="step-label">{label}</span>
    </div>"""
step_html += "</div>"
st.markdown(step_html, unsafe_allow_html=True)

nav_cols = st.columns([1,1,6,1,1])
with nav_cols[0]:
    if st.button("◀ Back") and st.session_state.step > 0:
        st.session_state.step -= 1
        st.rerun()
with nav_cols[3]:
    if st.button("Next ▶") and st.session_state.step < 6:
        st.session_state.step += 1
        st.rerun()

# ============================================================
# HELPER UTILITIES
# ============================================================
def explain(title, body, kind="learn"):
    xp = st.session_state.get("xp", "🟢 Beginner")
    if xp == "🔴 Advanced" and kind == "learn":
        return
    css = "learn-box" if kind == "learn" else ("warn-box" if kind == "warn" else "insight")
    title_css = "learn-title" if kind == "learn" else ""
    body_css = "learn-body" if kind == "learn" else ""
    st.markdown(f"""
    <div class="{css}">
      <div class="{title_css}">{title}</div>
      <div class="{body_css}">{body}</div>
    </div>""", unsafe_allow_html=True)

def section(label):
    st.markdown(f'<div class="sec">{label}</div>', unsafe_allow_html=True)

def card_open(title, desc="", icon=""):
    st.markdown(f"""
    <div class="card">
      <div class="card-title">{icon} {title}</div>
      {"<div class='card-desc'>"+desc+"</div>" if desc else ""}
    """, unsafe_allow_html=True)

@st.cache_data
def load_csv(f):
    return pd.read_csv(f)

def get_numeric_cols(df):
    return df.select_dtypes(include=np.number).columns.tolist()

def get_cat_cols(df):
    return df.select_dtypes(exclude=np.number).columns.tolist()

def safe_silhouette(X, labels):
    try:
        valid = np.array(labels) != -1
        if valid.sum() < 2 or len(set(np.array(labels)[valid])) < 2:
            return None
        return round(silhouette_score(X[valid], np.array(labels)[valid]), 4)
    except:
        return None

def compute_all_metrics(X, labels):
    m = {}
    arr = np.array(labels)
    valid = arr != -1
    try:
        if valid.sum() >= 2 and len(set(arr[valid])) >= 2:
            m["Silhouette ↑"] = round(silhouette_score(X[valid], arr[valid]), 4)
            m["Davies-Bouldin ↓"] = round(davies_bouldin_score(X[valid], arr[valid]), 4)
            m["Calinski-Harabasz ↑"] = round(calinski_harabasz_score(X[valid], arr[valid]), 1)
    except:
        pass
    m["Clusters"] = len(set(arr)) - (1 if -1 in arr else 0)
    m["Noise pts"] = int((arr == -1).sum())
    return m

def preprocess_X(df, scaler_name, imputer_name):
    num = get_numeric_cols(df)
    cat = get_cat_cols(df)
    steps_num = []
    if imputer_name == "Mean":
        steps_num.append(("imp", SimpleImputer(strategy="mean")))
    elif imputer_name == "Median":
        steps_num.append(("imp", SimpleImputer(strategy="median")))
    elif imputer_name == "KNN":
        steps_num.append(("imp", KNNImputer(n_neighbors=5)))
    scaler_map = {"StandardScaler": StandardScaler(), "MinMaxScaler": MinMaxScaler(), "RobustScaler": RobustScaler()}
    steps_num.append(("scaler", scaler_map[scaler_name]))

    transformers = []
    if num:
        transformers.append(("num", Pipeline(steps_num), num))
    if cat:
        transformers.append(("cat", Pipeline([
            ("imp", SimpleImputer(strategy="constant", fill_value="Missing")),
            ("enc", OneHotEncoder(sparse_output=False, handle_unknown="ignore"))
        ]), cat))
    if not transformers:
        return np.zeros((len(df), 1))
    ct = ColumnTransformer(transformers)
    return ct.fit_transform(df)

def reduce_2d(X, method="PCA"):
    if X.shape[1] == 1:
        X = np.hstack([X, np.zeros((X.shape[0], 1))])
    if method == "t-SNE":
        perp = min(30, max(5, X.shape[0] // 3))
        return TSNE(n_components=2, perplexity=perp, random_state=42).fit_transform(X)
    pca = PCA(n_components=2, random_state=42)
    return pca.fit_transform(X)

# ============================================================
# ── STEP 0: LOAD DATA ──
# ============================================================
if st.session_state.step == 0:
    section("Step 1 · Load Your Dataset")

    explain("📥 What is this step?",
        "We start by <strong>loading your CSV file</strong> into the platform. "
        "Think of it as opening a spreadsheet — every row is a record (like a customer or product), "
        "and every column is a feature (like age, price, or category). "
        "The goal of clustering is to <strong>find hidden groups</strong> in that data automatically.",
        "learn")

    if not uploaded:
        st.markdown("""
        <div style="text-align:center;padding:4rem 2rem;background:#111225;
        border:2px dashed #1e2035;border-radius:14px;margin:1rem 0;">
          <div style="font-size:3rem;margin-bottom:0.8rem">🧬</div>
          <div style="font-family:IBM Plex Mono;font-size:0.9rem;color:#6b7090;margin-bottom:0.3rem;">
          Upload a CSV from the sidebar to begin
          </div>
          <div style="font-size:0.75rem;color:#2e3050;">
          Supports numeric + categorical columns · Auto-detects types
          </div>
        </div>""", unsafe_allow_html=True)

        section("Or load a sample dataset")
        sample_opts = {
            "🛒 Mall Customers": "https://raw.githubusercontent.com/dsrscientist/dataset1/master/Mall_Customers.csv",
            "🌸 Iris Flowers": "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv",
        }
        s_col1, s_col2 = st.columns(2)
        for idx, (name, url) in enumerate(sample_opts.items()):
            col = s_col1 if idx == 0 else s_col2
            with col:
                if st.button(f"Load {name}", use_container_width=True):
                    try:
                        import urllib.request
                        import io as _io
                        with urllib.request.urlopen(url) as r:
                            data = r.read().decode("utf-8")
                        df = pd.read_csv(_io.StringIO(data))
                        st.session_state.df_raw = df
                        st.session_state.step = 1
                        st.rerun()
                    except Exception as e:
                        st.error(f"Could not load sample: {e}")
        st.stop()

    df = load_csv(uploaded)
    if df.empty:
        st.error("Empty dataset."); st.stop()

    st.session_state.df_raw = df

    section("Dataset Preview")
    st.dataframe(df.head(20), use_container_width=True, height=300)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", df.shape[0])
    c2.metric("Columns", df.shape[1])
    c3.metric("Numeric", len(get_numeric_cols(df)))
    c4.metric("Missing", int(df.isnull().sum().sum()))

    section("Column Overview")
    col_info = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        n_null = int(df[col].isnull().sum())
        n_unique = int(df[col].nunique())
        col_info.append({"Column": col, "Type": dtype, "Missing": n_null,
                         "Unique Values": n_unique, "Sample": str(df[col].dropna().iloc[0]) if not df[col].dropna().empty else "—"})
    st.dataframe(pd.DataFrame(col_info), use_container_width=True, hide_index=True)

    explain("💡 Column Types Matter",
        "<strong>Numeric columns</strong> (int, float) are used directly for clustering. "
        "<strong>Object/string columns</strong> are categorical and will be encoded automatically. "
        "Columns like ID or Name that are unique per row should be removed in the Clean step.",
        "learn")

    if st.button("Proceed to EDA →", type="primary"):
        st.session_state.step = 1
        st.rerun()

# ============================================================
# ── STEP 1: EDA ──
# ============================================================
elif st.session_state.step == 1:
    section("Step 2 · Exploratory Data Analysis")
    df = st.session_state.df_raw
    if df is None:
        st.warning("Please load a dataset first."); st.stop()

    explain("🔍 What is EDA?",
        "<strong>Exploratory Data Analysis (EDA)</strong> is about <em>understanding</em> your data before doing anything else. "
        "We look at distributions, spot outliers, find correlations, and understand what the data is really saying. "
        "Professionals spend 60-80% of their time on EDA — it's the most important step!",
        "learn")

    num_cols = get_numeric_cols(df)
    cat_cols = get_cat_cols(df)

    tab_dist, tab_corr, tab_outlier, tab_cat, tab_stats = st.tabs([
        "📊 Distributions", "🔗 Correlations", "🎯 Outliers", "🏷️ Categorical", "📋 Statistics"
    ])

    with tab_dist:
        section("Feature Distributions")
        if num_cols:
            sel_feat = st.selectbox("Select feature", num_cols, key="eda_feat")
            c1, c2 = st.columns(2)
            with c1:
                fig = px.histogram(df, x=sel_feat, nbins=40,
                    color_discrete_sequence=["#22d3ee"], title=f"Histogram · {sel_feat}")
                fig.update_layout(**PT, height=300, xaxis=dict(showgrid=False),
                    yaxis=dict(gridcolor="#1e2035"))
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                fig2 = px.box(df, y=sel_feat, color_discrete_sequence=["#a78bfa"],
                    title=f"Box Plot · {sel_feat}")
                fig2.update_layout(**PT, height=300)
                st.plotly_chart(fig2, use_container_width=True)

            if len(num_cols) >= 2:
                section("Feature vs Feature Scatter")
                c1, c2, c3 = st.columns(3)
                fx = c1.selectbox("X axis", num_cols, key="sc_x")
                fy = c2.selectbox("Y axis", num_cols, index=min(1, len(num_cols)-1), key="sc_y")
                color_by = c3.selectbox("Color by (optional)", ["None"] + cat_cols, key="sc_c")
                color_col = None if color_by == "None" else color_by
                fig3 = px.scatter(df, x=fx, y=fy, color=color_col,
                    color_discrete_sequence=COLORS, opacity=0.6,
                    title=f"{fx} vs {fy}")
                fig3.update_layout(**PT, height=380)
                st.plotly_chart(fig3, use_container_width=True)
        else:
            st.info("No numeric features found.")

    with tab_corr:
        section("Correlation Matrix")
        if len(num_cols) >= 2:
            corr = df[num_cols].corr()
            fig = px.imshow(corr,
                color_continuous_scale=[[0,"#fb7185"],[0.5,"#111225"],[1,"#22d3ee"]],
                aspect="auto", title="Pearson Correlation")
            fig.update_layout(**PT, height=max(350, 22*len(num_cols)),
                title=dict(font=dict(family="IBM Plex Mono", size=12)))
            st.plotly_chart(fig, use_container_width=True)

            high_corr = []
            for i in range(len(corr.columns)):
                for j in range(i+1, len(corr.columns)):
                    v = corr.iloc[i,j]
                    if abs(v) > 0.8:
                        high_corr.append((corr.columns[i], corr.columns[j], round(v,3)))
            if high_corr:
                st.markdown(f"""<div class="warn-box">
                <strong>⚠ High Correlation Detected:</strong> {len(high_corr)} feature pair(s) with |r| > 0.8.
                Highly correlated features can bias clustering. Consider removing one from each pair.
                </div>""", unsafe_allow_html=True)
                st.dataframe(pd.DataFrame(high_corr, columns=["Feature A","Feature B","Correlation"]),
                    hide_index=True, use_container_width=True)
            explain("📐 What does correlation mean?",
                "Correlation measures how two features move together. "
                "<strong>+1</strong> = perfectly in sync, <strong>-1</strong> = perfectly opposite, "
                "<strong>0</strong> = no relationship. Features with very high correlation (>0.8) "
                "carry redundant information for clustering.", "learn")
        else:
            st.info("Need at least 2 numeric features.")

    with tab_outlier:
        section("Outlier Detection")
        if num_cols:
            sel = st.selectbox("Feature", num_cols, key="out_feat")
            col_data = df[sel].dropna()
            z = np.abs(zscore(col_data))
            iqr_low = col_data.quantile(0.25) - 1.5*(col_data.quantile(0.75)-col_data.quantile(0.25))
            iqr_high = col_data.quantile(0.75) + 1.5*(col_data.quantile(0.75)-col_data.quantile(0.25))
            z_outliers = int((z > 3).sum())
            iqr_outliers = int(((col_data < iqr_low) | (col_data > iqr_high)).sum())

            c1, c2, c3 = st.columns(3)
            c1.metric("Z-Score Outliers (|z|>3)", z_outliers)
            c2.metric("IQR Outliers", iqr_outliers)
            c3.metric("Total Rows", len(col_data))

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=list(range(len(col_data))), y=col_data.values,
                mode="markers", marker=dict(
                    color=["#fb7185" if v > 3 else "#22d3ee" for v in z],
                    size=4, opacity=0.7),
                name="Values"))
            fig.update_layout(**PT, height=300,
                title=dict(text=f"Outlier View · {sel} (red = Z-score > 3)",
                    font=dict(family="IBM Plex Mono", size=12)),
                xaxis=dict(showgrid=False), yaxis=dict(gridcolor="#1e2035"))
            st.plotly_chart(fig, use_container_width=True)

            explain("🎯 What are outliers?",
                "Outliers are data points far from the rest. "
                "The <strong>Z-score method</strong> flags points more than 3 standard deviations away. "
                "The <strong>IQR method</strong> uses the interquartile range. "
                "Outliers can heavily distort KMeans — but DBSCAN is actually designed to handle them!",
                "learn")
        else:
            st.info("No numeric features.")

    with tab_cat:
        section("Categorical Features")
        if cat_cols:
            sel_c = st.selectbox("Column", cat_cols, key="cat_sel")
            vc = df[sel_c].value_counts().head(20)
            fig = px.bar(x=vc.index, y=vc.values,
                color_discrete_sequence=["#a78bfa"],
                title=f"Value Counts · {sel_c}")
            fig.update_layout(**PT, height=320,
                xaxis=dict(showgrid=False, title=""),
                yaxis=dict(gridcolor="#1e2035", title="Count"))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No categorical features found.")

    with tab_stats:
        section("Descriptive Statistics")
        st.dataframe(df[num_cols].describe().round(3), use_container_width=True)
        explain("📋 What do these numbers mean?",
            "<strong>mean</strong> = average · <strong>std</strong> = spread of data · "
            "<strong>min/max</strong> = range · <strong>25%/50%/75%</strong> = quartiles. "
            "Look for features where std is very large relative to mean — "
            "these might need scaling in the Clean step.",
            "learn")

    st.session_state.eda_done = True
    if st.button("Proceed to Data Cleaning →", type="primary"):
        st.session_state.step = 2
        st.rerun()

# ============================================================
# ── STEP 2: CLEAN ──
# ============================================================
elif st.session_state.step == 2:
    section("Step 3 · Data Cleaning")
    df = st.session_state.df_raw
    if df is None:
        st.warning("Please load a dataset first."); st.stop()

    explain("🧹 Why clean data?",
        "Real-world data is messy. <strong>Missing values</strong> confuse algorithms. "
        "<strong>Useless columns</strong> (like IDs) add noise. "
        "<strong>Outliers</strong> can drag cluster centers far from where they should be. "
        "Cleaning is about making the data <em>ready</em> for a machine to learn from.",
        "learn")

    section("Missing Value Strategy")
    miss_pct = (df.isnull().sum() / len(df) * 100).round(1)
    miss_df = miss_pct[miss_pct > 0].reset_index()
    if miss_df.empty:
        st.markdown('<div class="success-box"><strong>✓ No missing values detected!</strong></div>', unsafe_allow_html=True)
    else:
        miss_df.columns = ["Column", "Missing %"]
        st.dataframe(miss_df, hide_index=True, use_container_width=True)
        imputer_choice = st.selectbox("Imputation Strategy",
            ["Mean", "Median", "KNN", "Drop Rows"],
            help="Mean/Median fill with average. KNN uses similar rows. Drop Rows removes them entirely.")
        st.session_state["imputer"] = imputer_choice
        explain("🔢 Which imputer to choose?",
            "<strong>Mean</strong>: fast, good for normally distributed data. "
            "<strong>Median</strong>: better when there are outliers. "
            "<strong>KNN</strong>: smartest — finds similar rows and borrows their values. "
            "<strong>Drop Rows</strong>: safest if missing data is random and you have many rows.",
            "learn")

    section("Column Management")
    all_cols = df.columns.tolist()
    num_cols = get_numeric_cols(df)
    cat_cols = get_cat_cols(df)

    auto_remove = [c for c in df.columns
                   if df[c].nunique() == len(df) or df[c].nunique() <= 1]
    if auto_remove:
        st.markdown(f"""<div class="warn-box"><strong>⚠ Suggested for removal:</strong>
        {', '.join(auto_remove)} — either unique per row (likely IDs) or constant value.</div>""",
        unsafe_allow_html=True)

    keep_cols = st.multiselect("Columns to KEEP for clustering",
        all_cols, default=[c for c in all_cols if c not in auto_remove])

    section("Outlier Handling")
    outlier_method = st.selectbox("Outlier Removal Method",
        ["None", "Z-Score (|z| > 3)", "IQR (1.5×IQR)", "Clip to 99th Percentile"],
        help="Remove or dampen extreme values before clustering.")
    st.session_state["outlier_method"] = outlier_method

    explain("✂️ When to remove outliers?",
        "Use <strong>Z-Score or IQR</strong> removal when outliers are clearly data errors. "
        "Use <strong>Clip</strong> when you want to keep all rows but reduce extreme influence. "
        "Use <strong>None</strong> if you plan to use DBSCAN (which handles outliers natively).",
        "learn")

    section("Scaling Method")
    scaler_choice = st.selectbox("Feature Scaler",
        ["StandardScaler", "MinMaxScaler", "RobustScaler"],
        help="Normalises feature magnitudes so no single feature dominates.")
    st.session_state["scaler"] = scaler_choice

    explain("📏 Why scale features?",
        "If one feature is 'Age (0–100)' and another is 'Salary (0–100,000)', "
        "salary will dominate distance calculations unfairly. Scaling puts all features on equal footing. "
        "<strong>StandardScaler</strong>: mean=0, std=1 (best for most cases). "
        "<strong>MinMaxScaler</strong>: range [0,1] (good for neural nets). "
        "<strong>RobustScaler</strong>: uses median/IQR — best when outliers exist.",
        "learn")

    if st.button("✅ Apply Cleaning & Continue →", type="primary"):
        df_c = df[keep_cols].copy()

        # Handle missing values
        imp = st.session_state["imputer"]
        if imp == "Drop Rows":
            df_c = df_c.dropna()
        else:
            for col in get_numeric_cols(df_c):
                if df_c[col].isnull().any():
                    if imp == "Mean":
                        df_c[col].fillna(df_c[col].mean(), inplace=True)
                    elif imp == "Median":
                        df_c[col].fillna(df_c[col].median(), inplace=True)
                    elif imp == "KNN":
                        vals = df_c[col].values.reshape(-1,1)
                        df_c[col] = KNNImputer(n_neighbors=5).fit_transform(vals)
            for col in get_cat_cols(df_c):
                df_c[col].fillna("Missing", inplace=True)

        # Outlier handling
        num_c = get_numeric_cols(df_c)
        if outlier_method == "Z-Score (|z| > 3)":
            if num_c:
                z = np.abs(zscore(df_c[num_c]))
                mask = (z < 3).all(axis=1)
                df_c = df_c[mask]
        elif outlier_method == "IQR (1.5×IQR)":
            for col in num_c:
                Q1, Q3 = df_c[col].quantile(0.25), df_c[col].quantile(0.75)
                IQR = Q3 - Q1
                df_c = df_c[(df_c[col] >= Q1-1.5*IQR) & (df_c[col] <= Q3+1.5*IQR)]
        elif outlier_method == "Clip to 99th Percentile":
            for col in num_c:
                lo, hi = df_c[col].quantile(0.01), df_c[col].quantile(0.99)
                df_c[col] = df_c[col].clip(lo, hi)

        st.session_state.df_clean = df_c
        st.session_state.preprocessing_done = True
        st.success(f"✓ Cleaned dataset: {df_c.shape[0]} rows × {df_c.shape[1]} columns")
        st.session_state.step = 3
        st.rerun()

# ============================================================
# ── STEP 3: FEATURE ENGINEERING ──
# ============================================================
elif st.session_state.step == 3:
    section("Step 4 · Feature Engineering & Selection")
    df = st.session_state.df_clean
    if df is None:
        st.warning("Complete the cleaning step first."); st.stop()

    explain("⚙️ What is Feature Engineering?",
        "Feature engineering is the art of <strong>creating or selecting the best inputs</strong> for your model. "
        "Sometimes combining two features (like Spend / Visits = Spend per Visit) reveals more signal than either alone. "
        "Removing noisy or redundant features often <em>improves</em> clustering quality.",
        "learn")

    num_cols = get_numeric_cols(df)
    cat_cols = get_cat_cols(df)
    df_eng = df.copy()

    section("Create New Features (Ratio / Interaction)")
    if len(num_cols) >= 2:
        with st.expander("➕ Add a ratio feature (A ÷ B)", expanded=False):
            c1, c2, c3 = st.columns(3)
            feat_a = c1.selectbox("Numerator", num_cols, key="ra")
            feat_b = c2.selectbox("Denominator", num_cols, index=1, key="rb")
            feat_name = c3.text_input("New Feature Name", value=f"{feat_a}_per_{feat_b}")
            if st.button("Create Ratio Feature"):
                df_eng[feat_name] = df_eng[feat_a] / (df_eng[feat_b].replace(0, np.nan) + 1e-9)
                st.session_state.df_clean = df_eng
                st.success(f"✓ Created feature: {feat_name}")
                st.rerun()

        with st.expander("✖️ Add an interaction feature (A × B)", expanded=False):
            c1, c2, c3 = st.columns(3)
            feat_a2 = c1.selectbox("Feature A", num_cols, key="ia")
            feat_b2 = c2.selectbox("Feature B", num_cols, index=min(1,len(num_cols)-1), key="ib")
            feat_name2 = c3.text_input("Name", value=f"{feat_a2}_x_{feat_b2}")
            if st.button("Create Interaction Feature"):
                df_eng[feat_name2] = df_eng[feat_a2] * df_eng[feat_b2]
                st.session_state.df_clean = df_eng
                st.success(f"✓ Created: {feat_name2}")
                st.rerun()
    else:
        st.info("Need at least 2 numeric columns to create engineered features.")

    section("Feature Selection")
    num_cols2 = get_numeric_cols(df_eng)
    if num_cols2:
        vt = VarianceThreshold()
        arr = df_eng[num_cols2].fillna(0).values
        try:
            vt.fit(arr)
            variances = dict(zip(num_cols2, vt.variances_))
            low_var = [c for c,v in variances.items() if v < 0.01]
        except:
            low_var = []

        if low_var:
            st.markdown(f"""<div class="warn-box"><strong>⚠ Low Variance Features:</strong>
            {', '.join(low_var)} — near-constant, carry little clustering signal.</div>""",
            unsafe_allow_html=True)

        selected_features = st.multiselect(
            "Features to include in clustering",
            df_eng.columns.tolist(),
            default=[c for c in df_eng.columns if c not in low_var]
        )
    else:
        selected_features = df_eng.columns.tolist()

    section("PCA Variance Explained")
    if len(num_cols2) >= 2:
        try:
            arr_scaled = StandardScaler().fit_transform(df_eng[num_cols2].fillna(0))
            pca_full = PCA().fit(arr_scaled)
            explained = np.cumsum(pca_full.explained_variance_ratio_) * 100
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=[f"PC{i+1}" for i in range(len(explained))],
                y=pca_full.explained_variance_ratio_*100,
                marker_color="#a78bfa", name="Individual"))
            fig.add_trace(go.Scatter(
                x=[f"PC{i+1}" for i in range(len(explained))],
                y=explained, mode="lines+markers",
                line=dict(color="#22d3ee", width=2),
                marker=dict(size=6), name="Cumulative"))
            fig.add_hline(y=80, line_dash="dash", line_color="#fbbf24",
                annotation_text="80% threshold")
            fig.update_layout(**PT, height=320,
                title=dict(text="PCA — Explained Variance", font=dict(family="IBM Plex Mono", size=12)),
                xaxis=dict(showgrid=False), yaxis=dict(gridcolor="#1e2035", title="%"))
            st.plotly_chart(fig, use_container_width=True)
            explain("🔵 What is PCA?",
                "<strong>Principal Component Analysis</strong> compresses your features into fewer dimensions "
                "while preserving as much information as possible. "
                "The chart shows how many components are needed to explain 80% of variance — "
                "a useful guide for understanding data complexity.",
                "learn")
        except:
            pass

    if st.button("✅ Lock Features & Continue →", type="primary"):
        df_final = df_eng[selected_features] if selected_features else df_eng
        st.session_state.df_engineered = df_final
        st.session_state.engineering_done = True
        st.session_state.step = 4
        st.rerun()

# ============================================================
# ── STEP 4: CLUSTERING ──
# ============================================================
elif st.session_state.step == 4:
    section("Step 5 · Clustering")
    df = st.session_state.df_engineered or st.session_state.df_clean
    if df is None:
        st.warning("Complete previous steps first."); st.stop()

    explain("🤖 What is Clustering?",
        "Clustering is <strong>unsupervised machine learning</strong> — we're not predicting a label, "
        "we're <em>discovering</em> natural groups. Imagine sorting people into personality types "
        "without knowing the types in advance. The algorithm finds them automatically based on similarity.",
        "learn")

    mode_tab, auto_tab = st.tabs(["🎓 Manual Mode", "⚡ AutoML Mode"])

    with mode_tab:
        section("Choose Your Algorithm")

        algo_info = {
            "KMeans": {
                "icon": "🎯", "level": "Beginner",
                "desc": "Partitions data into K spherical clusters by minimising within-cluster variance. Fast and scalable.",
                "best": "Clean data, roughly equal cluster sizes, you know roughly how many clusters to expect.",
                "worst": "Outliers, non-spherical clusters, varying density."
            },
            "DBSCAN": {
                "icon": "🌌", "level": "Intermediate",
                "desc": "Density-based — clusters are regions of high density separated by low density. Finds outliers naturally.",
                "best": "Arbitrary-shaped clusters, data with noise/outliers, unknown number of clusters.",
                "worst": "Varying density clusters, high-dimensional data."
            },
            "Agglomerative": {
                "icon": "🌳", "level": "Intermediate",
                "desc": "Hierarchical bottom-up: starts with each point as its own cluster, merges closest pairs.",
                "best": "When you want a hierarchy/dendrogram, small-to-medium datasets.",
                "worst": "Large datasets (slow), need to specify K."
            },
            "Spectral": {
                "icon": "🌊", "level": "Advanced",
                "desc": "Uses graph/eigenvalue decomposition to find clusters. Excellent for non-convex shapes.",
                "best": "Complex non-spherical clusters, image segmentation.",
                "worst": "Very large datasets (memory intensive)."
            },
            "Birch": {
                "icon": "🌿", "level": "Intermediate",
                "desc": "Builds a tree structure for fast incremental clustering. Memory efficient.",
                "best": "Very large datasets, streaming data.",
                "worst": "Non-spherical clusters, outlier-heavy data."
            },
            "MeanShift": {
                "icon": "🎱", "level": "Intermediate",
                "desc": "Finds cluster centres by shifting towards high-density regions. Auto-finds K.",
                "best": "Unknown K, smooth density distributions.",
                "worst": "Large datasets (slow), choosing bandwidth is tricky."
            },
        }

        algo = st.selectbox("Algorithm", list(algo_info.keys()))
        info = algo_info[algo]

        c1, c2 = st.columns([1,2])
        with c1:
            st.markdown(f"""
            <div class="card" style="height:100%">
              <div style="font-size:2.5rem;margin-bottom:0.5rem">{info['icon']}</div>
              <div style="font-family:IBM Plex Mono;font-size:0.6rem;letter-spacing:0.12em;
              text-transform:uppercase;color:#fbbf24;margin-bottom:0.4rem">{info['level']}</div>
              <div style="font-size:0.82rem;color:#6b7090;line-height:1.6">{info['desc']}</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="card">
              <div style="margin-bottom:0.6rem">
                <div style="font-family:IBM Plex Mono;font-size:0.6rem;color:#34d399;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:0.3rem">✓ Best when</div>
                <div style="font-size:0.82rem;color:#6b7090">{info['best']}</div>
              </div>
              <div>
                <div style="font-family:IBM Plex Mono;font-size:0.6rem;color:#fb7185;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:0.3rem">✗ Avoid when</div>
                <div style="font-size:0.82rem;color:#6b7090">{info['worst']}</div>
              </div>
            </div>""", unsafe_allow_html=True)

        section("Hyperparameters")
        params = {}
        if algo == "KMeans":
            c1, c2, c3 = st.columns(3)
            params["n_clusters"] = c1.slider("K (clusters)", 2, 15, 3)
            params["init"] = c2.selectbox("Init method", ["k-means++", "random"])
            params["max_iter"] = c3.slider("Max iterations", 100, 1000, 300, step=50)
            explain("🔧 KMeans Parameters",
                "<strong>K</strong>: number of clusters — use the elbow curve in Results to tune. "
                "<strong>k-means++</strong>: smarter initialisation, almost always better than random. "
                "<strong>max_iter</strong>: how long to optimise — 300 is usually enough.",
                "learn")
        elif algo == "DBSCAN":
            c1, c2 = st.columns(2)
            params["eps"] = c1.slider("ε (epsilon)", 0.05, 3.0, 0.5, step=0.05)
            params["min_samples"] = c2.slider("Min Samples", 2, 30, 5)
            explain("🔧 DBSCAN Parameters",
                "<strong>ε (epsilon)</strong>: max distance to be considered a neighbour. Too small = many noise points. Too large = everything merges. "
                "<strong>min_samples</strong>: how many neighbours needed to form a core point. "
                "Rule of thumb: min_samples ≈ 2× dimensions.",
                "learn")
        elif algo == "Agglomerative":
            c1, c2 = st.columns(2)
            params["n_clusters"] = c1.slider("Clusters", 2, 15, 3)
            params["linkage"] = c2.selectbox("Linkage", ["ward","complete","average","single"])
            explain("🔧 Agglomerative Parameters",
                "<strong>ward</strong>: minimises within-cluster variance (usually best). "
                "<strong>complete</strong>: max distance between clusters. "
                "<strong>average</strong>: average distance. "
                "<strong>single</strong>: min distance — prone to chaining effect.",
                "learn")
        elif algo == "Spectral":
            c1, c2 = st.columns(2)
            params["n_clusters"] = c1.slider("Clusters", 2, 10, 3)
            params["affinity"] = c2.selectbox("Affinity", ["rbf","nearest_neighbors"])
        elif algo == "Birch":
            c1, c2, c3 = st.columns(3)
            params["n_clusters"] = c1.slider("Clusters", 2, 15, 3)
            params["threshold"] = c2.slider("Threshold", 0.1, 1.0, 0.5, step=0.05)
            params["branching_factor"] = c3.slider("Branch Factor", 10, 100, 50, step=10)
        elif algo == "MeanShift":
            bw = st.slider("Bandwidth (0 = auto-estimate)", 0.0, 5.0, 0.0, step=0.1)
            params["bandwidth"] = bw if bw > 0 else None

        scaler_c = st.selectbox("Scaler", ["StandardScaler","MinMaxScaler","RobustScaler"], key="cluster_scaler")
        reduction_c = st.selectbox("Visualisation Reduction", ["PCA","t-SNE"], key="cluster_red")
        st.session_state["reduction"] = reduction_c

        if st.button("▶ Train Model", type="primary", key="train_manual"):
            with st.spinner("Training…"):
                try:
                    X = preprocess_X(df, scaler_c, st.session_state.get("imputer","Mean"))
                    st.session_state.X_processed = X

                    model_map = {
                        "KMeans": KMeans(n_clusters=params.get("n_clusters",3),
                            init=params.get("init","k-means++"),
                            max_iter=params.get("max_iter",300),
                            random_state=42, n_init="auto"),
                        "DBSCAN": DBSCAN(eps=params.get("eps",0.5),
                            min_samples=params.get("min_samples",5)),
                        "Agglomerative": AgglomerativeClustering(
                            n_clusters=params.get("n_clusters",3),
                            linkage=params.get("linkage","ward")),
                        "Spectral": SpectralClustering(
                            n_clusters=params.get("n_clusters",3),
                            affinity=params.get("affinity","rbf"), random_state=42),
                        "Birch": Birch(n_clusters=params.get("n_clusters",3),
                            threshold=params.get("threshold",0.5),
                            branching_factor=params.get("branching_factor",50)),
                        "MeanShift": MeanShift(bandwidth=params.get("bandwidth")),
                    }
                    model = model_map[algo]
                    labels = model.fit_predict(X)
                    st.session_state.model = model
                    st.session_state.model_name = algo
                    st.session_state.labels = labels
                    st.session_state.metrics = compute_all_metrics(X, labels)
                    st.session_state.clustering_done = True
                    st.success(f"✓ Training complete! Found {st.session_state.metrics.get('Clusters','?')} clusters.")
                    st.session_state.step = 5
                    st.rerun()
                except Exception as e:
                    st.error(f"Training failed: {e}")

    with auto_tab:
        section("AutoML — Automated Model Selection")
        explain("⚡ What does AutoML do?",
            "AutoML tries <strong>many algorithms and configurations automatically</strong> "
            "and picks the one with the best Silhouette Score. "
            "Great for when you don't know which algorithm to choose — "
            "it's like having an expert try everything for you!",
            "learn")

        scaler_a = st.selectbox("Scaler", ["StandardScaler","MinMaxScaler","RobustScaler"], key="automl_scaler")
        reduction_a = st.selectbox("Visualisation", ["PCA","t-SNE"], key="automl_red")

        n_km = st.slider("KMeans: max K to try", 2, 12, 8)

        if st.button("⚡ Run AutoML", type="primary", key="run_automl"):
            X = preprocess_X(df, scaler_a, st.session_state.get("imputer","Mean"))
            st.session_state.X_processed = X
            st.session_state["reduction"] = reduction_a

            prog = st.progress(0)
            status = st.empty()
            results = []
            configs = []

            for k in range(2, n_km+1):
                configs.append(("KMeans", {"n_clusters":k, "random_state":42, "n_init":"auto"}))
            for k in range(2, 6):
                for link in ["ward","complete","average"]:
                    configs.append(("Agglomerative", {"n_clusters":k, "linkage":link}))
            for eps in [0.2, 0.4, 0.6, 0.8, 1.0]:
                configs.append(("DBSCAN", {"eps":eps, "min_samples":5}))
            for k in range(2, 5):
                configs.append(("Birch", {"n_clusters":k, "threshold":0.5}))

            for i, (name, cfg) in enumerate(configs):
                prog.progress((i+1)/len(configs))
                status.markdown(f"""<div style="font-family:IBM Plex Mono;font-size:0.72rem;
                color:#6b7090;">▸ Testing {name} {cfg}…</div>""", unsafe_allow_html=True)
                try:
                    m_map = {
                        "KMeans": KMeans(**cfg),
                        "Agglomerative": AgglomerativeClustering(**cfg),
                        "DBSCAN": DBSCAN(**cfg),
                        "Birch": Birch(**cfg),
                    }
                    m = m_map[name]
                    lbl = m.fit_predict(X)
                    arr = np.array(lbl)
                    valid = arr != -1
                    n_clust = len(set(arr[valid]))
                    if n_clust < 2:
                        continue
                    sil = silhouette_score(X[valid], arr[valid]) if valid.sum() >= 2 else -1
                    db = davies_bouldin_score(X[valid], arr[valid]) if valid.sum() >= 2 else 999
                    results.append({
                        "Algorithm": name,
                        "Config": str(cfg),
                        "Silhouette ↑": round(sil,4),
                        "Davies-Bouldin ↓": round(db,4),
                        "Clusters": n_clust,
                        "_model": m, "_labels": lbl,
                    })
                except:
                    continue

            prog.empty(); status.empty()

            if results:
                results.sort(key=lambda x: x["Silhouette ↑"], reverse=True)
                st.session_state.automl_results = results
                best = results[0]
                st.session_state.model = best["_model"]
                st.session_state.model_name = f"AutoML · {best['Algorithm']}"
                st.session_state.labels = best["_labels"]
                st.session_state.metrics = compute_all_metrics(X, best["_labels"])
                st.session_state.clustering_done = True

                st.markdown(f"""<div class="success-box">
                <strong>✓ AutoML Complete!</strong> Best: <strong>{best['Algorithm']}</strong>
                with Silhouette = <strong>{best['Silhouette ↑']}</strong>
                ({best['Clusters']} clusters)
                </div>""", unsafe_allow_html=True)

                display_df = pd.DataFrame([{k:v for k,v in r.items() if not k.startswith("_")} for r in results[:15]])
                st.dataframe(display_df, use_container_width=True, hide_index=True)

                st.session_state.step = 5
                st.rerun()
            else:
                st.error("No valid configurations found. Try different cleaning settings.")

# ============================================================
# ── STEP 5: RESULTS ──
# ============================================================
elif st.session_state.step == 5:
    section("Step 6 · Results & Visualisation")
    df = st.session_state.df_engineered or st.session_state.df_clean
    X = st.session_state.X_processed
    labels = st.session_state.labels
    metrics = st.session_state.metrics

    if labels is None or X is None:
        st.warning("Run clustering first."); st.stop()

    df_r = (st.session_state.df_raw if st.session_state.df_raw is not None else df).copy()
    if len(df_r) == len(labels):
        df_r["Cluster"] = labels
    else:
        df_r = df.copy()
        df_r["Cluster"] = labels

    model_name = st.session_state.model_name
    n_clusters = metrics.get("Clusters", "?")

    # Score strip
    cols = st.columns(5)
    m_display = [
        ("Model", model_name.split("·")[-1].strip(), "#22d3ee"),
        ("Clusters", n_clusters, "#a78bfa"),
        ("Silhouette ↑", metrics.get("Silhouette ↑","N/A"), "#34d399"),
        ("Davies-Bouldin ↓", metrics.get("Davies-Bouldin ↓","N/A"), "#fbbf24"),
        ("Calinski-H ↑", metrics.get("Calinski-Harabasz ↑","N/A"), "#fb7185"),
    ]
    for i, (lbl, val, clr) in enumerate(m_display):
        cols[i].markdown(f"""
        <div class="metric-tile">
          <span class="val" style="color:{clr}">{val}</span>
          <span class="lbl">{lbl}</span>
        </div>""", unsafe_allow_html=True)

    if metrics.get("Noise pts", 0) > 0:
        st.markdown(f"""<div class="warn-box">
        <strong>⚠ {metrics['Noise pts']} noise points</strong> (label -1) — DBSCAN outliers excluded from scoring.
        </div>""", unsafe_allow_html=True)

    explain("📊 How to read the scores?",
        "<strong>Silhouette Score</strong> (−1 to 1): higher is better. >0.5 = strong clusters. "
        "<strong>Davies-Bouldin</strong>: lower is better. Near 0 = tight, well-separated clusters. "
        "<strong>Calinski-Harabasz</strong>: higher is better. Measures cluster density vs separation.",
        "learn")

    reduction_method = st.session_state.get("reduction", "PCA")

    tab_scatter, tab_dist, tab_profile, tab_heatmap, tab_elbow, tab_dendro, tab_export = st.tabs([
        "🗺️ Scatter", "📊 Distribution", "🧬 Profiles", "🌡️ Heatmap", "📐 Elbow / Sweep", "🌳 Dendrogram", "💾 Export"
    ])

    with tab_scatter:
        section(f"{reduction_method} Cluster Visualisation")
        with st.spinner(f"Computing {reduction_method}…"):
            X2d = reduce_2d(X, reduction_method)
        df_plot = pd.DataFrame({
            "x": X2d[:,0], "y": X2d[:,1],
            "Cluster": [str(l) for l in labels],
        })
        num_cols_r = [c for c in df_r.select_dtypes(include=np.number).columns if c != "Cluster"]
        if num_cols_r:
            hover_col = st.selectbox("Hover info column", ["None"] + num_cols_r, key="hover")
            if hover_col != "None" and len(df_r) == len(df_plot):
                df_plot["hover"] = df_r[hover_col].values
                fig = px.scatter(df_plot, x="x", y="y", color="Cluster",
                    hover_data={"hover":True},
                    color_discrete_sequence=COLORS,
                    title=f"Cluster Map · {reduction_method}")
            else:
                fig = px.scatter(df_plot, x="x", y="y", color="Cluster",
                    color_discrete_sequence=COLORS,
                    title=f"Cluster Map · {reduction_method}")
        else:
            fig = px.scatter(df_plot, x="x", y="y", color="Cluster",
                color_discrete_sequence=COLORS, title=f"Cluster Map · {reduction_method}")

        fig.update_traces(marker=dict(size=5, opacity=0.75, line=dict(width=0)))
        fig.update_layout(**PT, height=480,
            title=dict(font=dict(family="IBM Plex Mono", size=12)),
            xaxis=dict(gridcolor="#1e2035", zeroline=False, title=""),
            yaxis=dict(gridcolor="#1e2035", zeroline=False, title=""),
            legend=dict(bgcolor="rgba(17,18,37,0.8)", bordercolor="#1e2035", borderwidth=1))
        st.plotly_chart(fig, use_container_width=True)

    with tab_dist:
        c1, c2 = st.columns(2)
        with c1:
            cnt = pd.Series(labels).value_counts().sort_index().reset_index()
            cnt.columns = ["Cluster","Count"]
            cnt["Cluster"] = cnt["Cluster"].astype(str)
            fig = px.bar(cnt, x="Cluster", y="Count", color="Cluster",
                color_discrete_sequence=COLORS, title="Cluster Sizes")
            fig.update_layout(**PT, height=320, showlegend=False,
                xaxis=dict(showgrid=False), yaxis=dict(gridcolor="#1e2035"), bargap=0.3)
            fig.update_traces(marker_line_width=0)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig2 = px.pie(cnt, values="Count", names="Cluster",
                color_discrete_sequence=COLORS, title="Cluster Share",
                hole=0.5)
            fig2.update_layout(**PT, height=320)
            st.plotly_chart(fig2, use_container_width=True)

    with tab_profile:
        section("Cluster Feature Profiles")
        num_cols_r = [c for c in df_r.select_dtypes(include=np.number).columns if c != "Cluster"]
        if num_cols_r:
            summary = df_r.groupby("Cluster")[num_cols_r].mean().round(3)
            fig = go.Figure()
            for i, row in summary.iterrows():
                fig.add_trace(go.Scatterpolar(
                    r=row.values,
                    theta=row.index.tolist(),
                    fill="toself",
                    name=f"Cluster {i}",
                    line=dict(color=COLORS[i % len(COLORS)]),
                    opacity=0.7
                ))
            fig.update_layout(**PT, height=420,
                polar=dict(
                    bgcolor="#0d0e1a",
                    radialaxis=dict(visible=True, gridcolor="#1e2035", showticklabels=False),
                    angularaxis=dict(gridcolor="#1e2035")
                ),
                title=dict(text="Radar: Cluster Profiles", font=dict(family="IBM Plex Mono", size=12)),
                legend=dict(bgcolor="rgba(17,18,37,0.8)", bordercolor="#1e2035", borderwidth=1))
            st.plotly_chart(fig, use_container_width=True)

            section("Mean Values by Cluster")
            st.dataframe(summary, use_container_width=True)

            diff = (summary.max() - summary.min()).sort_values(ascending=True).tail(10)
            fig2 = go.Figure(go.Bar(
                x=diff.values, y=diff.index, orientation="h",
                marker=dict(color=diff.values,
                    colorscale=[[0,"#1e2035"],[0.5,"#a78bfa"],[1,"#22d3ee"]]),
            ))
            fig2.update_layout(**PT, height=350,
                title=dict(text="Top Differentiating Features", font=dict(family="IBM Plex Mono", size=12)),
                xaxis=dict(gridcolor="#1e2035", title="Mean Range"),
                yaxis=dict(showgrid=False))
            st.plotly_chart(fig2, use_container_width=True)

    with tab_heatmap:
        num_cols_r = [c for c in df_r.select_dtypes(include=np.number).columns if c != "Cluster"]
        if len(num_cols_r) >= 2:
            summary = df_r.groupby("Cluster")[num_cols_r].mean()
            norm = (summary - summary.min()) / (summary.max() - summary.min() + 1e-9)
            fig = px.imshow(norm.T,
                color_continuous_scale=[[0,"#07080f"],[0.4,"#1e2035"],[0.7,"#a78bfa"],[1,"#22d3ee"]],
                aspect="auto", title="Feature Heatmap (Normalised per feature)")
            fig.update_layout(**PT, height=max(300, 28*len(num_cols_r)),
                title=dict(font=dict(family="IBM Plex Mono", size=12)))
            st.plotly_chart(fig, use_container_width=True)

            section("Pair Scatter Matrix (Top 4 Features)")
            diff = (summary.max() - summary.min()).sort_values(ascending=False)
            top4 = diff.head(4).index.tolist()
            if len(top4) >= 2:
                fig2 = px.scatter_matrix(df_r, dimensions=top4,
                    color=df_r["Cluster"].astype(str),
                    color_discrete_sequence=COLORS, title="Pair Scatter")
                fig2.update_traces(marker=dict(size=3, opacity=0.5))
                fig2.update_layout(**PT, height=520,
                    title=dict(font=dict(family="IBM Plex Mono", size=12)))
                st.plotly_chart(fig2, use_container_width=True)

    with tab_elbow:
        section("KMeans Optimisation Curves")
        c1, c2 = st.columns(2)
        max_k = st.slider("Max K to evaluate", 3, 15, 10)
        inertias, sils = [], []
        for k in range(2, max_k+1):
            try:
                m = KMeans(n_clusters=k, random_state=42, n_init="auto")
                lbl_k = m.fit_predict(X)
                inertias.append(m.inertia_)
                sils.append(silhouette_score(X, lbl_k) if len(set(lbl_k))>=2 else 0)
            except:
                inertias.append(0); sils.append(0)
        ks = list(range(2, max_k+1))
        with c1:
            fig = go.Figure(go.Scatter(x=ks, y=inertias, mode="lines+markers",
                line=dict(color="#22d3ee", width=2), marker=dict(size=7)))
            fig.update_layout(**PT, height=300,
                title=dict(text="Elbow Curve", font=dict(family="IBM Plex Mono", size=12)),
                xaxis=dict(gridcolor="#1e2035", dtick=1),
                yaxis=dict(gridcolor="#1e2035", title="Inertia"))
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig2 = go.Figure(go.Scatter(x=ks, y=sils, mode="lines+markers",
                line=dict(color="#a78bfa", width=2), marker=dict(size=7),
                fill="tozeroy", fillcolor="rgba(167,139,250,0.08)"))
            fig2.update_layout(**PT, height=300,
                title=dict(text="Silhouette by K", font=dict(family="IBM Plex Mono", size=12)),
                xaxis=dict(gridcolor="#1e2035", dtick=1),
                yaxis=dict(gridcolor="#1e2035", title="Score"))
            st.plotly_chart(fig2, use_container_width=True)

        explain("📐 How to use these charts?",
            "The <strong>Elbow Curve</strong> shows where adding more clusters gives diminishing returns — "
            "look for a 'kink' in the line. "
            "The <strong>Silhouette chart</strong> shows which K gives best-separated clusters — "
            "pick the peak.",
            "learn")

    with tab_dendro:
        section("Hierarchical Dendrogram")
        num_cols_d = [c for c in df_r.select_dtypes(include=np.number).columns if c != "Cluster"]
        if num_cols_d:
            max_rows = min(500, len(df_r))
            sample_df = df_r[num_cols_d].dropna().sample(min(max_rows, len(df_r)), random_state=42)
            Z = sch.linkage(StandardScaler().fit_transform(sample_df), method="ward")
            fig_d, ax = plt.subplots(figsize=(12, 4))
            fig_d.patch.set_facecolor("#0d0e1a")
            ax.set_facecolor("#0d0e1a")
            sch.dendrogram(Z, ax=ax, leaf_rotation=90,
                color_threshold=0.7*max(Z[:,2]),
                above_threshold_color="#22d3ee",
                link_color_func=lambda k: "#a78bfa")
            ax.tick_params(colors="#6b7090", labelsize=7)
            ax.spines[:].set_color("#1e2035")
            ax.set_title("Hierarchical Dendrogram (Ward Linkage)",
                color="#6b7090", fontsize=10, pad=8)
            plt.tight_layout()
            st.pyplot(fig_d)
            plt.close()
            explain("🌳 Reading the Dendrogram",
                "Each leaf is a data point. Points that merge low down are most similar. "
                "The <strong>height of a merge</strong> = how different the groups are. "
                "Draw a horizontal line to decide how many clusters to cut — count the branches below it.",
                "learn")

    with tab_export:
        section("Export Your Results")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.download_button("⬇ Clustered CSV", df_r.to_csv(index=False).encode(),
                "clustered_data.csv", "text/csv", use_container_width=True)
        with c2:
            if st.session_state.model and hasattr(st.session_state.model, "fit"):
                st.download_button("💾 Model (.pkl)", pickle.dumps(st.session_state.model),
                    "model.pkl", use_container_width=True)
        with c3:
            num_cols_s = [c for c in df_r.select_dtypes(include=np.number).columns if c != "Cluster"]
            if num_cols_s:
                summary_csv = df_r.groupby("Cluster")[num_cols_s].describe().to_csv()
                st.download_button("📄 Stats Summary", summary_csv.encode(),
                    "cluster_summary.csv", "text/csv", use_container_width=True)
        with c4:
            metrics_df = pd.DataFrame([metrics])
            st.download_button("📊 Metrics CSV", metrics_df.to_csv(index=False).encode(),
                "metrics.csv", "text/csv", use_container_width=True)

    if st.button("📈 View Learning Module →", type="primary"):
        st.session_state.step = 6
        st.rerun()

# ============================================================
# ── STEP 6: LEARNING MODULE ──
# ============================================================
elif st.session_state.step == 6:
    section("Step 7 · Learning Module — Beginner to Pro")

    st.markdown("""
    <div style="background:linear-gradient(135deg,rgba(34,211,238,0.06),rgba(167,139,250,0.06));
    border:1px solid rgba(34,211,238,0.15);border-radius:12px;padding:1.5rem 2rem;margin-bottom:1.5rem;">
      <div style="font-family:Syne;font-size:1.3rem;font-weight:800;color:#e2e4f0;margin-bottom:0.4rem;">
        🎓 The ML Clustering Roadmap
      </div>
      <div style="font-size:0.85rem;color:#6b7090;line-height:1.7;">
        This module teaches you everything you need to understand, apply, and explain clustering —
        from your very first model to production-grade deployments.
      </div>
    </div>
    """, unsafe_allow_html=True)

    level_tab1, level_tab2, level_tab3, level_tab4 = st.tabs([
        "🟢 Beginner", "🟡 Intermediate", "🔴 Advanced", "📚 Glossary"
    ])

    with level_tab1:
        topics = [
            ("What is Machine Learning?",
             "Machine learning is teaching computers to find patterns <em>without being explicitly programmed</em>. "
             "Instead of writing rules, you show the computer data and it figures out the patterns itself.",
             "Think of it like this: instead of telling a child 'a dog has 4 legs, fur, and barks', "
             "you show them 1000 pictures of dogs and they learn to recognise dogs on their own."),
            ("What is Clustering?",
             "Clustering is <strong>unsupervised learning</strong> — we have data but <em>no labels</em>. "
             "The algorithm finds natural groups (clusters) on its own. "
             "Nobody tells it what the groups are — it discovers them.",
             "Customer segmentation: a retailer uploads transaction data. "
             "Clustering discovers 'high-value shoppers', 'bargain hunters', and 'occasional buyers' — "
             "without anyone labelling customers first."),
            ("How Does KMeans Work?",
             "1. Pick K random points as cluster centres. "
             "2. Assign every data point to the nearest centre. "
             "3. Move each centre to the average of its assigned points. "
             "4. Repeat steps 2–3 until nothing changes.",
             "Imagine 3 magnets and iron filings. The magnets attract nearby filings, "
             "then the magnets move to the centre of their cluster — repeat until stable."),
            ("What is a Silhouette Score?",
             "The silhouette score measures how well-separated your clusters are. "
             "<strong>Close to +1</strong>: point is clearly in the right cluster. "
             "<strong>Close to 0</strong>: point is on the boundary between clusters. "
             "<strong>Close to -1</strong>: point might be in the wrong cluster.",
             "Score 0.7 = great separation. Score 0.3 = clusters are fuzzy/overlapping."),
        ]
        for title, body, example in topics:
            with st.expander(f"📖 {title}"):
                st.markdown(f'<div style="font-size:0.85rem;color:#6b7090;line-height:1.8;margin-bottom:1rem">{body}</div>', unsafe_allow_html=True)
                st.markdown(f"""
                <div style="background:rgba(34,211,238,0.05);border-left:3px solid #22d3ee;
                border-radius:4px;padding:0.8rem 1rem;font-size:0.82rem;color:#6b7090;line-height:1.7;">
                💡 <strong>Example:</strong> {example}
                </div>""", unsafe_allow_html=True)

    with level_tab2:
        topics2 = [
            ("The Full ML Pipeline",
             ["1️⃣ Load Data", "2️⃣ EDA — understand your data",
              "3️⃣ Clean — handle missing values & outliers",
              "4️⃣ Feature Engineering — create/select best inputs",
              "5️⃣ Preprocessing — scale/encode features",
              "6️⃣ Model Selection — choose & tune algorithm",
              "7️⃣ Evaluation — measure quality with multiple metrics",
              "8️⃣ Interpret — what do the clusters mean?",
              "9️⃣ Deploy — use the model in production"]),
            ("When to Use Each Algorithm",
             ["<strong>KMeans</strong>: default choice. Fast. Needs K. Assumes spherical clusters.",
              "<strong>DBSCAN</strong>: when you expect noise/outliers. Finds K automatically. Works with any shape.",
              "<strong>Agglomerative</strong>: when you want a hierarchy. Good for small-medium data.",
              "<strong>Spectral</strong>: for complex non-convex shapes. Computationally expensive.",
              "<strong>Birch</strong>: very large datasets. Memory efficient.",
              "<strong>MeanShift</strong>: no K needed. Slow on large data."]),
            ("The Curse of Dimensionality",
             ["As features increase, 'distance' becomes meaningless — everything seems equally far apart.",
              "Fix 1: PCA — compress to fewer dimensions while keeping variance.",
              "Fix 2: Feature selection — drop redundant/noisy features.",
              "Fix 3: Use cosine similarity instead of Euclidean distance.",
              "Rule of thumb: for N features, aim for at most √N clusters."]),
            ("How to Choose the Right K",
             ["<strong>Elbow Method</strong>: plot inertia vs K. Pick the 'elbow'.",
              "<strong>Silhouette Method</strong>: plot silhouette score vs K. Pick the peak.",
              "<strong>Domain knowledge</strong>: business logic often dictates K.",
              "<strong>Gap Statistic</strong>: compares inertia to random reference data.",
              "Always try multiple methods and triangulate."]),
        ]
        for title, bullets in topics2:
            with st.expander(f"⚙️ {title}"):
                for b in bullets:
                    st.markdown(f'<div style="font-size:0.83rem;color:#6b7090;line-height:1.9;padding:0.15rem 0">• {b}</div>', unsafe_allow_html=True)

    with level_tab3:
        topics3 = [
            ("Evaluation Beyond Silhouette",
             "Silhouette is not the only truth. Use a <strong>battery of metrics</strong>: "
             "Calinski-Harabasz (density ratio), Davies-Bouldin (average inter/intra ratio), "
             "Dunn Index (min inter / max intra). "
             "External metrics (if you have ground truth): ARI, NMI, Fowlkes-Mallows. "
             "Real insight comes from <em>business validation</em> — do the clusters make sense to domain experts?"),
            ("Preprocessing Choices Matter",
             "StandardScaler assumes Gaussian distribution. RobustScaler is better with outliers. "
             "PCA before clustering can help (removes noise) or hurt (loses cluster structure). "
             "For text: TF-IDF + cosine distance. "
             "For mixed types: Gower distance handles categorical + numeric natively. "
             "Pipelines ensure no data leakage between train and test."),
            ("DBSCAN Parameter Tuning",
             "ε: use the k-distance graph — plot sorted distances to kth nearest neighbour, "
             "pick ε at the 'elbow'. min_samples: rule of thumb = 2 × n_features. "
             "HDBSCAN (Hierarchical DBSCAN) removes the need to set ε at all — "
             "and is generally superior for real-world noisy data."),
            ("Production Deployment",
             "Save your preprocessor + model together in a Pipeline using sklearn.pipeline. "
             "Version models with MLflow or DVC. "
             "Monitor cluster drift over time — distributions shift, clusters can merge or split. "
             "For real-time: Mini-batch KMeans scales to streaming data. "
             "For explainability: SHAP values can explain cluster membership."),
        ]
        for title, body in topics3:
            with st.expander(f"🔬 {title}"):
                st.markdown(f'<div style="font-size:0.83rem;color:#6b7090;line-height:1.9">{body}</div>', unsafe_allow_html=True)

    with level_tab4:
        glossary = {
            "Clustering": "Unsupervised grouping of data points by similarity.",
            "Silhouette Score": "Measures cluster separation quality. Range: −1 to +1.",
            "Davies-Bouldin": "Lower = better. Average ratio of within-cluster scatter to inter-cluster distance.",
            "Calinski-Harabasz": "Higher = better. Ratio of between-cluster to within-cluster dispersion.",
            "Inertia": "KMeans objective: sum of squared distances from each point to its cluster centre.",
            "Elbow Method": "Plot inertia vs K — pick K where improvement slows (the 'elbow').",
            "PCA": "Dimensionality reduction that finds axes of maximum variance.",
            "t-SNE": "Non-linear dimensionality reduction for visualisation. Preserves local structure.",
            "Imputation": "Filling missing values using statistical methods (mean, median, KNN).",
            "StandardScaler": "Normalises to mean=0, std=1. Assumes roughly Gaussian distribution.",
            "RobustScaler": "Uses median and IQR — robust to outliers.",
            "One-Hot Encoding": "Converts categorical labels into binary columns.",
            "Outlier": "Data point far from others. Can distort KMeans; DBSCAN handles them natively.",
            "Hyperparameter": "Setting set before training (e.g. K in KMeans) — not learned from data.",
            "Supervised Learning": "Learning with labelled data (e.g. classification, regression).",
            "Unsupervised Learning": "Learning without labels — finds structure in data (e.g. clustering).",
            "Feature Engineering": "Creating or transforming features to improve model performance.",
            "Dendrogram": "Tree diagram showing hierarchical cluster merges.",
            "DBSCAN": "Density-Based Spatial Clustering. Groups dense regions, marks sparse points as noise.",
            "Variance Threshold": "Feature selection — removes features with very low variance (near-constant).",
        }
        c1, c2 = st.columns(2)
        items = list(glossary.items())
        for i, (term, defn) in enumerate(items):
            col = c1 if i % 2 == 0 else c2
            with col:
                st.markdown(f"""
                <div style="background:#111225;border:1px solid #1e2035;border-radius:6px;
                padding:0.7rem 0.9rem;margin-bottom:0.6rem;">
                  <div style="font-family:IBM Plex Mono;font-size:0.72rem;color:#22d3ee;
                  margin-bottom:0.2rem">{term}</div>
                  <div style="font-size:0.8rem;color:#6b7090;line-height:1.5">{defn}</div>
                </div>""", unsafe_allow_html=True)

    section("Your Pipeline Progress")
    steps_done = [
        ("📥 Load", st.session_state.df_raw is not None),
        ("🔍 EDA", st.session_state.eda_done),
        ("🧹 Clean", st.session_state.preprocessing_done),
        ("⚙️ Features", st.session_state.engineering_done),
        ("🤖 Cluster", st.session_state.clustering_done),
        ("📈 Results", st.session_state.clustering_done),
    ]
    cols = st.columns(len(steps_done))
    for i, (name, done) in enumerate(steps_done):
        clr = "#34d399" if done else "#2e3050"
        sym = "✓" if done else "○"
        cols[i].markdown(f"""
        <div style="text-align:center;padding:0.8rem 0.4rem;background:#111225;
        border:1px solid {'#34d399' if done else '#1e2035'};border-radius:8px;">
          <div style="font-size:1.2rem;color:{clr};margin-bottom:0.2rem">{sym}</div>
          <div style="font-family:IBM Plex Mono;font-size:0.6rem;color:{clr};
          letter-spacing:0.08em;text-transform:uppercase">{name}</div>
        </div>""", unsafe_allow_html=True)

    if st.button("🔄 Start New Analysis", type="primary"):
        for k, v in DEFAULTS.items():
            st.session_state[k] = v
        st.rerun()

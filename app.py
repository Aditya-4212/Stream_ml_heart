"""
ClusterMind — End-to-End Unsupervised ML Pipeline
AutoML + Interactive Learning Platform with Explanation System
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline as SKPipeline

# ─────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ClusterMind — ML Clustering Platform",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Syne:wght@400;600;700;800&family=Inter:wght@300;400;500;600&display=swap');

:root {
    --bg-primary: #080d18;
    --bg-card: #0f1623;
    --bg-elevated: #162032;
    --bg-input: #1a2640;
    --accent-cyan: #00d4ff;
    --accent-violet: #7c3aed;
    --accent-violet-light: #a78bfa;
    --accent-emerald: #10b981;
    --accent-amber: #f59e0b;
    --accent-rose: #f43f5e;
    --accent-blue: #3b82f6;
    --text-primary: #f1f5f9;
    --text-secondary: #cbd5e1;
    --text-muted: #64748b;
    --border: rgba(148,163,184,0.1);
    --border-hover: rgba(0,212,255,0.25);
    --glow-cyan: 0 0 20px rgba(0,212,255,0.15);
    --glow-violet: 0 0 20px rgba(124,58,237,0.2);
}

html, .stApp {
    background-color: var(--bg-primary) !important;
    font-family: 'Inter', sans-serif;
    color: var(--text-primary);
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1rem 2rem 3rem 2rem; max-width: 1400px; }

/* ── HEADER ── */
.app-header {
    background: linear-gradient(135deg, #0a0f1e 0%, #111827 50%, #130a2e 100%);
    border: 1px solid rgba(124,58,237,0.3);
    border-radius: 20px;
    padding: 2rem 2.8rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.app-header::before {
    content:'';
    position:absolute; top:-60%; right:-5%;
    width:500px; height:500px;
    background:radial-gradient(circle, rgba(124,58,237,0.18) 0%, transparent 65%);
    pointer-events:none;
}
.app-header::after {
    content:'';
    position:absolute; bottom:-40%; left:15%;
    width:350px; height:350px;
    background:radial-gradient(circle, rgba(0,212,255,0.1) 0%, transparent 65%);
    pointer-events:none;
}
.app-title {
    font-family:'Syne', sans-serif;
    font-size:2.6rem;
    font-weight:800;
    background:linear-gradient(90deg, var(--accent-cyan) 0%, var(--accent-violet-light) 100%);
    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;
    background-clip:text;
    margin:0; line-height:1.1; z-index:1; position:relative;
}
.app-tagline {
    color:var(--text-muted);
    font-size:0.95rem;
    font-weight:300;
    margin-top:0.5rem;
    z-index:1; position:relative;
    letter-spacing:0.02em;
}
.mode-pills {
    display:flex; gap:0.6rem; margin-top:1rem;
    z-index:1; position:relative;
}
.mode-pill {
    background:rgba(124,58,237,0.15);
    border:1px solid rgba(124,58,237,0.35);
    border-radius:20px;
    padding:0.3rem 0.9rem;
    font-size:0.75rem;
    font-family:'DM Mono', monospace;
    color:var(--accent-violet-light);
    letter-spacing:0.05em;
}

/* ── METRIC CARDS ── */
.metric-row { display:flex; gap:1rem; margin-bottom:1rem; flex-wrap:wrap; }
.metric-card {
    flex:1; min-width:140px;
    background:var(--bg-card);
    border:1px solid var(--border);
    border-radius:14px;
    padding:1.2rem 1.4rem;
    transition:border-color 0.25s, box-shadow 0.25s;
}
.metric-card:hover { border-color:var(--border-hover); box-shadow:var(--glow-cyan); }
.metric-card.highlight { border-color:rgba(124,58,237,0.4); }
.metric-label {
    font-family:'DM Mono', monospace;
    font-size:0.65rem;
    text-transform:uppercase;
    letter-spacing:0.12em;
    color:var(--text-muted);
    margin-bottom:0.35rem;
}
.metric-value {
    font-family:'Syne', sans-serif;
    font-size:1.9rem;
    font-weight:700;
    color:var(--accent-cyan);
    line-height:1;
}
.metric-value.violet { color:var(--accent-violet-light); }
.metric-value.emerald { color:var(--accent-emerald); }
.metric-value.amber { color:var(--accent-amber); }
.metric-sub {
    font-size:0.78rem;
    color:var(--text-muted);
    margin-top:0.3rem;
}

/* ── SECTION TITLES ── */
.section-title {
    font-family:'Syne', sans-serif;
    font-size:1.05rem;
    font-weight:700;
    color:var(--text-primary);
    margin:1.6rem 0 0.8rem 0;
    display:flex;
    align-items:center;
    gap:0.5rem;
}
.section-title::after {
    content:'';
    flex:1; height:1px;
    background:linear-gradient(90deg, rgba(148,163,184,0.15) 0%, transparent 100%);
    margin-left:0.5rem;
}

/* ── STAGE BADGE ── */
.stage-badge {
    display:inline-flex;
    align-items:center;
    gap:0.4rem;
    background:rgba(124,58,237,0.12);
    border:1px solid rgba(124,58,237,0.3);
    border-radius:20px;
    padding:0.3rem 0.9rem;
    font-family:'DM Mono', monospace;
    font-size:0.73rem;
    color:var(--accent-violet-light);
    margin-bottom:0.8rem;
    letter-spacing:0.04em;
}

/* ── EXPLAIN / CALLOUT BOXES ── */
.explain-box {
    background:linear-gradient(135deg, rgba(0,212,255,0.04), rgba(124,58,237,0.04));
    border:1px solid rgba(0,212,255,0.18);
    border-left:3px solid var(--accent-cyan);
    border-radius:0 12px 12px 0;
    padding:1.2rem 1.5rem;
    margin:0.8rem 0;
    color:var(--text-secondary);
    line-height:1.75;
    font-size:0.88rem;
}
.explain-box h4 {
    font-family:'Syne', sans-serif;
    color:var(--accent-cyan);
    margin:0 0 0.5rem 0;
    font-size:0.93rem;
    font-weight:600;
}

.tip-box {
    background:rgba(245,158,11,0.06);
    border:1px solid rgba(245,158,11,0.2);
    border-left:3px solid var(--accent-amber);
    border-radius:0 10px 10px 0;
    padding:0.8rem 1.2rem;
    margin:0.5rem 0;
    font-size:0.84rem;
    color:#fde68a;
    line-height:1.6;
}
.warn-box {
    background:rgba(244,63,94,0.06);
    border:1px solid rgba(244,63,94,0.2);
    border-left:3px solid var(--accent-rose);
    border-radius:0 10px 10px 0;
    padding:0.8rem 1.2rem;
    margin:0.5rem 0;
    font-size:0.84rem;
    color:#fda4af;
    line-height:1.6;
}
.success-box {
    background:rgba(16,185,129,0.06);
    border:1px solid rgba(16,185,129,0.2);
    border-left:3px solid var(--accent-emerald);
    border-radius:0 10px 10px 0;
    padding:0.8rem 1.2rem;
    margin:0.5rem 0;
    font-size:0.84rem;
    color:#6ee7b7;
    line-height:1.6;
}

/* ── DIVIDER ── */
.custom-divider {
    height:1px;
    background:linear-gradient(90deg, transparent, rgba(148,163,184,0.15), transparent);
    margin:1.5rem 0;
}

/* ── TABS ── */
.stTabs [data-baseweb="tab-list"] {
    background:var(--bg-card);
    border-radius:12px;
    padding:0.35rem;
    gap:0.25rem;
    border:1px solid var(--border);
    overflow-x:auto;
}
.stTabs [data-baseweb="tab"] {
    background:transparent !important;
    border-radius:8px !important;
    color:var(--text-muted) !important;
    font-family:'Inter', sans-serif !important;
    font-size:0.8rem !important;
    font-weight:500 !important;
    padding:0.4rem 0.85rem !important;
    white-space:nowrap;
}
.stTabs [aria-selected="true"] {
    background:var(--accent-violet) !important;
    color:white !important;
}

/* ── SIDEBAR ── */
[data-testid="stSidebar"] {
    background:var(--bg-card) !important;
    border-right:1px solid var(--border) !important;
}
[data-testid="stSidebar"] .block-container { padding:1rem 1rem 2rem 1rem; }
.sidebar-logo {
    font-family:'Syne', sans-serif;
    font-size:1.3rem;
    font-weight:800;
    background:linear-gradient(90deg, var(--accent-cyan), var(--accent-violet-light));
    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;
    background-clip:text;
    text-align:center;
    padding:0.5rem 0 1rem 0;
    border-bottom:1px solid var(--border);
    margin-bottom:1rem;
}

/* ── BUTTONS ── */
.stButton > button {
    background:linear-gradient(135deg, var(--accent-violet), #4f46e5) !important;
    color:white !important;
    border:none !important;
    border-radius:9px !important;
    font-family:'Inter', sans-serif !important;
    font-weight:500 !important;
    font-size:0.87rem !important;
    padding:0.5rem 1.3rem !important;
    transition:all 0.2s !important;
    letter-spacing:0.02em;
}
.stButton > button:hover {
    transform:translateY(-1px) !important;
    box-shadow:0 5px 18px rgba(124,58,237,0.45) !important;
}
.stDownloadButton > button {
    background:linear-gradient(135deg, var(--accent-emerald), #059669) !important;
    color:white !important;
    border:none !important;
    border-radius:9px !important;
}

/* ── WIDGETS ── */
.stSelectbox > div, .stMultiSelect > div {
    font-family:'Inter', sans-serif !important;
}
[data-baseweb="select"] > div {
    background:var(--bg-input) !important;
    border-color:var(--border) !important;
    border-radius:8px !important;
    color:var(--text-primary) !important;
}
.stSlider > div > div > div { background:var(--accent-violet) !important; }
.stCheckbox span { color:var(--text-secondary) !important; }
.stRadio div { color:var(--text-secondary) !important; }
input[type="number"], .stTextInput input {
    background:var(--bg-input) !important;
    border:1px solid var(--border) !important;
    border-radius:7px !important;
    color:var(--text-primary) !important;
}

/* ── EXPANDER ── */
[data-testid="stExpander"] {
    border:1px solid var(--border) !important;
    border-radius:10px !important;
    background:var(--bg-card) !important;
}
[data-testid="stExpander"] summary {
    font-family:'Inter', sans-serif !important;
    font-size:0.88rem !important;
    color:var(--text-secondary) !important;
    padding:0.7rem 1rem !important;
}

/* ── DATAFRAME ── */
.stDataFrame {
    border-radius:10px !important;
    border:1px solid var(--border) !important;
    overflow:hidden;
}

/* ── ALERTS ── */
.stSuccess > div { background:rgba(16,185,129,0.1) !important; border-color:var(--accent-emerald) !important; }
.stInfo > div { background:rgba(0,212,255,0.07) !important; border-color:var(--accent-cyan) !important; }
.stWarning > div { background:rgba(245,158,11,0.07) !important; border-color:var(--accent-amber) !important; }
.stError > div { background:rgba(244,63,94,0.07) !important; border-color:var(--accent-rose) !important; }

/* ── PROGRESS ── */
.stProgress > div > div { background:var(--accent-violet) !important; border-radius:4px; }

/* ── CHART BACKGROUND OVERRIDE ── */
.js-plotly-plot .plotly, .js-plotly-plot .plotly .main-svg {
    background:transparent !important;
}

/* Scrollable container */
.scroll-container { max-height:420px; overflow-y:auto; }

/* Step progress indicator */
.step-progress {
    display:flex; gap:0.3rem; margin:1rem 0; flex-wrap:wrap;
}
.step-dot {
    width:32px; height:32px;
    border-radius:50%;
    display:flex; align-items:center; justify-content:center;
    font-family:'DM Mono', monospace;
    font-size:0.7rem; font-weight:600;
    transition:all 0.2s;
}
.step-dot.done {
    background:var(--accent-emerald);
    color:white;
    border:none;
}
.step-dot.active {
    background:var(--accent-violet);
    color:white;
    border:none;
    box-shadow:0 0 12px rgba(124,58,237,0.5);
}
.step-dot.pending {
    background:var(--bg-elevated);
    color:var(--text-muted);
    border:1px solid var(--border);
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────
_defaults = {
    "df_raw": None,
    "df_cleaned": None,
    "df_engineered": None,
    "df_preprocessed_display": None,
    "X_processed": None,
    "cluster_labels": None,
    "pca_coords": None,
    # Online mode
    "auto_results": {},
    # Manual mode
    "manual_stage": 1,
    "cleaning_done": False,
    "engineering_done": False,
    "preprocessing_done": False,
    "model_trained": False,
    "numeric_cols": [],
    "categorical_cols": [],
    "scale_applied": False,
    "encode_applied": False,
    "chosen_model": "KMeans",
    "chosen_k": 3,
    "chosen_eps": 0.5,
    "chosen_min_samples": 5,
    "manual_score": None,
    "manual_n_clusters": None,
    "cleaning_action": "Fill with Mean / Median / Mode",
    "engineered_features": [],
    "manual_model_name": "KMeans",
}
for _k, _v in _defaults.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ─────────────────────────────────────────────────────────────────
# UTILITY FUNCTIONS
# ─────────────────────────────────────────────────────────────────

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(15,22,35,1)",
    plot_bgcolor="rgba(15,22,35,1)",
    font=dict(family="Inter", color="#94a3b8", size=12),
    title_font=dict(family="Syne", size=15, color="#f1f5f9"),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(148,163,184,0.15)", borderwidth=1),
    margin=dict(l=40, r=20, t=50, b=40),
    xaxis=dict(gridcolor="rgba(148,163,184,0.07)", zerolinecolor="rgba(148,163,184,0.12)"),
    yaxis=dict(gridcolor="rgba(148,163,184,0.07)", zerolinecolor="rgba(148,163,184,0.12)"),
)
CLUSTER_COLORS = px.colors.qualitative.Vivid + px.colors.qualitative.Bold

def detect_columns(df):
    num = df.select_dtypes(include=np.number).columns.tolist()
    cat = df.select_dtypes(exclude=np.number).columns.tolist()
    return num, cat

def safe_clean(df, method="Fill with Mean / Median / Mode"):
    """
    Robust cleaning that handles both np.nan and Python None in any column type.
    SimpleImputer fails silently on Python None in object columns — we use fillna(mode) instead.
    """
    df = df.copy()
    num, cat = detect_columns(df)
    if method == "Drop rows with missing values":
        df.dropna(inplace=True)
        df.reset_index(drop=True, inplace=True)
    else:
        if num:
            imp = SimpleImputer(strategy="median")
            df[num] = imp.fit_transform(df[num])
        if cat:
            # Use fillna(mode) — handles both np.nan and Python None correctly
            for c in cat:
                fill_val = df[c].dropna().mode()
                if len(fill_val) > 0:
                    df[c] = df[c].fillna(fill_val[0])
    return df

def build_X(df, num_cols, cat_cols, scale=True, encode=True):
    """Build preprocessed feature matrix. Returns (X, preprocessor_info_str)"""
    df = df.copy()
    transformers = []
    info_parts = []

    if num_cols:
        steps = [("imputer", SimpleImputer(strategy="median"))]
        if scale:
            steps.append(("scaler", StandardScaler()))
            info_parts.append(f"StandardScaler applied to {len(num_cols)} numeric feature(s)")
        else:
            info_parts.append(f"Numeric features used as-is (no scaling)")
        transformers.append(("num", SKPipeline(steps), num_cols))

    if cat_cols:
        if encode:
            steps = [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(sparse_output=False, handle_unknown="ignore")),
            ]
            info_parts.append(f"OneHotEncoder applied to {len(cat_cols)} categorical feature(s)")
            transformers.append(("cat", SKPipeline(steps), cat_cols))
        else:
            for c in cat_cols:
                df[c] = LabelEncoder().fit_transform(df[c].astype(str))
            transformers.append(("cat_pass", "passthrough", cat_cols))
            info_parts.append(f"Categorical features label-encoded (no one-hot)")

    if not transformers:
        return None, "No valid features found"

    ct = ColumnTransformer(transformers, remainder="drop")
    try:
        X = ct.fit_transform(df)
    except Exception as e:
        return None, f"Preprocessing error: {e}"
    return X, " | ".join(info_parts)

def safe_silhouette(X, labels):
    unique = list(set(labels))
    valid = [l for l in unique if l != -1]
    if len(valid) < 2:
        return -1.0
    try:
        mask = np.array(labels) != -1
        if mask.sum() <= len(valid):
            return -1.0
        return float(silhouette_score(X[mask], np.array(labels)[mask]))
    except Exception:
        return -1.0

def run_all_models(X, k=3, eps=0.5, min_samples=5):
    results = {}
    # KMeans
    try:
        m = KMeans(n_clusters=k, random_state=42, n_init=10)
        lbl = m.fit_predict(X)
        sc = safe_silhouette(X, lbl)
        results["KMeans"] = {"labels": lbl, "score": sc, "n_clusters": k, "model": m}
    except Exception as e:
        results["KMeans"] = {"labels": None, "score": -1, "n_clusters": 0, "error": str(e)}

    # DBSCAN
    try:
        m = DBSCAN(eps=eps, min_samples=min_samples)
        lbl = m.fit_predict(X)
        nc = len(set(lbl)) - (1 if -1 in lbl else 0)
        sc = safe_silhouette(X, lbl)
        results["DBSCAN"] = {"labels": lbl, "score": sc, "n_clusters": nc, "model": m}
    except Exception as e:
        results["DBSCAN"] = {"labels": None, "score": -1, "n_clusters": 0, "error": str(e)}

    # Agglomerative
    try:
        m = AgglomerativeClustering(n_clusters=k)
        lbl = m.fit_predict(X)
        sc = safe_silhouette(X, lbl)
        results["Agglomerative"] = {"labels": lbl, "score": sc, "n_clusters": k, "model": m}
    except Exception as e:
        results["Agglomerative"] = {"labels": None, "score": -1, "n_clusters": k, "error": str(e)}

    return results

def get_pca(X):
    n = min(2, X.shape[1], X.shape[0] - 1)
    if n < 1:
        return np.zeros((X.shape[0], 2))
    pca = PCA(n_components=n, random_state=42)
    coords = pca.fit_transform(X)
    if coords.shape[1] == 1:
        coords = np.hstack([coords, np.zeros((coords.shape[0], 1))])
    return coords

def fig_pca(coords, labels, title="PCA — 2D Cluster View"):
    lbl_str = [str(l) for l in labels]
    df_p = pd.DataFrame({"PC1": coords[:, 0], "PC2": coords[:, 1], "Cluster": lbl_str})
    fig = px.scatter(df_p, x="PC1", y="PC2", color="Cluster",
                     title=title,
                     color_discrete_sequence=CLUSTER_COLORS,
                     template="plotly_dark", hover_data={"PC1":":.3f","PC2":":.3f"})
    fig.update_traces(marker=dict(size=7, opacity=0.85, line=dict(width=0.4, color="rgba(255,255,255,0.3)")))
    fig.update_layout(**PLOTLY_LAYOUT)
    return fig

def fig_cluster_bar(labels, title="Cluster Distribution"):
    cnt = pd.Series(labels).value_counts().sort_index().reset_index()
    cnt.columns = ["Cluster", "Count"]
    cnt["Cluster"] = cnt["Cluster"].astype(str)
    fig = px.bar(cnt, x="Cluster", y="Count", color="Cluster",
                 title=title,
                 color_discrete_sequence=CLUSTER_COLORS,
                 template="plotly_dark",
                 text="Count")
    fig.update_traces(textposition="outside", marker_line_width=0)
    fig.update_layout(**PLOTLY_LAYOUT, showlegend=False)
    return fig

def fig_missing_heatmap(df):
    miss = df.isnull().astype(int)
    if miss.sum().sum() == 0:
        return None
    fig = px.imshow(miss.T, title="Missing Values Heatmap",
                    color_continuous_scale=[[0,"#162032"],[1,"#f43f5e"]],
                    template="plotly_dark",
                    labels=dict(x="Row Index", y="Column", color="Missing"))
    fig.update_layout(**PLOTLY_LAYOUT)
    return fig

def fig_feature_dist(df, col, labels, kind="box"):
    lbl_str = [str(l) for l in labels]
    df_p = df.copy()
    df_p["Cluster"] = lbl_str
    if kind == "box":
        fig = px.box(df_p, x="Cluster", y=col, color="Cluster",
                     title=f"{col} by Cluster",
                     color_discrete_sequence=CLUSTER_COLORS,
                     template="plotly_dark")
    else:
        fig = px.histogram(df_p, x=col, color="Cluster",
                           title=f"{col} Distribution by Cluster",
                           barmode="overlay",
                           color_discrete_sequence=CLUSTER_COLORS,
                           template="plotly_dark", opacity=0.75)
    fig.update_layout(**PLOTLY_LAYOUT)
    return fig

def download_csv(df):
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode()

def section(icon, text):
    st.markdown(f'<div class="section-title">{icon}&nbsp;{text}</div>', unsafe_allow_html=True)

def explain(title, content):
    st.markdown(f'<div class="explain-box"><h4>{title}</h4>{content}</div>', unsafe_allow_html=True)

def tip(content):
    st.markdown(f'<div class="tip-box">💡 <b>Tip:</b> {content}</div>', unsafe_allow_html=True)

def warn(content):
    st.markdown(f'<div class="warn-box">⚠️ <b>Warning:</b> {content}</div>', unsafe_allow_html=True)

def success_box(content):
    st.markdown(f'<div class="success-box">✅ {content}</div>', unsafe_allow_html=True)

def metric_card(label, value, sub="", color="cyan"):
    color_class = {"cyan":"","violet":" violet","emerald":" emerald","amber":" amber"}.get(color,"")
    sub_html = f"<div class='metric-sub'>{sub}</div>" if sub else ""
    return f"""<div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value{color_class}">{value}</div>
        {sub_html}
    </div>"""

def score_quality(s):
    if s > 0.7: return "Excellent 🟢", "emerald"
    if s > 0.5: return "Good 🟡", "amber"
    if s > 0.25: return "Moderate 🟠", "amber"
    return "Low 🔴 — try adjusting parameters", "cyan"

# ─────────────────────────────────────────────────────────────────
# EXPLANATION ENGINE
# ─────────────────────────────────────────────────────────────────

def generate_online_explanation(df_raw, auto_results):
    if df_raw is None:
        return "No dataset loaded yet."
    rows, cols = df_raw.shape
    num_c, cat_c = detect_columns(df_raw)
    missing_total = int(df_raw.isnull().sum().sum())
    best = auto_results.get("best_model", "N/A")
    score = auto_results.get("best_score", -1)
    k = auto_results.get("best_k", "N/A")
    km_score = auto_results.get("scores", {}).get("KMeans", -1)
    db_score = auto_results.get("scores", {}).get("DBSCAN", -1)
    agg_score = auto_results.get("scores", {}).get("Agglomerative", -1)
    sq, _ = score_quality(score) if score > -1 else ("N/A", "cyan")

    missing_action = ("Rows with missing values were **dropped**."
                      if missing_total == 0 else
                      f"**{missing_total}** missing values were filled: median for numeric, mode for categorical.")

    return f"""
## 🧠 Full Pipeline Explanation — AutoML Mode

---

### 1️⃣ Data Understanding
- **Dataset:** {rows:,} rows × {cols} columns
- **Numeric features ({len(num_c)}):** `{', '.join(num_c[:8])}{'...' if len(num_c)>8 else ''}`
- **Categorical features ({len(cat_c)}):** `{', '.join(cat_c[:5])}{'...' if len(cat_c)>5 else ''}` {' *(none)*' if not cat_c else ''}
- **Missing values:** {missing_total}
- *Why this matters:* Knowing your data shape, types, and quality prevents silent errors downstream. Numeric and categorical features need different treatment — you can't scale text, and you can't encode numbers as categories.

---

### 2️⃣ Data Cleaning
- **Action taken:** {missing_action}
- *Why this matters:* Most ML algorithms cannot handle `NaN` values. Dropping removes information; filling (imputation) preserves rows. Median is robust to outliers (unlike mean). Mode picks the most common category.

---

### 3️⃣ Preprocessing
- **Numeric features → StandardScaler:** Each feature is transformed to have mean=0, std=1. This prevents features with large ranges (e.g., income vs. age) from dominating the distance calculations clustering relies on.
- **Categorical features → OneHotEncoder:** Each category becomes a separate 0/1 column. Algorithms cannot compare "NY" to "LA" numerically — encoding gives them a meaningful numeric form.
- *Why this matters:* Clustering is distance-based. Unscaled or unencoded features create misleading distances.

---

### 4️⃣ Model Training — Three Models Compared

| Model | Silhouette Score | What it does |
|-------|-----------------|--------------|
| **KMeans** | {km_score:.4f} | Partitions data into K spherical clusters by minimizing intra-cluster variance. Fast and interpretable. |
| **DBSCAN** | {db_score:.4f} | Finds clusters of arbitrary shape based on density. Can detect noise/outliers as cluster -1. |
| **Agglomerative** | {agg_score:.4f} | Builds a hierarchy by progressively merging nearest points. Good for non-spherical patterns. |

---

### 5️⃣ Model Selection — Why **{best}** Won
- **Silhouette Score** measures how similar a point is to its own cluster vs. other clusters.
  - Range: **-1** (wrong cluster) → **0** (boundary) → **+1** (perfect cluster)
  - Score achieved: **{score:.4f}** → **{sq}**
- The model with the highest valid silhouette score was automatically selected.

---

### 6️⃣ Output Interpretation
- **{k} clusters** were identified in your dataset.
- Each data point is assigned a **Cluster ID** (0, 1, 2, ...).
- The **PCA plot** projects all features into 2D so you can visually inspect separation.
- **Cluster-wise averages** show the "personality" of each group — what makes them different.

---

### 7️⃣ Insights & Next Steps
- 🔍 **Examine cluster averages** — look for patterns (e.g., Cluster 0 = high income + low age?)
- 🔄 **Try different K values** — silhouette score is just one metric; domain knowledge matters
- ⚠️ If DBSCAN score is -1, it may have found only 1 cluster or all noise — try adjusting eps/min_samples
- 💾 **Download** the labeled dataset and use clusters as a new feature for supervised learning
- 🌍 **Real-world use cases:** Customer segmentation, anomaly detection, document grouping, patient stratification
"""

def generate_manual_explanation(stage, df_raw, df_cleaned, df_engineered,
                                 cleaning_action, engineered_features,
                                 scale_applied, encode_applied,
                                 chosen_model, chosen_k, chosen_eps, chosen_min_samples,
                                 score, n_clusters, numeric_cols, categorical_cols):
    parts = []

    if stage >= 1:
        if df_raw is not None:
            rows, cols = df_raw.shape
            num_c, cat_c = detect_columns(df_raw)
            miss = int(df_raw.isnull().sum().sum())
        else:
            rows, cols, num_c, cat_c, miss = "?", "?", [], [], 0
        parts.append(f"""
### 📂 Stage 1 — Data Understanding
**What was done:** Loaded and inspected the raw dataset.
- Dataset size: **{rows} rows × {cols} columns**
- Numeric features: **{len(num_c)}** → `{', '.join(num_c[:6])}{'...' if len(num_c)>6 else ''}`
- Categorical features: **{len(cat_c)}** → `{', '.join(cat_c[:4])}{'...' if len(cat_c)>4 else ''}` {' *(none)*' if not cat_c else ''}
- Missing values: **{miss}**

**Why:** Before touching any data, you must understand its structure. Shape tells you scale; types tell you what transformations are needed; missing values signal data quality issues.

**What to learn:** Always start with `df.info()`, `df.describe()`, and `df.isnull().sum()`. No assumptions — let the data tell you its story.
""")

    if stage >= 2:
        cleaned_rows = df_cleaned.shape[0] if df_cleaned is not None else "?"
        parts.append(f"""
### 🧹 Stage 2 — Data Cleaning
**What was done:** {cleaning_action}
- Resulting dataset: **{cleaned_rows} rows**

**Why:** Raw data is rarely clean. Missing values cause errors in most ML algorithms. Imputation (filling) is preferred over dropping when data is limited, but dropping is fine if missingness is minimal and random.

**What changed:** Missing cells were replaced with representative values. The dataset is now complete.

**What to learn:** `SimpleImputer(strategy='median')` is your friend for numeric data. Median is more robust than mean when outliers are present.
""")

    if stage >= 3:
        eng_list = engineered_features if engineered_features else ["No new features created"]
        parts.append(f"""
### ⚙️ Stage 3 — Feature Engineering
**What was done:** Attempted to create new derived features.
- New features: {', '.join(f'`{f}`' for f in eng_list)}

**Why:** Raw features don't always capture the full story. BMI = weight/height² is more meaningful than raw weight alone. Binning (e.g., age groups) can reveal categorical patterns in continuous data.

**What changed:** Additional columns may have been added to the dataset.

**What to learn:** Feature engineering is often where domain expertise pays off. The best features come from understanding what the data represents in the real world.

💡 *Tip: Even if no features were created here, think about ratios, differences, and groupings that might be meaningful for your domain.*
""")

    if stage >= 4:
        sc_txt = "✅ StandardScaler applied" if scale_applied else "❌ Scaling skipped"
        enc_txt = "✅ OneHotEncoder applied" if encode_applied else "❌ Encoding skipped (label encoding used)"
        parts.append(f"""
### 🔧 Stage 4 — Preprocessing
**What was done:**
- Numeric ({len(numeric_cols)} cols): {sc_txt}
- Categorical ({len(categorical_cols)} cols): {enc_txt}

**Why StandardScaler:** Clustering algorithms use distance metrics. If one feature ranges 0–10,000 and another 0–1, the first dominates every distance calculation. Scaling normalizes this.

**Why OneHotEncoder:** Machine learning models work with numbers. "Red", "Blue", "Green" → [1,0,0], [0,1,0], [0,0,1]. This preserves category identity without implying an order.

**What changed:** The data is now a purely numeric matrix ready for model input.

**What to learn:** Always scale before clustering. Always encode categoricals. These are non-negotiable preprocessing steps.

⚠️ *Warning: Never scale the target variable (but here we have none — unsupervised!)*
""")

    if stage >= 5:
        param_str = f"K={chosen_k}" if chosen_model == "KMeans" else (
            f"eps={chosen_eps}, min_samples={chosen_min_samples}" if chosen_model == "DBSCAN" else f"K={chosen_k}")
        parts.append(f"""
### 🎯 Stage 5 — Model Selection
**What was done:** Chose **{chosen_model}** with parameters: {param_str}

**Why {chosen_model}:**
- **KMeans** — Best for convex clusters of roughly equal size. Requires K upfront. Fast and interpretable.
- **DBSCAN** — Best when cluster shapes are irregular or you expect noise/outliers. Does not require K.
- **Agglomerative** — Best for hierarchical structure in data. Deterministic (same result every run).

**What to learn:** There's no universally best algorithm. Try all three and compare silhouette scores. Domain knowledge about expected cluster shapes guides selection.

💡 *Tip: Start with KMeans (k=3–5), compare with Agglomerative, use DBSCAN only if you suspect non-spherical clusters or outliers.*
""")

    if stage >= 6:
        parts.append(f"""
### 🚀 Stage 6 — Model Training
**What was done:** **{chosen_model}** was trained on the preprocessed feature matrix.
- The model assigned each data point to a cluster.
- Cluster IDs start from 0 (DBSCAN uses -1 for noise points).

**Why:** Training = the algorithm iteratively finds the best assignment of points to clusters by minimizing its objective function (e.g., KMeans minimizes within-cluster sum of squares).

**What changed:** Each row in your dataset now has a cluster label.

**What to learn:** In unsupervised learning, there's no "correct" answer — we evaluate quality using internal metrics like silhouette score.
""")

    if stage >= 7 and score is not None:
        sq, _ = score_quality(score)
        n_c = n_clusters if n_clusters else "?"
        parts.append(f"""
### 📊 Stage 7 — Evaluation
**What was done:** Silhouette score calculated.
- **Score: {score:.4f}** → **{sq}**
- **Clusters found: {n_c}**

**How silhouette score works:**
- For each point: compare its average distance to its own cluster vs. the nearest other cluster
- Formula: `s = (b - a) / max(a, b)` where a = intra-cluster dist, b = nearest-cluster dist
- Range: -1 (misclassified) → 0 (on boundary) → +1 (perfectly separated)

**What to learn:**
- Score > 0.5 = meaningful structure found
- Score < 0.25 = clusters may not be reliable
- Always check visually (PCA plot) even if the score is high — statistics can miss patterns

⚠️ *Warning: A high silhouette score doesn't mean the clusters are "correct" — just internally consistent. Always validate with domain knowledge.*
""")

    if stage >= 8:
        parts.append(f"""
### 🎨 Stage 8 — Visualization
**What was done:** PCA reduced the preprocessed data to 2 dimensions for plotting. Feature distributions per cluster were visualized.

**Why PCA for visualization:** You can't plot 10+ dimensions on a 2D screen. PCA finds the 2 axes that capture the most variance — think of it as finding the "most informative angle" to view the data.

**What the PCA plot tells you:**
- Well-separated colored blobs → clusters are meaningfully different
- Overlapping colors → clusters are similar / the model struggled
- The axes (PC1, PC2) are combinations of your original features

**Box/Histogram plots tell you:** How each feature's distribution differs between clusters — the "personality" of each group.

**What to learn:** Visualization is not optional. Numbers (like silhouette score) never tell the full story. Always look at the data.

🎓 *You have now completed the full ML clustering pipeline! From raw data to insights — this is exactly how data scientists work in production.*
""")

    header = f"## 🧠 Manual Mode — Pipeline Explanation (Stages 1–{stage})\n\n---\n"
    return header + "\n---\n".join(parts)

# ─────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown('<div class="sidebar-logo">🧠 ClusterMind</div>', unsafe_allow_html=True)

    st.markdown("**Select Mode**")
    app_mode = st.radio(
        "",
        ["🤖 Online Mode (AutoML)", "📚 Manual Mode (Learning)"],
        key="app_mode_radio",
        label_visibility="collapsed"
    )

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    st.markdown("**Upload Dataset**")
    uploaded_file = st.file_uploader(
        "Upload CSV file",
        type=["csv"],
        help="Any CSV with numeric or categorical columns",
        label_visibility="collapsed"
    )

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            if df.empty:
                st.error("The uploaded file is empty!")
            elif df.shape[1] < 2:
                st.warning("Dataset has only 1 column. Results may be limited.")
                st.session_state.df_raw = df
            else:
                # Reset derived states on new upload
                if st.session_state.df_raw is None or not df.equals(st.session_state.df_raw):
                    for k in ["df_cleaned","df_engineered","df_preprocessed_display","X_processed",
                              "cluster_labels","pca_coords","auto_results","manual_stage",
                              "cleaning_done","engineering_done","preprocessing_done","model_trained",
                              "engineered_features","manual_score","manual_n_clusters"]:
                        st.session_state[k] = _defaults.get(k)
                    st.session_state.manual_stage = 1
                st.session_state.df_raw = df
                num_c, cat_c = detect_columns(df)
                st.session_state.numeric_cols = num_c
                st.session_state.categorical_cols = cat_c
                st.success(f"✅ {df.shape[0]:,} rows × {df.shape[1]} cols")
        except Exception as e:
            st.error(f"Could not read file: {e}")

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    if app_mode == "🤖 Online Mode (AutoML)":
        st.markdown("**AutoML Parameters**")
        auto_k = st.slider("K (clusters)", 2, 10, 3, key="auto_k")
        auto_eps = st.slider("DBSCAN eps", 0.1, 5.0, 0.5, 0.1, key="auto_eps")
        auto_min_s = st.slider("DBSCAN min_samples", 2, 20, 5, key="auto_min_s")
    else:
        st.markdown("**Manual Mode Progress**")
        stages = ["Data Understanding","Data Cleaning","Feature Engineering",
                  "Preprocessing","Model Selection","Model Training","Evaluation","Visualization"]
        cur = st.session_state.manual_stage
        for i, s in enumerate(stages, 1):
            icon = "✅" if i < cur else ("▶️" if i == cur else "⚪")
            st.markdown(f"{icon} **{i}. {s}**" if i == cur else f"{icon} {i}. {s}")

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    st.markdown('<p style="color:#64748b;font-size:0.72rem;text-align:center;">ClusterMind v2.0 · AutoML + ML Tutor</p>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# MAIN HEADER
# ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
    <div class="app-title">🧠 ClusterMind</div>
    <div class="app-tagline">End-to-End Unsupervised ML Pipeline · AutoML Tool + Interactive Learning Platform</div>
    <div class="mode-pills">
        <span class="mode-pill">KMeans</span>
        <span class="mode-pill">DBSCAN</span>
        <span class="mode-pill">Agglomerative</span>
        <span class="mode-pill">PCA Visualization</span>
        <span class="mode-pill">Silhouette Evaluation</span>
        <span class="mode-pill">🧠 Explanation System</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# NO FILE UPLOADED — LANDING
# ─────────────────────────────────────────────────────────────────
if st.session_state.df_raw is None:
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.markdown("""
<div style="text-align:center; padding:3rem 1rem;">
    <div style="font-size:4rem; margin-bottom:1rem;">📁</div>
    <div style="font-family:'Syne',sans-serif; font-size:1.4rem; font-weight:700; color:#f1f5f9; margin-bottom:0.8rem;">
        Upload a CSV to Get Started
    </div>
    <div style="color:#64748b; font-size:0.92rem; line-height:1.7; max-width:400px; margin:0 auto;">
        ClusterMind works with <b style="color:#a78bfa">any dataset</b> — numeric, categorical, or mixed.<br><br>
        Choose <b style="color:#00d4ff">Online Mode</b> for fully automatic clustering, or<br>
        <b style="color:#10b981">Manual Mode</b> to learn the ML pipeline step-by-step.
    </div>
    <div style="margin-top:2rem; display:flex; gap:1rem; justify-content:center; flex-wrap:wrap;">
        <div style="background:#0f1623; border:1px solid rgba(0,212,255,0.2); border-radius:10px; padding:1rem 1.5rem; font-size:0.8rem; color:#94a3b8;">
            🤖 <b style="color:#f1f5f9">AutoML Mode</b><br>Upload → Auto-cluster → Insights
        </div>
        <div style="background:#0f1623; border:1px solid rgba(124,58,237,0.2); border-radius:10px; padding:1rem 1.5rem; font-size:0.8rem; color:#94a3b8;">
            📚 <b style="color:#f1f5f9">Learning Mode</b><br>8-stage guided ML tutorial
        </div>
        <div style="background:#0f1623; border:1px solid rgba(16,185,129,0.2); border-radius:10px; padding:1rem 1.5rem; font-size:0.8rem; color:#94a3b8;">
            🧠 <b style="color:#f1f5f9">Explanation System</b><br>Mentor-guided AI explanations
        </div>
    </div>
</div>
""", unsafe_allow_html=True)
    st.stop()

# ─────────────────────────────────────────────────────────────────
# FILE IS LOADED — MODE ROUTING
# ─────────────────────────────────────────────────────────────────
df_raw = st.session_state.df_raw
num_cols = st.session_state.numeric_cols
cat_cols = st.session_state.categorical_cols

# ═══════════════════════════════════════════════════════════════
# ONLINE MODE (AUTOML)
# ═══════════════════════════════════════════════════════════════
if app_mode == "🤖 Online Mode (AutoML)":

    tab_overview, tab_results, tab_viz, tab_download = st.tabs([
        "📊 Data Overview", "🤖 AutoML Results", "🎨 Visualizations", "💾 Download"
    ])

    # ── TAB 1: DATA OVERVIEW ──────────────────────────────────────
    with tab_overview:
        st.markdown('<div class="stage-badge">📂 Dataset Overview</div>', unsafe_allow_html=True)

        # Metrics row
        missing_total = int(df_raw.isnull().sum().sum())
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        with col_m1:
            st.markdown(metric_card("ROWS", f"{df_raw.shape[0]:,}", "data points"), unsafe_allow_html=True)
        with col_m2:
            st.markdown(metric_card("COLUMNS", str(df_raw.shape[1]), "features"), unsafe_allow_html=True)
        with col_m3:
            st.markdown(metric_card("NUMERIC", str(len(num_cols)), "features", color="violet"), unsafe_allow_html=True)
        with col_m4:
            st.markdown(metric_card("MISSING", str(missing_total),
                                    "values" if missing_total > 0 else "all clean ✅",
                                    color="amber" if missing_total > 0 else "emerald"), unsafe_allow_html=True)

        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

        # View controls
        col_v1, col_v2, col_v3 = st.columns([1,1,2])
        with col_v1:
            view_mode = st.selectbox("View", ["Head (50 rows)", "Full Dataset"], key="online_view_mode")
        with col_v2:
            col_filter = st.multiselect("Select columns", df_raw.columns.tolist(),
                                        default=df_raw.columns.tolist()[:min(8, len(df_raw.columns))],
                                        key="online_col_filter")
        with col_v3:
            search_val = st.text_input("🔍 Filter rows (search any column)", "", key="online_search")

        display_df = df_raw[col_filter] if col_filter else df_raw
        if search_val:
            mask = display_df.astype(str).apply(lambda x: x.str.contains(search_val, case=False, na=False)).any(axis=1)
            display_df = display_df[mask]

        if view_mode == "Head (50 rows)":
            display_df = display_df.head(50)

        section("📋", f"Dataset — {display_df.shape[0]:,} rows shown")
        st.dataframe(display_df, use_container_width=True, height=380)

        col_s1, col_s2 = st.columns(2)
        with col_s1:
            section("📈", "Statistical Summary")
            st.dataframe(df_raw.describe().round(3), use_container_width=True)
        with col_s2:
            section("🏷️", "Column Types & Missing Values")
            dtype_df = pd.DataFrame({
                "Type": df_raw.dtypes.astype(str),
                "Missing": df_raw.isnull().sum(),
                "Missing %": (df_raw.isnull().sum() / len(df_raw) * 100).round(1)
            })
            st.dataframe(dtype_df, use_container_width=True)

        # Missing heatmap
        if missing_total > 0:
            section("🔥", "Missing Values Heatmap")
            fig_miss = fig_missing_heatmap(df_raw)
            if fig_miss:
                st.plotly_chart(fig_miss, use_container_width=True)
        else:
            success_box("Dataset is complete — no missing values detected.")

    # ── TAB 2: AUTOML RESULTS ─────────────────────────────────────
    with tab_results:
        st.markdown('<div class="stage-badge">🤖 Automated ML Pipeline</div>', unsafe_allow_html=True)

        explain("What AutoML does", """
        The automated pipeline: <b>(1)</b> Cleans missing values → <b>(2)</b> Scales numeric features → 
        <b>(3)</b> Encodes categorical features → <b>(4)</b> Trains KMeans, DBSCAN, and Agglomerative → 
        <b>(5)</b> Selects the best model by silhouette score → <b>(6)</b> Returns cluster labels and insights.
        """)

        run_col, _ = st.columns([1, 3])
        with run_col:
            run_auto = st.button("▶ Run AutoML Pipeline", key="run_auto_btn")

        if run_auto:
            with st.spinner("Running full pipeline..."):
                progress = st.progress(0, text="Cleaning data...")
                df_c = safe_clean(df_raw)
                progress.progress(20, text="Building preprocessor...")

                X, info = build_X(df_c, num_cols, cat_cols, scale=True, encode=True)
                if X is None:
                    st.error(f"Preprocessing failed: {info}")
                    st.stop()

                progress.progress(50, text="Training models...")
                k_val = st.session_state.get("auto_k", 3)
                eps_val = st.session_state.get("auto_eps", 0.5)
                ms_val = st.session_state.get("auto_min_s", 5)
                results = run_all_models(X, k=k_val, eps=eps_val, min_samples=ms_val)
                progress.progress(80, text="Selecting best model...")

                scores = {m: r["score"] for m, r in results.items()}
                best_model = max(scores, key=scores.get)
                best_result = results[best_model]
                labels = best_result["labels"]

                coords = get_pca(X)

                st.session_state.auto_results = {
                    "best_model": best_model,
                    "best_score": best_result["score"],
                    "best_k": best_result["n_clusters"],
                    "all_results": results,
                    "scores": scores,
                    "labels": labels,
                    "coords": coords,
                    "X": X,
                    "df_cleaned": df_c,
                    "preprocess_info": info,
                }
                st.session_state.cluster_labels = labels
                st.session_state.pca_coords = coords
                progress.progress(100, text="Done!")
                st.success("✅ AutoML pipeline completed!")

        ar = st.session_state.auto_results
        if ar:
            best = ar["best_model"]
            score_val = ar["best_score"]
            k_val = ar["best_k"]
            sq, sq_color = score_quality(score_val)
            labels = ar["labels"]
            df_c = ar["df_cleaned"]

            st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
            section("🏆", "Best Model Results")

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.markdown(metric_card("BEST MODEL", best, "selected by silhouette score", color="violet"), unsafe_allow_html=True)
            with c2:
                st.markdown(metric_card("CLUSTERS", str(k_val), "groups found"), unsafe_allow_html=True)
            with c3:
                st.markdown(metric_card("SILHOUETTE", f"{score_val:.4f}", sq, color=sq_color), unsafe_allow_html=True)
            with c4:
                noise = int(np.sum(labels == -1)) if -1 in labels else 0
                st.markdown(metric_card("NOISE PTS", str(noise), "DBSCAN outliers" if noise > 0 else "no outliers", color="amber"), unsafe_allow_html=True)

            # All model comparison
            section("📊", "All Models Comparison")
            comp_data = []
            for mname, res in ar["all_results"].items():
                comp_data.append({
                    "Model": mname,
                    "Clusters Found": res["n_clusters"],
                    "Silhouette Score": f"{res['score']:.4f}" if res['score'] > -1 else "N/A",
                    "Selected": "✅ Best" if mname == best else ""
                })
            st.dataframe(pd.DataFrame(comp_data), use_container_width=True, hide_index=True)

            # Cluster distribution bar
            section("📈", "Cluster Distribution")
            st.plotly_chart(fig_cluster_bar(labels), use_container_width=True)

            # Cluster-wise feature summary
            section("🔍", "Cluster-wise Feature Averages")
            if num_cols:
                df_labeled = df_c[num_cols].copy()
                df_labeled["Cluster"] = labels
                summary = df_labeled.groupby("Cluster").agg(["mean","count"]).round(3)
                st.dataframe(summary, use_container_width=True)
            else:
                st.info("No numeric columns available for cluster summary.")

            # Explain button
            st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
            if st.button("🧠 Explain What Happened", key="explain_online_btn"):
                explanation = generate_online_explanation(df_raw, ar)
                with st.expander("📖 Detailed Pipeline Explanation — Click to expand", expanded=True):
                    st.markdown(explanation)
                    tip("Save this explanation as a reference for your portfolio or interviews!")
        else:
            explain("Ready to run", "Configure parameters in the sidebar, then click <b>▶ Run AutoML Pipeline</b> to start.")

    # ── TAB 3: VISUALIZATIONS ─────────────────────────────────────
    with tab_viz:
        st.markdown('<div class="stage-badge">🎨 Visualizations</div>', unsafe_allow_html=True)
        ar = st.session_state.auto_results

        if not ar:
            explain("No results yet", "Run the AutoML pipeline first to generate visualizations.")
        else:
            labels = ar["labels"]
            coords = ar["coords"]
            df_c = ar["df_cleaned"]

            section("🔮", "PCA — 2D Cluster Projection")
            explain("What is PCA?", "Principal Component Analysis projects your high-dimensional data onto 2 axes that capture the most variance. Well-separated blobs = distinct clusters. Overlapping = similar groups.")
            st.plotly_chart(fig_pca(coords, labels, "PCA — All Clusters (AutoML)"), use_container_width=True)

            st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
            section("📦", "Feature Distributions by Cluster")

            if num_cols:
                col_sel, type_sel = st.columns([2, 1])
                with col_sel:
                    feat = st.selectbox("Select feature", num_cols, key="viz_feat_sel")
                with type_sel:
                    plot_type = st.radio("Plot type", ["Box Plot", "Histogram"], key="viz_plot_type", horizontal=True)

                kind = "box" if plot_type == "Box Plot" else "hist"
                st.plotly_chart(fig_feature_dist(df_c, feat, labels, kind=kind), use_container_width=True)

                # Multi-feature overview
                section("🗂️", "All Numeric Features — Distribution Overview")
                n_feat = min(6, len(num_cols))
                cols_per_row = 2
                for i in range(0, n_feat, cols_per_row):
                    row_cols = st.columns(cols_per_row)
                    for j, rc in enumerate(row_cols):
                        idx = i + j
                        if idx < n_feat:
                            with rc:
                                st.plotly_chart(
                                    fig_feature_dist(df_c, num_cols[idx], labels, "box"),
                                    use_container_width=True
                                )
            else:
                st.info("No numeric features available for distribution plots.")

    # ── TAB 4: DOWNLOAD ───────────────────────────────────────────
    with tab_download:
        st.markdown('<div class="stage-badge">💾 Export Results</div>', unsafe_allow_html=True)
        ar = st.session_state.auto_results

        if not ar:
            explain("No results yet", "Run AutoML pipeline first to generate downloadable results.")
        else:
            labels = ar["labels"]
            df_export = df_raw.copy()
            df_export["Cluster_Label"] = labels

            section("📥", "Download Dataset with Cluster Labels")
            explain("What you're downloading",
                    "The original dataset with an additional <b>Cluster_Label</b> column containing the cluster ID for each row. "
                    "You can use this as a feature in supervised learning, for segmentation reports, or business analysis.")

            col_p, col_d = st.columns([2, 1])
            with col_p:
                st.dataframe(df_export.head(20), use_container_width=True)
            with col_d:
                st.markdown(metric_card("TOTAL ROWS", f"{len(df_export):,}", "with cluster labels"), unsafe_allow_html=True)
                st.markdown(metric_card("CLUSTERS", str(len(set(labels))), "unique groups"), unsafe_allow_html=True)
                st.download_button(
                    label="⬇️ Download CSV",
                    data=download_csv(df_export),
                    file_name="clustermind_results.csv",
                    mime="text/csv",
                    key="download_btn"
                )
                tip("Use this labeled dataset as training data for supervised classification models!")


# ═══════════════════════════════════════════════════════════════
# MANUAL MODE (LEARNING)
# ═══════════════════════════════════════════════════════════════
else:
    stage = st.session_state.manual_stage
    STAGE_NAMES = {1:"Data Understanding",2:"Data Cleaning",3:"Feature Engineering",
                   4:"Preprocessing",5:"Model Selection",6:"Model Training",
                   7:"Evaluation",8:"Visualization"}

    # ── STAGE PROGRESS BAR ──
    st.markdown('<div class="step-progress">' + "".join([
        f'<div class="step-dot {"done" if i < stage else ("active" if i == stage else "pending")}">{i}</div>'
        for i in range(1, 9)
    ]) + '</div>', unsafe_allow_html=True)

    st.markdown(f'<div class="stage-badge">📚 Stage {stage} of 8 — {STAGE_NAMES[stage]}</div>', unsafe_allow_html=True)

    # Navigation
    nav_left, nav_title, nav_right = st.columns([1, 4, 1])
    with nav_left:
        if stage > 1:
            if st.button("← Previous", key="prev_btn"):
                st.session_state.manual_stage -= 1
                st.rerun()
    with nav_title:
        st.markdown(f"<h3 style='text-align:center;font-family:Syne,sans-serif;color:#f1f5f9;margin:0;'>Stage {stage}: {STAGE_NAMES[stage]}</h3>", unsafe_allow_html=True)
    with nav_right:
        if stage < 8:
            if st.button("Next →", key="next_btn"):
                st.session_state.manual_stage += 1
                st.rerun()

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    # ── STAGE 1: DATA UNDERSTANDING ──────────────────────────────
    if stage == 1:
        explain("🎓 What you'll learn",
                "In this stage, you become familiar with the dataset: its size, types of features, "
                "statistical summaries, and data quality (missing values). This is the foundation of every ML project.")

        section("📊", "Dataset Metrics")
        missing_total = int(df_raw.isnull().sum().sum())
        c1, c2, c3, c4 = st.columns(4)
        with c1: st.markdown(metric_card("ROWS", f"{df_raw.shape[0]:,}", "observations"), unsafe_allow_html=True)
        with c2: st.markdown(metric_card("COLUMNS", str(df_raw.shape[1]), "features"), unsafe_allow_html=True)
        with c3: st.markdown(metric_card("NUMERIC", str(len(num_cols)), "features", color="violet"), unsafe_allow_html=True)
        with c4: st.markdown(metric_card("MISSING", str(missing_total), "total values", color="amber" if missing_total > 0 else "emerald"), unsafe_allow_html=True)

        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

        col_v1, col_v2 = st.columns([1, 1])
        with col_v1:
            view_mode_m = st.selectbox("View Mode", ["Head (50 rows)", "Full Dataset"], key="m1_view")
        with col_v2:
            col_filter_m = st.multiselect("Columns to display", df_raw.columns.tolist(),
                                          default=df_raw.columns.tolist()[:min(8, len(df_raw.columns))],
                                          key="m1_cols")
        search_m = st.text_input("🔍 Search rows", "", key="m1_search")

        disp = df_raw[col_filter_m] if col_filter_m else df_raw
        if search_m:
            mask = disp.astype(str).apply(lambda x: x.str.contains(search_m, case=False, na=False)).any(axis=1)
            disp = disp[mask]
        if view_mode_m == "Head (50 rows)":
            disp = disp.head(50)

        section("📋", f"Dataset — {disp.shape[0]:,} rows")
        st.dataframe(disp, use_container_width=True, height=350)

        col_s1, col_s2 = st.columns(2)
        with col_s1:
            section("📈", "Statistical Summary (.describe)")
            st.dataframe(df_raw.describe().round(3), use_container_width=True)
            explain("What this shows",
                    "<b>count</b>=non-null entries · <b>mean</b>=average · <b>std</b>=spread · "
                    "<b>min/max</b>=range · <b>25%/50%/75%</b>=quartiles (distribution shape)")
        with col_s2:
            section("🏷️", "Column Types")
            dtype_df = pd.DataFrame({
                "Type": df_raw.dtypes.astype(str),
                "Non-Null": df_raw.count(),
                "Missing": df_raw.isnull().sum(),
                "Missing %": (df_raw.isnull().sum() / len(df_raw) * 100).round(1)
            })
            st.dataframe(dtype_df, use_container_width=True)

        if missing_total > 0:
            section("🔥", "Missing Values Heatmap")
            fig_m = fig_missing_heatmap(df_raw)
            if fig_m:
                st.plotly_chart(fig_m, use_container_width=True)
            warn(f"{missing_total} missing values found. You'll handle these in Stage 2.")
        else:
            success_box("No missing values — dataset is complete!")

        tip("Numeric features need scaling; categorical features need encoding. Make a mental note of each column type.")

    # ── STAGE 2: DATA CLEANING ────────────────────────────────────
    elif stage == 2:
        explain("🎓 What you'll learn",
                "Real-world data is messy. Missing values must be handled before any model can run. "
                "You'll choose a strategy: <b>drop</b> incomplete rows, or <b>fill</b> (impute) them.")

        missing_total = int(df_raw.isnull().sum().sum())
        if missing_total == 0:
            success_box("This dataset has no missing values — cleaning is already done! You can proceed to Stage 3.")
            st.dataframe(df_raw.head(20), use_container_width=True)
            st.session_state.df_cleaned = df_raw.copy()
            st.session_state.cleaning_done = True
        else:
            section("⚙️", "Choose Cleaning Strategy")
            method = st.radio(
                "Missing value treatment:",
                ["Fill with Mean / Median / Mode", "Drop rows with missing values"],
                key="cleaning_method_radio",
                help="Fill: impute missing cells. Drop: remove incomplete rows."
            )
            explain("Imputation vs Dropping",
                    "<b>Fill (Impute):</b> Replaces NaN with median (numeric) or mode (categorical). "
                    "Preserves all rows — preferred when data is limited.<br>"
                    "<b>Drop:</b> Removes any row with at least one NaN. Simple but loses information.")

            if st.button("🧹 Apply Cleaning", key="apply_cleaning_btn"):
                df_cl = safe_clean(df_raw, method)
                st.session_state.df_cleaned = df_cl
                st.session_state.cleaning_action = method
                st.session_state.cleaning_done = True
                st.success(f"✅ Cleaning applied! {df_raw.shape[0] - df_cl.shape[0]} rows removed." if "Drop" in method else "✅ Missing values filled!")

            if st.session_state.df_cleaned is not None:
                section("📊", "Before vs After")
                col_b, col_a = st.columns(2)
                with col_b:
                    st.markdown("**Before Cleaning**")
                    st.dataframe(df_raw.head(15), use_container_width=True)
                    st.caption(f"Shape: {df_raw.shape[0]} × {df_raw.shape[1]} | Missing: {int(df_raw.isnull().sum().sum())}")
                with col_a:
                    st.markdown("**After Cleaning**")
                    st.dataframe(st.session_state.df_cleaned.head(15), use_container_width=True)
                    st.caption(f"Shape: {st.session_state.df_cleaned.shape[0]} × {st.session_state.df_cleaned.shape[1]} | Missing: {int(st.session_state.df_cleaned.isnull().sum().sum())}")
            else:
                explain("Waiting for action", "Click <b>Apply Cleaning</b> to see the before/after comparison.")

        tip("In production, always understand WHY values are missing before choosing a strategy. Random missingness → impute. Structural missingness → investigate first.")

    # ── STAGE 3: FEATURE ENGINEERING ─────────────────────────────
    elif stage == 3:
        if st.session_state.df_cleaned is None:
            st.session_state.df_cleaned = df_raw.copy()
        df_eng_base = st.session_state.df_cleaned.copy()

        explain("🎓 What you'll learn",
                "Feature engineering creates new, more informative features from existing ones. "
                "A model can't know that BMI = weight/height² is meaningful — you have to create it. "
                "Good features often matter more than model choice.")

        section("🔬", "Automatic Feature Detection")
        eligible_pairs = []
        for c in num_cols:
            if c in df_eng_base.columns:
                eligible_pairs.append(c)

        new_features = []

        # BMI detection
        has_bmi = False
        weight_col = next((c for c in df_eng_base.columns if "weight" in c.lower()), None)
        height_col = next((c for c in df_eng_base.columns if "height" in c.lower()), None)
        if weight_col and height_col:
            create_bmi = st.checkbox(f"Create BMI from `{weight_col}` / `{height_col}`²", key="eng_bmi")
            if create_bmi:
                try:
                    df_eng_base["BMI"] = (df_eng_base[weight_col] /
                                          (df_eng_base[height_col] / 100) ** 2).round(2)
                    new_features.append("BMI")
                    has_bmi = True
                    success_box("BMI created successfully!")
                except Exception as e:
                    warn(f"Could not create BMI: {e}")

        # Ratio features
        section("➗", "Create Ratio Features")
        col_r1, col_r2 = st.columns(2)
        with col_r1:
            num_available = [c for c in num_cols if c in df_eng_base.columns]
            ratio_num = st.selectbox("Numerator column", ["(none)"] + num_available, key="eng_ratio_num")
        with col_r2:
            ratio_den = st.selectbox("Denominator column", ["(none)"] + num_available, key="eng_ratio_den")
        if ratio_num != "(none)" and ratio_den != "(none)" and ratio_num != ratio_den:
            if st.button("➕ Create Ratio Feature", key="eng_ratio_btn"):
                feat_name = f"{ratio_num}_per_{ratio_den}"
                try:
                    df_eng_base[feat_name] = (df_eng_base[ratio_num] / df_eng_base[ratio_den].replace(0, np.nan)).round(4)
                    new_features.append(feat_name)
                    success_box(f"Ratio feature `{feat_name}` created!")
                except Exception as e:
                    warn(f"Could not create ratio: {e}")

        # Binning
        section("📦", "Bin a Numeric Feature (Categorize)")
        bin_col = st.selectbox("Column to bin", ["(none)"] + [c for c in num_cols if c in df_eng_base.columns], key="eng_bin_col")
        if bin_col != "(none)":
            n_bins = st.slider("Number of bins", 2, 10, 4, key="eng_bin_n")
            if st.button("🗂️ Apply Binning", key="eng_bin_btn"):
                try:
                    bin_name = f"{bin_col}_bin"
                    df_eng_base[bin_name] = pd.cut(df_eng_base[bin_col], bins=n_bins,
                                                    labels=[f"B{i+1}" for i in range(n_bins)])
                    df_eng_base[bin_name] = df_eng_base[bin_name].astype(str)
                    new_features.append(bin_name)
                    success_box(f"Binned `{bin_col}` into {n_bins} bins as `{bin_name}`")
                except Exception as e:
                    warn(f"Binning failed: {e}")

        # Skip option
        if not new_features:
            st.info("No new features created. That's fine! You can skip this stage and proceed to Preprocessing.")

        st.session_state.df_engineered = df_eng_base
        st.session_state.engineered_features = new_features
        st.session_state.engineering_done = True

        if new_features:
            section("✅", f"Updated Dataset — {len(new_features)} new feature(s) added")
            st.dataframe(df_eng_base.head(20), use_container_width=True)

        tip("The best features come from domain knowledge. Ask: what combinations of raw measurements are physically or logically meaningful?")
        explain("🌍 Real-world example",
                "In healthcare: BMI, pulse pressure, MAP (Mean Arterial Pressure) are all engineered features "
                "that carry more diagnostic value than raw weight/height/BP alone.")

    # ── STAGE 4: PREPROCESSING ───────────────────────────────────
    elif stage == 4:
        if st.session_state.df_engineered is None:
            st.session_state.df_engineered = (st.session_state.df_cleaned or df_raw).copy()
        df_for_prep = st.session_state.df_engineered.copy()

        # Refresh column detection after engineering
        num_c_updated, cat_c_updated = detect_columns(df_for_prep)

        explain("🎓 What you'll learn",
                "Preprocessing transforms raw data into a format ML models can learn from. "
                "Scaling normalizes feature ranges; encoding converts categories to numbers.")

        section("📋", "Feature Inventory")
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            st.markdown("**Numeric Features**")
            if num_c_updated:
                num_df = pd.DataFrame({"Feature": num_c_updated, "Type": "Numeric",
                                       "Action": "StandardScaler" })
                st.dataframe(num_df, use_container_width=True, hide_index=True)
            else:
                st.info("No numeric features detected.")
        with col_p2:
            st.markdown("**Categorical Features**")
            if cat_c_updated:
                cat_df = pd.DataFrame({"Feature": cat_c_updated, "Type": "Categorical",
                                       "Action": "OneHotEncoder" })
                st.dataframe(cat_df, use_container_width=True, hide_index=True)
            else:
                st.info("No categorical features detected.")

        section("⚙️", "Transformation Options")
        col_sc, col_en = st.columns(2)
        with col_sc:
            apply_scale = st.checkbox("Apply StandardScaler (Numeric)", value=True, key="prep_scale")
            explain("StandardScaler",
                    "Transforms each numeric feature to mean=0, std=1. Formula: <code>z = (x - μ) / σ</code>.<br>"
                    "Critical for distance-based algorithms like KMeans and DBSCAN.")
        with col_en:
            apply_encode = st.checkbox("Apply OneHotEncoder (Categorical)", value=True, key="prep_encode")
            explain("OneHotEncoder",
                    "Converts each category into a binary column. E.g., color=[Red,Blue,Green] → "
                    "color_Red, color_Blue, color_Green (each 0 or 1).<br>"
                    "Prevents the model from treating categories as ordered.")

        if st.button("🔧 Apply Preprocessing", key="apply_prep_btn"):
            X_out, info_str = build_X(df_for_prep, num_c_updated, cat_c_updated,
                                      scale=apply_scale, encode=apply_encode)
            if X_out is None:
                st.error(f"Preprocessing failed: {info_str}")
            else:
                st.session_state.X_processed = X_out
                st.session_state.scale_applied = apply_scale
                st.session_state.encode_applied = apply_encode
                st.session_state.numeric_cols = num_c_updated
                st.session_state.categorical_cols = cat_c_updated
                st.session_state.preprocessing_done = True
                st.session_state.df_preprocessed_display = pd.DataFrame(
                    X_out, columns=[f"feat_{i}" for i in range(X_out.shape[1])])
                success_box(f"Preprocessing done! Output shape: {X_out.shape[0]} × {X_out.shape[1]}. {info_str}")

        if st.session_state.preprocessing_done and st.session_state.X_processed is not None:
            section("📊", "Before vs After Preprocessing")
            col_b, col_a = st.columns(2)
            with col_b:
                st.markdown("**Before (first 5 rows, selected cols)**")
                cols_show = (num_c_updated + cat_c_updated)[:6]
                st.dataframe(df_for_prep[cols_show].head(5).round(3), use_container_width=True)
            with col_a:
                st.markdown("**After (preprocessed matrix)**")
                st.dataframe(st.session_state.df_preprocessed_display.head(5).round(4),
                             use_container_width=True)
            tip("Notice how numeric values are now centered around 0 after scaling. Categorical columns became multiple 0/1 columns.")
        elif not st.session_state.preprocessing_done:
            explain("Waiting for action", "Click <b>Apply Preprocessing</b> to transform the data.")

        warn("Never apply scaling/encoding based on test set statistics — always fit on training data only (in supervised learning). Here we fit on all data since it's unsupervised.")

    # ── STAGE 5: MODEL SELECTION ──────────────────────────────────
    elif stage == 5:
        explain("🎓 What you'll learn",
                "Choosing the right clustering algorithm is critical. Each has different assumptions "
                "about cluster shape, size, and density. You'll configure parameters interactively.")

        section("🔍", "Algorithm Comparison")
        algo_data = pd.DataFrame({
            "Algorithm": ["KMeans", "DBSCAN", "Agglomerative"],
            "Best For": ["Spherical, equal-size clusters", "Arbitrary shapes, noise detection", "Hierarchical structure"],
            "Requires K?": ["Yes", "No (auto)", "Yes"],
            "Handles Noise?": ["No", "Yes (as -1)", "No"],
            "Speed": ["Fast", "Medium", "Slow (large data)"],
        })
        st.dataframe(algo_data, use_container_width=True, hide_index=True)

        section("⚙️", "Configure Your Model")
        chosen_model = st.selectbox("Select algorithm", ["KMeans", "DBSCAN", "Agglomerative"],
                                    key="m5_model_select")
        st.session_state.chosen_model = chosen_model

        if chosen_model == "KMeans":
            explain("KMeans",
                    "<b>How it works:</b> Randomly places K centroids, assigns each point to nearest centroid, "
                    "recalculates centroids, repeats until stable.<br>"
                    "<b>Objective:</b> Minimize within-cluster sum of squares (WCSS).<br>"
                    "<b>Assumption:</b> Clusters are convex and roughly equally sized.")
            k = st.slider("K — Number of clusters", 2, 15, 3, key="m5_k")
            st.session_state.chosen_k = k
            tip("Try K=2 to 8 and compare silhouette scores. The Elbow Method (WCSS vs K) can also guide selection.")
            warn("K=1 is meaningless for clustering. K too large → noise. K too small → merged groups.")

        elif chosen_model == "DBSCAN":
            explain("DBSCAN",
                    "<b>How it works:</b> A point is a 'core point' if it has ≥ min_samples neighbors within radius eps. "
                    "Core points and their reachable neighbors form a cluster. Points in no cluster are 'noise' (label -1).<br>"
                    "<b>No K needed!</b> Clusters can have any shape.")
            c1, c2 = st.columns(2)
            with c1:
                eps = st.slider("eps — neighborhood radius", 0.1, 10.0, 0.5, 0.05, key="m5_eps")
                st.session_state.chosen_eps = eps
            with c2:
                min_s = st.slider("min_samples — core point threshold", 2, 30, 5, key="m5_mins")
                st.session_state.chosen_min_samples = min_s
            tip("If too many noise points: increase eps or decrease min_samples. If everything is one cluster: decrease eps.")
            warn("DBSCAN is sensitive to scale — always preprocess before using it.")

        elif chosen_model == "Agglomerative":
            explain("Agglomerative Clustering",
                    "<b>How it works:</b> Starts with each point as its own cluster. Iteratively merges "
                    "the two closest clusters until K clusters remain. Builds a dendrogram (tree).<br>"
                    "<b>Linkage:</b> Ward linkage minimizes within-cluster variance (default, usually best).")
            k = st.slider("K — Number of clusters", 2, 15, 3, key="m5_k_agg")
            st.session_state.chosen_k = k
            tip("Agglomerative is deterministic — same result every run. Good for small-to-medium datasets.")

    # ── STAGE 6: MODEL TRAINING ───────────────────────────────────
    elif stage == 6:
        if st.session_state.X_processed is None:
            warn("Preprocessing not done! Please go back to Stage 4 and apply preprocessing first.")
            st.stop()

        X = st.session_state.X_processed
        model = st.session_state.chosen_model
        k = st.session_state.chosen_k
        eps = st.session_state.chosen_eps
        min_s = st.session_state.chosen_min_samples

        explain("🎓 What you'll learn",
                f"You've configured <b>{model}</b>. Now we train it and assign cluster labels to every data point.")

        section("🚀", f"Training {model}")
        st.markdown(f"""
<div style="background:var(--bg-elevated);border:1px solid var(--border);border-radius:12px;padding:1rem 1.5rem;margin-bottom:1rem;">
<b style="color:#a78bfa">Model:</b> <span style="color:#f1f5f9">{model}</span> &nbsp;|&nbsp;
<b style="color:#a78bfa">Parameters:</b> <span style="color:#f1f5f9">{"K=" + str(k) if model != "DBSCAN" else f"eps={eps}, min_samples={min_s}"}</span> &nbsp;|&nbsp;
<b style="color:#a78bfa">Data shape:</b> <span style="color:#f1f5f9">{X.shape[0]} × {X.shape[1]}</span>
</div>
""", unsafe_allow_html=True)

        if st.button(f"▶ Train {model}", key="train_model_btn"):
            with st.spinner(f"Training {model}..."):
                try:
                    if model == "KMeans":
                        m = KMeans(n_clusters=k, random_state=42, n_init=10)
                        labels = m.fit_predict(X)
                        n_cl = k
                    elif model == "DBSCAN":
                        m = DBSCAN(eps=eps, min_samples=min_s)
                        labels = m.fit_predict(X)
                        n_cl = len(set(labels)) - (1 if -1 in labels else 0)
                    else:
                        m = AgglomerativeClustering(n_clusters=k)
                        labels = m.fit_predict(X)
                        n_cl = k

                    score = safe_silhouette(X, labels)
                    coords = get_pca(X)

                    st.session_state.cluster_labels = labels
                    st.session_state.pca_coords = coords
                    st.session_state.manual_score = score
                    st.session_state.manual_n_clusters = n_cl
                    st.session_state.manual_model_name = model
                    st.session_state.model_trained = True
                    st.success(f"✅ {model} trained! Found {n_cl} cluster(s). Silhouette: {score:.4f}")

                    # Show cluster assignment preview
                    section("🏷️", "Cluster Assignment Preview")
                    df_preview = (st.session_state.df_engineered or df_raw).copy()
                    df_preview["Cluster"] = labels
                    st.dataframe(df_preview.head(20), use_container_width=True)

                    # Quick distribution
                    st.plotly_chart(fig_cluster_bar(labels, f"{model} Cluster Distribution"), use_container_width=True)

                except Exception as e:
                    st.error(f"Training failed: {e}")
                    st.stop()
        elif st.session_state.model_trained and st.session_state.cluster_labels is not None:
            labels = st.session_state.cluster_labels
            score = st.session_state.manual_score
            n_cl = st.session_state.manual_n_clusters

            success_box(f"Model already trained — {n_cl} cluster(s), silhouette={score:.4f}. Proceed to Evaluation →")
            df_preview = (st.session_state.df_engineered or df_raw).copy()
            df_preview["Cluster"] = labels
            st.dataframe(df_preview.head(20), use_container_width=True)
            st.plotly_chart(fig_cluster_bar(labels, f"{model} Cluster Distribution"), use_container_width=True)
        else:
            explain("Ready to train", f"Click <b>▶ Train {model}</b> to assign cluster labels.")

        tip("If training is fast, it doesn't mean results are good! Always evaluate with silhouette score and visual inspection.")

    # ── STAGE 7: EVALUATION ───────────────────────────────────────
    elif stage == 7:
        if not st.session_state.model_trained or st.session_state.cluster_labels is None:
            warn("Model not trained yet! Please complete Stage 6 first.")
            st.stop()

        labels = st.session_state.cluster_labels
        score = st.session_state.manual_score
        n_cl = st.session_state.manual_n_clusters
        model = st.session_state.manual_model_name
        sq, sq_color = score_quality(score) if score > -1 else ("N/A", "cyan")

        explain("🎓 What you'll learn",
                "Evaluating clustering is harder than supervised learning — there's no ground truth label. "
                "We use the <b>Silhouette Score</b>, an internal metric that measures cluster quality.")

        section("📊", "Evaluation Metrics")
        c1, c2, c3 = st.columns(3)
        with c1: st.markdown(metric_card("MODEL", model, "used", color="violet"), unsafe_allow_html=True)
        with c2: st.markdown(metric_card("CLUSTERS", str(n_cl), "groups found"), unsafe_allow_html=True)
        with c3: st.markdown(metric_card("SILHOUETTE", f"{score:.4f}" if score > -1 else "N/A", sq, color=sq_color), unsafe_allow_html=True)

        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

        section("📐", "Understanding the Silhouette Score")
        explain("Silhouette Score Deep Dive", f"""
        <b>Formula:</b> <code>s(i) = (b(i) - a(i)) / max(a(i), b(i))</code><br><br>
        <b>a(i)</b> = average distance from point i to all other points in its cluster (intra-cluster)<br>
        <b>b(i)</b> = average distance from point i to all points in the nearest other cluster<br><br>
        <b>Interpretation:</b><br>
        • <b>+1.0</b> = perfect: point is deep inside its cluster, far from others<br>
        • <b>0.0</b> = on the boundary between two clusters<br>
        • <b>-1.0</b> = likely mis-clustered: closer to another cluster than its own<br><br>
        <b>Your score: {score:.4f}</b> → <b>{sq}</b>
        """)

        if score < 0:
            warn("Score is -1 or invalid — this often means DBSCAN found only 1 cluster or all noise. Try adjusting eps/min_samples in Stage 5.")
        elif score < 0.25:
            warn("Low silhouette score. The clusters may not be well-separated. Try: different K, different algorithm, or more feature engineering.")
        elif score > 0.5:
            success_box("Good clustering! The data shows meaningful separation between clusters.")

        # Cluster statistics
        section("📋", "Cluster-wise Statistics")
        df_base = (st.session_state.df_engineered or df_raw).copy()
        num_c_eval, _ = detect_columns(df_base)
        if num_c_eval:
            df_base["Cluster"] = labels
            summary = df_base.groupby("Cluster")[num_c_eval].agg(["mean","std","count"]).round(3)
            st.dataframe(summary, use_container_width=True)
        else:
            st.info("No numeric features for cluster statistics.")

        # Noise info for DBSCAN
        if -1 in labels:
            noise_count = int(np.sum(np.array(labels) == -1))
            warn(f"DBSCAN found {noise_count} noise points (Cluster = -1). These are outliers not belonging to any cluster.")

        tip("High within-cluster similarity + low between-cluster similarity = good clustering. The silhouette score captures exactly this.")
        tip("Also try the Elbow Method (WCSS vs K for KMeans) and Davies-Bouldin Index for additional validation.")

    # ── STAGE 8: VISUALIZATION ────────────────────────────────────
    elif stage == 8:
        if not st.session_state.model_trained or st.session_state.cluster_labels is None:
            warn("Model not trained yet! Please complete Stages 4–6 first.")
            st.stop()

        labels = st.session_state.cluster_labels
        coords = st.session_state.pca_coords
        df_base = (st.session_state.df_engineered or df_raw).copy()
        num_c_viz, _ = detect_columns(df_base)
        model = st.session_state.manual_model_name

        explain("🎓 What you'll learn",
                "Visualization is the final validation step. Numbers tell you scores; plots tell you the story. "
                "PCA reduces dimensions to make high-dimensional clusters visible in 2D.")

        # PCA
        section("🔮", "PCA — 2D Cluster Projection")
        explain("Interpreting the PCA Plot",
                "Each dot = one data point. Colors = cluster assignment. "
                "Well-separated color blobs → strong clusters. Overlapping colors → weak separation. "
                "PCA axes (PC1, PC2) are linear combinations of your original features.")
        if coords is not None:
            st.plotly_chart(fig_pca(coords, labels, f"PCA — {model} Clusters"), use_container_width=True)
        else:
            warn("PCA coordinates not available. Retrain model in Stage 6.")

        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

        # Cluster bar
        section("📊", "Cluster Size Distribution")
        st.plotly_chart(fig_cluster_bar(labels, f"{model} — Cluster Distribution"), use_container_width=True)

        # Feature distribution
        if num_c_viz:
            section("📦", "Feature Distributions by Cluster")
            col_f, col_t = st.columns([2, 1])
            with col_f:
                feat_select = st.selectbox("Select feature", num_c_viz, key="viz8_feat")
            with col_t:
                kind_select = st.radio("Plot type", ["Box Plot", "Histogram"], key="viz8_kind", horizontal=True)

            kind = "box" if kind_select == "Box Plot" else "hist"
            df_base["Cluster"] = labels
            st.plotly_chart(fig_feature_dist(df_base, feat_select, labels, kind=kind), use_container_width=True)

            # Feature heatmap per cluster
            section("🌡️", "Feature Averages Heatmap by Cluster")
            df_base["Cluster"] = labels
            cluster_means = df_base.groupby("Cluster")[num_c_viz[:10]].mean()
            fig_heat = px.imshow(
                cluster_means,
                title="Feature Means per Cluster",
                color_continuous_scale="RdBu_r",
                template="plotly_dark",
                aspect="auto",
                text_auto=".2f"
            )
            fig_heat.update_layout(**PLOTLY_LAYOUT)
            st.plotly_chart(fig_heat, use_container_width=True)

        # Download
        section("💾", "Download Labeled Dataset")
        df_export = df_raw.copy()
        df_export["Cluster_Label"] = labels
        st.download_button(
            label="⬇️ Download Dataset with Cluster Labels",
            data=download_csv(df_export),
            file_name=f"clustermind_{model.lower()}_labeled.csv",
            mime="text/csv",
            key="m8_download"
        )

        success_box("🎉 Congratulations! You've completed the full ML clustering pipeline — from raw data to insights!")
        explain("🌍 What to do next",
                "• Use your cluster labels as a feature for supervised learning<br>"
                "• Share your cluster analysis in a report or presentation<br>"
                "• Try different algorithms and K values — compare silhouette scores<br>"
                "• Apply domain knowledge to name your clusters (e.g., 'High Value', 'At Risk')<br>"
                "• Collect more data or engineer new features for better separation")

    # ── EXPLAIN BUTTON (MANUAL MODE) ─────────────────────────────
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    if st.button("🧠 Explain What Happened", key="explain_manual_btn"):
        with st.spinner("Generating stage-aware explanation..."):
            explanation = generate_manual_explanation(
                stage=stage,
                df_raw=df_raw,
                df_cleaned=st.session_state.df_cleaned,
                df_engineered=st.session_state.df_engineered,
                cleaning_action=st.session_state.get("cleaning_action", "Not yet done"),
                engineered_features=st.session_state.get("engineered_features", []),
                scale_applied=st.session_state.scale_applied,
                encode_applied=st.session_state.encode_applied,
                chosen_model=st.session_state.chosen_model,
                chosen_k=st.session_state.chosen_k,
                chosen_eps=st.session_state.chosen_eps,
                chosen_min_samples=st.session_state.chosen_min_samples,
                score=st.session_state.manual_score,
                n_clusters=st.session_state.manual_n_clusters,
                numeric_cols=st.session_state.numeric_cols,
                categorical_cols=st.session_state.categorical_cols,
            )
        with st.expander(f"📖 Pipeline Explanation — Stages 1 to {stage} (click to expand)", expanded=True):
            st.markdown(explanation)
            tip("Use these explanations to build your understanding. By Stage 8, you'll know the entire clustering pipeline!")

# =========================================
# ENHANCED ML CLUSTERING PLATFORM
# =========================================
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
import io
import time

from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering, MeanShift, Birch
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.pipeline import Pipeline
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# =========================================
# PAGE CONFIG & DARK THEME
# =========================================
st.set_page_config(
    page_title="ClusterForge · ML Platform",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================
# CUSTOM CSS — DARK INDUSTRIAL THEME
# =========================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --bg-primary: #0a0a0f;
    --bg-secondary: #111118;
    --bg-card: #16161f;
    --bg-hover: #1e1e2a;
    --accent-cyan: #00e5ff;
    --accent-violet: #8b5cf6;
    --accent-green: #10b981;
    --accent-amber: #f59e0b;
    --accent-rose: #f43f5e;
    --text-primary: #f0f0ff;
    --text-secondary: #8888aa;
    --text-muted: #44445a;
    --border: #2a2a3a;
    --border-glow: rgba(0,229,255,0.3);
    --font-mono: 'Space Mono', monospace;
    --font-body: 'DM Sans', sans-serif;
}

html, body, [class*="css"] {
    font-family: var(--font-body) !important;
    background-color: var(--bg-primary) !important;
    color: var(--text-primary) !important;
}

/* Main container */
.main .block-container {
    background: var(--bg-primary);
    padding: 2rem 2.5rem;
    max-width: 1400px;
}

/* Header */
.hero-header {
    display: flex;
    align-items: center;
    gap: 1rem;
    padding: 2rem 0 1.5rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 2rem;
}
.hero-title {
    font-family: var(--font-mono);
    font-size: 2.2rem;
    font-weight: 700;
    color: var(--accent-cyan);
    letter-spacing: -0.03em;
    margin: 0;
    line-height: 1;
}
.hero-subtitle {
    font-size: 0.85rem;
    color: var(--text-secondary);
    font-family: var(--font-mono);
    margin-top: 0.3rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}
.badge {
    background: rgba(0,229,255,0.08);
    border: 1px solid rgba(0,229,255,0.2);
    color: var(--accent-cyan);
    font-family: var(--font-mono);
    font-size: 0.65rem;
    padding: 0.2rem 0.6rem;
    border-radius: 2px;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    display: inline-block;
    margin-left: 0.5rem;
}

/* Metric cards */
.metric-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 1.2rem 1.5rem;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s;
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, var(--accent-cyan), var(--accent-violet));
}
.metric-card:hover { border-color: var(--border-glow); }
.metric-label {
    font-size: 0.7rem;
    color: var(--text-secondary);
    font-family: var(--font-mono);
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 0.4rem;
}
.metric-value {
    font-size: 1.8rem;
    font-family: var(--font-mono);
    font-weight: 700;
    color: var(--text-primary);
    line-height: 1;
}
.metric-delta {
    font-size: 0.75rem;
    color: var(--accent-green);
    margin-top: 0.3rem;
    font-family: var(--font-mono);
}

/* Section headers */
.section-header {
    font-family: var(--font-mono);
    font-size: 0.7rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--text-secondary);
    padding: 0.4rem 0;
    border-bottom: 1px solid var(--border);
    margin: 2rem 0 1.2rem;
    display: flex;
    align-items: center;
    gap: 0.6rem;
}
.section-header::before {
    content: '';
    width: 4px; height: 4px;
    border-radius: 50%;
    background: var(--accent-cyan);
    display: inline-block;
}

/* Score cards row */
.score-row {
    display: flex;
    gap: 1rem;
    margin: 1.5rem 0;
    flex-wrap: wrap;
}
.score-card {
    flex: 1;
    min-width: 140px;
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 1rem 1.2rem;
    text-align: center;
}
.score-card .label {
    font-family: var(--font-mono);
    font-size: 0.6rem;
    color: var(--text-secondary);
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}
.score-card .value {
    font-family: var(--font-mono);
    font-size: 1.5rem;
    font-weight: 700;
}
.score-cyan  { color: var(--accent-cyan); }
.score-violet { color: var(--accent-violet); }
.score-green  { color: var(--accent-green); }
.score-amber  { color: var(--accent-amber); }

/* Alert boxes */
.alert-info {
    background: rgba(0,229,255,0.05);
    border: 1px solid rgba(0,229,255,0.2);
    border-left: 3px solid var(--accent-cyan);
    border-radius: 4px;
    padding: 0.8rem 1rem;
    font-size: 0.85rem;
    color: var(--text-secondary);
    margin: 1rem 0;
}
.alert-warn {
    background: rgba(245,158,11,0.05);
    border: 1px solid rgba(245,158,11,0.2);
    border-left: 3px solid var(--accent-amber);
    border-radius: 4px;
    padding: 0.8rem 1rem;
    font-size: 0.85rem;
    color: var(--text-secondary);
    margin: 1rem 0;
}
.alert-success {
    background: rgba(16,185,129,0.05);
    border: 1px solid rgba(16,185,129,0.2);
    border-left: 3px solid var(--accent-green);
    border-radius: 4px;
    padding: 0.8rem 1rem;
    font-size: 0.85rem;
    color: var(--text-secondary);
    margin: 1rem 0;
}

/* Streamlit overrides */
.stButton > button {
    background: var(--accent-cyan) !important;
    color: #0a0a0f !important;
    font-family: var(--font-mono) !important;
    font-weight: 700 !important;
    font-size: 0.8rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    border: none !important;
    border-radius: 4px !important;
    padding: 0.6rem 2rem !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: #00ccee !important;
    transform: translateY(-1px);
    box-shadow: 0 4px 20px rgba(0,229,255,0.3) !important;
}
.stButton > button[kind="secondary"] {
    background: var(--bg-card) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border) !important;
}

.stSelectbox > div > div,
.stMultiselect > div > div,
.stTextInput > div > div input,
.stSlider {
    background: var(--bg-card) !important;
    border-color: var(--border) !important;
    color: var(--text-primary) !important;
}

.stDataFrame {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
}

.stTabs [data-baseweb="tab-list"] {
    background: var(--bg-secondary) !important;
    border-bottom: 1px solid var(--border) !important;
    gap: 0 !important;
}
.stTabs [data-baseweb="tab"] {
    font-family: var(--font-mono) !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    color: var(--text-secondary) !important;
    padding: 0.8rem 1.5rem !important;
    border-bottom: 2px solid transparent !important;
}
.stTabs [aria-selected="true"] {
    color: var(--accent-cyan) !important;
    border-bottom-color: var(--accent-cyan) !important;
    background: transparent !important;
}

.stSidebar {
    background: var(--bg-secondary) !important;
    border-right: 1px solid var(--border) !important;
}
.stSidebar .stRadio > label,
.stSidebar label {
    font-family: var(--font-mono) !important;
    font-size: 0.75rem !important;
    color: var(--text-secondary) !important;
    letter-spacing: 0.05em !important;
}
.stSidebar .stRadio [data-testid="stMarkdownContainer"] p {
    font-size: 0.8rem !important;
}

.stProgress > div > div {
    background: linear-gradient(90deg, var(--accent-cyan), var(--accent-violet)) !important;
}

div[data-testid="stMetricValue"] {
    font-family: var(--font-mono) !important;
    font-size: 1.6rem !important;
    color: var(--text-primary) !important;
}
div[data-testid="stMetricLabel"] {
    font-family: var(--font-mono) !important;
    font-size: 0.65rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    color: var(--text-secondary) !important;
}
div[data-testid="stMetricDelta"] {
    font-family: var(--font-mono) !important;
    font-size: 0.75rem !important;
}

hr { border-color: var(--border) !important; }

.stExpander {
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
    background: var(--bg-card) !important;
}
.stExpander > div > div > div > div > p {
    font-family: var(--font-mono) !important;
    font-size: 0.75rem !important;
    color: var(--text-secondary) !important;
}

/* Upload zone */
[data-testid="stFileUploadDropzone"] {
    background: var(--bg-card) !important;
    border: 1px dashed var(--border) !important;
    border-radius: 8px !important;
}
[data-testid="stFileUploadDropzone"]:hover {
    border-color: var(--accent-cyan) !important;
    background: rgba(0,229,255,0.03) !important;
}
</style>
""", unsafe_allow_html=True)

# =========================================
# HEADER
# =========================================
st.markdown("""
<div class="hero-header">
  <div>
    <div class="hero-title">⬡ ClusterForge</div>
    <div class="hero-subtitle">ML Clustering Platform · AutoML + Manual Mode
      <span class="badge">v2.0</span>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# =========================================
# PLOTLY DARK THEME DEFAULTS
# =========================================
PLOTLY_THEME = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(22,22,31,0)",
    plot_bgcolor="rgba(22,22,31,0)",
    font=dict(family="Space Mono, monospace", color="#8888aa", size=11),
    colorway=["#00e5ff","#8b5cf6","#10b981","#f59e0b","#f43f5e","#06b6d4","#a78bfa","#34d399"],
)
ACCENT_COLORS = ["#00e5ff","#8b5cf6","#10b981","#f59e0b","#f43f5e","#06b6d4","#a78bfa","#34d399"]

# =========================================
# SESSION STATE
# =========================================
for key, val in {
    "labels": None,
    "model": None,
    "X_processed": None,
    "df_result": None,
    "automl_done": False,
    "manual_done": False,
}.items():
    if key not in st.session_state:
        st.session_state[key] = val

# =========================================
# SIDEBAR
# =========================================
with st.sidebar:
    st.markdown("""
    <div style="font-family:Space Mono;font-size:0.6rem;letter-spacing:0.15em;
    text-transform:uppercase;color:#444460;padding:0.5rem 0 1rem;">
    ▸ Control Panel
    </div>
    """, unsafe_allow_html=True)

    mode = st.radio("Mode", ["⚡ AutoML", "🎓 Manual Mode", "📊 Data Explorer"], index=0)

    st.divider()
    st.markdown("""<div style="font-family:Space Mono;font-size:0.6rem;letter-spacing:0.15em;
    text-transform:uppercase;color:#444460;margin-bottom:0.5rem;">▸ Dataset</div>""", unsafe_allow_html=True)
    file = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")

    st.divider()
    st.markdown("""<div style="font-family:Space Mono;font-size:0.6rem;letter-spacing:0.15em;
    text-transform:uppercase;color:#444460;margin-bottom:0.5rem;">▸ Preprocessing</div>""", unsafe_allow_html=True)

    scaler_choice = st.selectbox("Scaler", ["StandardScaler", "MinMaxScaler", "RobustScaler"])
    reduction = st.selectbox("Dim Reduction (Viz)", ["PCA", "t-SNE"])

    st.divider()
    st.caption("ClusterForge · Built with Streamlit + scikit-learn")

# =========================================
# NO FILE
# =========================================
if not file:
    c1, c2, c3 = st.columns([1,2,1])
    with c2:
        st.markdown("""
        <div style="text-align:center;padding:4rem 2rem;background:var(--bg-card);
        border:1px dashed #2a2a3a;border-radius:12px;margin-top:3rem;">
          <div style="font-size:3rem;margin-bottom:1rem;opacity:0.4">⬡</div>
          <div style="font-family:Space Mono;font-size:0.9rem;color:#8888aa;margin-bottom:0.5rem;">
          Upload a CSV to begin
          </div>
          <div style="font-size:0.75rem;color:#44445a;">
          Supports numerical and categorical features · Auto-cleans missing values
          </div>
        </div>
        """, unsafe_allow_html=True)
    st.stop()

# =========================================
# LOAD DATA
# =========================================
@st.cache_data
def load_data(f):
    return pd.read_csv(f)

df = load_data(file)
if df.empty:
    st.error("Empty dataset — please upload a valid CSV.")
    st.stop()

# =========================================
# CLEANING
# =========================================
def clean_data(df):
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].fillna("Missing")
        else:
            df[col] = df[col].fillna(df[col].median())
    return df

# =========================================
# PREPROCESS
# =========================================
@st.cache_data
def preprocess(df, scaler_name="StandardScaler"):
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()

    scaler_map = {
        "StandardScaler": StandardScaler(),
        "MinMaxScaler": MinMaxScaler(),
        "RobustScaler": RobustScaler(),
    }
    scaler = scaler_map[scaler_name]

    steps = []
    if num_cols:
        steps.append(("num", scaler, num_cols))
    if cat_cols:
        steps.append(("cat", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), cat_cols))

    if not steps:
        return np.zeros((len(df), 1))

    transformer = ColumnTransformer(steps)
    return transformer.fit_transform(df)

# =========================================
# DIMENSIONALITY REDUCTION
# =========================================
@st.cache_data
def reduce_dims(X, method="PCA"):
    if method == "PCA":
        return PCA(n_components=2, random_state=42).fit_transform(X)
    else:
        n = min(30, X.shape[0]-1)
        return TSNE(n_components=2, perplexity=n, random_state=42).fit_transform(X)

# =========================================
# METRICS
# =========================================
def compute_metrics(X, labels):
    metrics = {}
    unique = set(labels)
    valid = [l for l in unique if l != -1]
    if len(valid) < 2:
        return metrics
    try:
        metrics["Silhouette"] = round(silhouette_score(X, labels), 4)
    except: pass
    try:
        metrics["Davies-Bouldin"] = round(davies_bouldin_score(X, labels), 4)
    except: pass
    try:
        metrics["Calinski-Harabasz"] = round(calinski_harabasz_score(X, labels), 1)
    except: pass
    return metrics

# =========================================
# AUTOML ENGINE
# =========================================
def run_automl(X, progress_cb=None):
    results = []
    total = 14
    done = 0

    def update(label):
        nonlocal done
        done += 1
        if progress_cb:
            progress_cb(done / total, label)

    # KMeans 2–9
    for k in range(2, 10):
        try:
            m = KMeans(n_clusters=k, random_state=42, n_init="auto")
            lbl = m.fit_predict(X)
            sil = silhouette_score(X, lbl) if len(set(lbl)) >= 2 else -1
            results.append(("KMeans", f"k={k}", m, lbl, sil))
        except: pass
        update(f"KMeans k={k}")

    # Agglomerative 2–4
    for k in range(2, 5):
        try:
            m = AgglomerativeClustering(n_clusters=k)
            lbl = m.fit_predict(X)
            sil = silhouette_score(X, lbl) if len(set(lbl)) >= 2 else -1
            results.append(("Agglomerative", f"k={k}", m, lbl, sil))
        except: pass
        update(f"Agglomerative k={k}")

    # DBSCAN
    for eps in [0.3, 0.5, 0.7]:
        try:
            m = DBSCAN(eps=eps)
            lbl = m.fit_predict(X)
            valid = [l for l in lbl if l != -1]
            if len(set(valid)) >= 2:
                sil = silhouette_score(X[np.array(lbl) != -1], np.array(lbl)[np.array(lbl) != -1])
                results.append(("DBSCAN", f"eps={eps}", m, lbl, sil))
        except: pass
        update(f"DBSCAN eps={eps}")

    if not results:
        return None, []

    results.sort(key=lambda x: x[4], reverse=True)
    best = results[0]
    return best, results

# =========================================
# CLUSTER PLOTS
# =========================================
def scatter_plot(X2d, labels, title="Cluster Visualization"):
    df_plot = pd.DataFrame({"x": X2d[:,0], "y": X2d[:,1], "Cluster": [str(l) for l in labels]})
    fig = px.scatter(
        df_plot, x="x", y="y", color="Cluster",
        title=title,
        color_discrete_sequence=ACCENT_COLORS,
    )
    fig.update_traces(marker=dict(size=5, opacity=0.75, line=dict(width=0)))
    fig.update_layout(
        **PLOTLY_THEME,
        title=dict(font=dict(family="Space Mono", size=13, color="#8888aa")),
        legend=dict(
            font=dict(family="Space Mono", size=10),
            bgcolor="rgba(22,22,31,0.8)",
            bordercolor="#2a2a3a",
            borderwidth=1,
        ),
        xaxis=dict(showgrid=True, gridcolor="#1e1e2a", zeroline=False, title=""),
        yaxis=dict(showgrid=True, gridcolor="#1e1e2a", zeroline=False, title=""),
        height=440,
    )
    return fig

def dist_chart(labels, title="Cluster Distribution"):
    cnt = pd.Series(labels).value_counts().sort_index().reset_index()
    cnt.columns = ["Cluster", "Count"]
    cnt["Cluster"] = cnt["Cluster"].astype(str)
    fig = px.bar(
        cnt, x="Cluster", y="Count",
        color="Cluster",
        color_discrete_sequence=ACCENT_COLORS,
        title=title,
    )
    fig.update_layout(
        **PLOTLY_THEME,
        showlegend=False,
        title=dict(font=dict(family="Space Mono", size=13, color="#8888aa")),
        xaxis=dict(showgrid=False, title=""),
        yaxis=dict(showgrid=True, gridcolor="#1e1e2a", title="Count"),
        height=300,
        bargap=0.35,
    )
    fig.update_traces(marker_line_width=0)
    return fig

def elbow_chart(X, max_k=10):
    inertias, ks = [], range(2, max_k+1)
    for k in ks:
        m = KMeans(n_clusters=k, random_state=42, n_init="auto")
        m.fit(X)
        inertias.append(m.inertia_)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(ks), y=inertias,
        mode="lines+markers",
        line=dict(color="#00e5ff", width=2),
        marker=dict(size=7, color="#00e5ff", symbol="circle"),
        name="Inertia",
    ))
    fig.update_layout(
        **PLOTLY_THEME,
        title=dict(text="Elbow Curve — KMeans Inertia", font=dict(family="Space Mono", size=13, color="#8888aa")),
        xaxis=dict(showgrid=True, gridcolor="#1e1e2a", title="k", dtick=1),
        yaxis=dict(showgrid=True, gridcolor="#1e1e2a", title="Inertia"),
        height=300,
    )
    return fig

def silhouette_sweep_chart(X, max_k=10):
    scores, ks = [], range(2, max_k+1)
    for k in ks:
        try:
            lbl = KMeans(n_clusters=k, random_state=42, n_init="auto").fit_predict(X)
            scores.append(silhouette_score(X, lbl))
        except:
            scores.append(0)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(ks), y=scores,
        mode="lines+markers",
        line=dict(color="#8b5cf6", width=2),
        marker=dict(size=7, color="#8b5cf6"),
        fill="tozeroy",
        fillcolor="rgba(139,92,246,0.08)",
        name="Silhouette",
    ))
    fig.update_layout(
        **PLOTLY_THEME,
        title=dict(text="Silhouette Score by k", font=dict(family="Space Mono", size=13, color="#8888aa")),
        xaxis=dict(showgrid=True, gridcolor="#1e1e2a", title="k", dtick=1),
        yaxis=dict(showgrid=True, gridcolor="#1e1e2a", title="Score"),
        height=300,
    )
    return fig

def feature_importance_chart(df_clustered):
    num_cols = df_clustered.select_dtypes(include=np.number).columns.tolist()
    num_cols = [c for c in num_cols if c != "Cluster"]
    if not num_cols:
        return None
    summary = df_clustered.groupby("Cluster")[num_cols].mean()
    diff = (summary.max() - summary.min()).sort_values(ascending=True).tail(10)
    fig = go.Figure(go.Bar(
        x=diff.values,
        y=diff.index,
        orientation="h",
        marker=dict(
            color=diff.values,
            colorscale=[[0, "#1e1e2a"], [0.5, "#8b5cf6"], [1, "#00e5ff"]],
            showscale=False,
        ),
    ))
    fig.update_layout(
        **PLOTLY_THEME,
        title=dict(text="Top Differentiating Features", font=dict(family="Space Mono", size=13, color="#8888aa")),
        xaxis=dict(showgrid=True, gridcolor="#1e1e2a", title="Mean Range Across Clusters"),
        yaxis=dict(showgrid=False, title=""),
        height=350,
        margin=dict(l=20, r=20),
    )
    return fig

def cluster_heatmap(df_clustered):
    num_cols = df_clustered.select_dtypes(include=np.number).columns.tolist()
    num_cols = [c for c in num_cols if c != "Cluster"]
    if not num_cols:
        return None
    summary = df_clustered.groupby("Cluster")[num_cols].mean()
    norm = (summary - summary.min()) / (summary.max() - summary.min() + 1e-9)
    fig = px.imshow(
        norm.T,
        color_continuous_scale=[[0,"#0a0a0f"],[0.3,"#1e1e40"],[0.7,"#8b5cf6"],[1,"#00e5ff"]],
        aspect="auto",
        title="Cluster Feature Heatmap (Normalized)",
    )
    fig.update_layout(
        **PLOTLY_THEME,
        title=dict(font=dict(family="Space Mono", size=13, color="#8888aa")),
        coloraxis_showscale=True,
        height=max(300, 30 * len(num_cols)),
        xaxis=dict(title="Cluster"),
        yaxis=dict(title="Feature"),
    )
    return fig

def pairplot_top_features(df_clustered, n=4):
    num_cols = df_clustered.select_dtypes(include=np.number).columns.tolist()
    num_cols = [c for c in num_cols if c != "Cluster"]
    if len(num_cols) < 2:
        return None
    summary = df_clustered.groupby("Cluster")[num_cols].mean()
    diff = (summary.max() - summary.min()).sort_values(ascending=False)
    top = diff.head(n).index.tolist()
    fig = px.scatter_matrix(
        df_clustered,
        dimensions=top,
        color=df_clustered["Cluster"].astype(str),
        color_discrete_sequence=ACCENT_COLORS,
        title=f"Scatter Matrix — Top {len(top)} Features",
    )
    fig.update_traces(marker=dict(size=3, opacity=0.5))
    fig.update_layout(
        **PLOTLY_THEME,
        title=dict(font=dict(family="Space Mono", size=13, color="#8888aa")),
        height=550,
    )
    return fig

# =========================================
# RESULTS DISPLAY
# =========================================
def show_results(df_clean, X, labels, model_name="", metrics=None):
    df_r = df_clean.copy()
    df_r["Cluster"] = labels
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    noise = (np.array(labels) == -1).sum()

    X2d = reduce_dims(X, reduction)

    # Score row
    if metrics:
        cols = st.columns(len(metrics) + 2)
        cols[0].metric("Model", model_name)
        cols[1].metric("Clusters", n_clusters)
        for i, (k, v) in enumerate(metrics.items()):
            cols[i+2].metric(k, v)
        if noise > 0:
            st.markdown(f'<div class="alert-warn">⚠ {noise} noise points (label -1) detected — DBSCAN outliers excluded from scoring.</div>', unsafe_allow_html=True)
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Scatter", "Distribution", "Features", "Heatmap", "Pair Matrix"])

    with tab1:
        col_a, col_b = st.columns([2,1])
        with col_a:
            st.plotly_chart(scatter_plot(X2d, labels, f"{model_name} · {reduction} Projection"), use_container_width=True)
        with col_b:
            st.plotly_chart(dist_chart(labels), use_container_width=True)

    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(elbow_chart(X), use_container_width=True)
        with c2:
            st.plotly_chart(silhouette_sweep_chart(X), use_container_width=True)

    with tab3:
        fig_imp = feature_importance_chart(df_r)
        if fig_imp:
            st.plotly_chart(fig_imp, use_container_width=True)
        else:
            st.info("No numeric features to analyse.")

    with tab4:
        fig_heat = cluster_heatmap(df_r)
        if fig_heat:
            st.plotly_chart(fig_heat, use_container_width=True)

    with tab5:
        fig_pair = pairplot_top_features(df_r)
        if fig_pair:
            st.plotly_chart(fig_pair, use_container_width=True)
        else:
            st.info("Need at least 2 numeric features.")

    # Cluster summary table
    st.markdown('<div class="section-header">Cluster Summary</div>', unsafe_allow_html=True)
    num_cols = df_r.select_dtypes(include=np.number).columns.tolist()
    num_cols = [c for c in num_cols if c != "Cluster"]
    if num_cols:
        summary = df_r.groupby("Cluster")[num_cols].agg(["mean","std","count"]).round(3)
        st.dataframe(summary, use_container_width=True)

    # Downloads
    st.markdown('<div class="section-header">Export</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.download_button(
            "⬇ Download Clustered CSV",
            df_r.to_csv(index=False).encode(),
            "clustered_data.csv",
            "text/csv",
        )
    with c2:
        if hasattr(st.session_state.model, "fit"):
            st.download_button(
                "💾 Download Model (.pkl)",
                pickle.dumps(st.session_state.model),
                "model.pkl",
                "application/octet-stream",
            )
    with c3:
        summary_text = df_r.groupby("Cluster").describe().to_csv()
        st.download_button(
            "📄 Download Summary CSV",
            summary_text.encode(),
            "cluster_summary.csv",
            "text/csv",
        )

    return df_r

# =========================================
# DATA EXPLORER MODE
# =========================================
if "Explorer" in mode:
    st.markdown('<div class="section-header">Data Explorer</div>', unsafe_allow_html=True)

    search = st.text_input("🔍 Filter rows (search across all columns)", "")
    col_sel = st.multiselect("Columns to display", df.columns.tolist(), default=df.columns.tolist())

    view = df[col_sel] if col_sel else df
    if search:
        view = view[view.astype(str).apply(lambda x: x.str.contains(search, case=False)).any(axis=1)]

    st.dataframe(view, use_container_width=True, height=400)

    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Rows", df.shape[0])
    r2.metric("Columns", df.shape[1])
    r3.metric("Missing Values", int(df.isnull().sum().sum()))
    r4.metric("Numeric Features", int(df.select_dtypes(include=np.number).shape[1]))

    st.markdown('<div class="section-header">Distributions</div>', unsafe_allow_html=True)
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    if num_cols:
        feat = st.selectbox("Feature to visualize", num_cols)
        fig = px.histogram(
            df, x=feat, nbins=40,
            color_discrete_sequence=["#00e5ff"],
            title=f"Distribution of {feat}",
        )
        fig.update_layout(**PLOTLY_THEME, height=300,
            xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor="#1e1e2a"))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-header">Correlation Matrix</div>', unsafe_allow_html=True)
    if len(num_cols) >= 2:
        corr = df[num_cols].corr()
        fig_corr = px.imshow(
            corr,
            color_continuous_scale=[[0,"#f43f5e"],[0.5,"#0a0a0f"],[1,"#00e5ff"]],
            aspect="auto",
            title="Feature Correlation",
        )
        fig_corr.update_layout(**PLOTLY_THEME, height=450,
            title=dict(font=dict(family="Space Mono", size=13, color="#8888aa")))
        st.plotly_chart(fig_corr, use_container_width=True)

# =========================================
# AUTOML MODE
# =========================================
elif "AutoML" in mode:
    st.markdown('<div class="section-header">AutoML · Automated Model Selection</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="alert-info">
    AutoML evaluates <b>KMeans</b> (k=2–9), <b>Agglomerative</b> (k=2–4), and <b>DBSCAN</b> (ε=0.3/0.5/0.7)
    — selecting the configuration with the highest Silhouette Score.
    </div>
    """, unsafe_allow_html=True)

    if st.button("⚡ Run AutoML", type="primary"):
        df_clean = clean_data(df)
        X = preprocess(df_clean, scaler_choice)

        progress_bar = st.progress(0)
        status = st.empty()

        def progress_cb(val, label):
            progress_bar.progress(val)
            status.markdown(f"""
            <div style="font-family:Space Mono;font-size:0.7rem;color:#8888aa;margin:0.3rem 0;">
            ▸ Testing {label}…
            </div>""", unsafe_allow_html=True)

        with st.spinner(""):
            best, all_results = run_automl(X, progress_cb)

        progress_bar.empty()
        status.empty()

        if best:
            name, cfg, model, labels, score = best
            metrics = compute_metrics(X, labels)
            st.session_state.model = model
            st.session_state.labels = labels
            st.session_state.X_processed = X
            st.session_state.automl_done = True

            st.markdown(f"""
            <div class="alert-success">
            ✓ Best model: <b>{name} ({cfg})</b> · Silhouette Score: <b>{round(score,4)}</b>
            </div>
            """, unsafe_allow_html=True)

            # Comparison table
            with st.expander("View all tested configurations", expanded=False):
                rows = []
                for r in all_results[:10]:
                    m_name, m_cfg, _, lbl, sil = r
                    db = davies_bouldin_score(X, lbl) if len(set(lbl)) >= 2 else None
                    rows.append({
                        "Model": m_name, "Config": m_cfg,
                        "Silhouette ↑": round(sil,4),
                        "Davies-Bouldin ↓": round(db,4) if db else "N/A",
                        "Clusters": len(set(lbl)) - (1 if -1 in lbl else 0),
                    })
                comp_df = pd.DataFrame(rows)
                st.dataframe(comp_df, use_container_width=True, hide_index=True)

            show_results(df_clean, X, labels, f"{name} {cfg}", metrics)

        else:
            st.error("No valid clustering found. Try a different dataset or adjust preprocessing.")

    elif st.session_state.automl_done and st.session_state.labels is not None:
        df_clean = clean_data(df)
        X = st.session_state.X_processed
        labels = st.session_state.labels
        metrics = compute_metrics(X, labels)
        show_results(df_clean, X, labels, "AutoML Best", metrics)

# =========================================
# MANUAL MODE
# =========================================
else:
    st.markdown('<div class="section-header">Manual Mode · Custom Configuration</div>', unsafe_allow_html=True)

    col_model, col_params = st.columns([1, 2])

    with col_model:
        model_type = st.selectbox("Algorithm", [
            "KMeans",
            "Agglomerative Clustering",
            "DBSCAN",
            "Spectral Clustering",
            "Birch",
            "MeanShift",
        ])

    with col_params:
        params = {}
        if model_type == "KMeans":
            c1, c2 = st.columns(2)
            params["n_clusters"] = c1.slider("Clusters (k)", 2, 15, 3)
            params["max_iter"] = c2.slider("Max Iterations", 100, 1000, 300, step=50)

        elif model_type == "Agglomerative Clustering":
            c1, c2 = st.columns(2)
            params["n_clusters"] = c1.slider("Clusters", 2, 15, 3)
            params["linkage"] = c2.selectbox("Linkage", ["ward","complete","average","single"])

        elif model_type == "DBSCAN":
            c1, c2 = st.columns(2)
            params["eps"] = c1.slider("ε (epsilon)", 0.05, 2.0, 0.5, step=0.05)
            params["min_samples"] = c2.slider("Min Samples", 2, 20, 5)

        elif model_type == "Spectral Clustering":
            c1, c2 = st.columns(2)
            params["n_clusters"] = c1.slider("Clusters", 2, 10, 3)
            params["affinity"] = c2.selectbox("Affinity", ["rbf","nearest_neighbors"])

        elif model_type == "Birch":
            c1, c2, c3 = st.columns(3)
            params["n_clusters"] = c1.slider("Clusters", 2, 15, 3)
            params["threshold"] = c2.slider("Threshold", 0.1, 1.0, 0.5, step=0.05)
            params["branching_factor"] = c3.slider("Branch Factor", 10, 100, 50, step=10)

        elif model_type == "MeanShift":
            params["bandwidth"] = st.slider("Bandwidth (0=auto)", 0.0, 5.0, 0.0, step=0.1)

    run_col, _ = st.columns([1, 4])
    with run_col:
        run_btn = st.button("▶ Train Model", type="primary")

    if run_btn:
        df_clean = clean_data(df)
        X = preprocess(df_clean, scaler_choice)
        st.session_state.X_processed = X

        try:
            with st.spinner("Training…"):
                if model_type == "KMeans":
                    model = KMeans(n_clusters=params["n_clusters"], max_iter=params["max_iter"], random_state=42, n_init="auto")
                elif model_type == "Agglomerative Clustering":
                    model = AgglomerativeClustering(n_clusters=params["n_clusters"], linkage=params["linkage"])
                elif model_type == "DBSCAN":
                    model = DBSCAN(eps=params["eps"], min_samples=params["min_samples"])
                elif model_type == "Spectral Clustering":
                    model = SpectralClustering(n_clusters=params["n_clusters"], affinity=params["affinity"], random_state=42)
                elif model_type == "Birch":
                    model = Birch(n_clusters=params["n_clusters"], threshold=params["threshold"], branching_factor=params["branching_factor"])
                elif model_type == "MeanShift":
                    bw = params["bandwidth"] if params["bandwidth"] > 0 else None
                    model = MeanShift(bandwidth=bw)

                labels = model.fit_predict(X)
                st.session_state.labels = labels
                st.session_state.model = model
                st.session_state.manual_done = True

        except Exception as e:
            st.error(f"Training failed: {e}")

    if st.session_state.manual_done and st.session_state.labels is not None:
        df_clean = clean_data(df)
        X = st.session_state.X_processed
        labels = st.session_state.labels
        metrics = compute_metrics(X, labels)
        short_name = model_type.split()[0]
        show_results(df_clean, X, labels, short_name, metrics)

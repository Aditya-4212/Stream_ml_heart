# =========================================
# IMPORTS
# =========================================
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import pickle

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score

# =========================================
# CONFIG
# =========================================
st.set_page_config(page_title="ML Clustering Platform", layout="wide")
st.title("🔥 ML Clustering Platform (AutoML + Manual Mode)")

# =========================================
# SESSION STATE
# =========================================
if "labels" not in st.session_state:
    st.session_state.labels = None

# =========================================
# SIDEBAR
# =========================================
mode = st.sidebar.radio("Mode", ["AutoML", "Manual Mode"])
file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

# =========================================
# LOAD DATA
# =========================================
if not file:
    st.info("Upload dataset to begin")
    st.stop()

@st.cache_data
def load_data(file):
    return pd.read_csv(file)

df = load_data(file)

if df.empty:
    st.error("Empty dataset")
    st.stop()

# =========================================
# DATA EXPLORER
# =========================================
st.subheader("📊 Data Explorer")

search = st.text_input("Search Data")
cols = st.multiselect("Select Columns", df.columns, default=df.columns)

data = df[cols]

if search:
    data = data[data.astype(str).apply(lambda x: x.str.contains(search, case=False)).any(axis=1)]

st.dataframe(data, width="stretch")

col1, col2, col3 = st.columns(3)
col1.metric("Rows", df.shape[0])
col2.metric("Columns", df.shape[1])
col3.metric("Missing Values", int(df.isnull().sum().sum()))

# =========================================
# CLEANING
# =========================================
def clean_data(df):
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].fillna("Missing")
        else:
            df[col] = df[col].fillna(df[col].mean())
    return df

# =========================================
# PREPROCESS
# =========================================
@st.cache_data
def preprocess(df):
    num = df.select_dtypes(include=np.number).columns
    cat = df.select_dtypes(exclude=np.number).columns

    transformer = ColumnTransformer([
        ("num", StandardScaler(), num),
        ("cat", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), cat)
    ])

    return transformer.fit_transform(df)

# =========================================
# SAFE SCORE
# =========================================
def safe_score(X, labels):
    try:
        if len(set(labels)) < 2:
            return None
        return silhouette_score(X, labels)
    except:
        return None

# =========================================
# AUTOML
# =========================================
def run_automl(X):
    best_score = -1
    best_result = None

    # KMeans
    for k in range(2, 9):
        try:
            model = KMeans(n_clusters=k, random_state=42)
            labels = model.fit_predict(X)
            score = safe_score(X, labels)

            if score and score > best_score:
                best_score = score
                best_result = ("KMeans", model, labels, score)
        except:
            continue

    # Agglomerative
    try:
        model = AgglomerativeClustering(n_clusters=3)
        labels = model.fit_predict(X)
        score = safe_score(X, labels)

        if score and score > best_score:
            best_result = ("Agglomerative", model, labels, score)
    except:
        pass

    # DBSCAN
    for eps in [0.3, 0.5, 0.7]:
        try:
            model = DBSCAN(eps=eps)
            labels = model.fit_predict(X)
            score = safe_score(X, labels)

            if score and score > best_score:
                best_result = ("DBSCAN", model, labels, score)
        except:
            continue

    return best_result

# =========================================
# INSIGHTS
# =========================================
def get_insights(df):
    summary = df.groupby("Cluster").mean(numeric_only=True)
    diff = (summary.max() - summary.min()).sort_values(ascending=False)
    return summary, diff.head(5)

# =========================================
# AUTOML MODE
# =========================================
if mode == "AutoML":

    st.header("⚡ AutoML Clustering")

    df_clean = clean_data(df)
    X = preprocess(df_clean)

    result = run_automl(X)

    if result:
        name, model, labels, score = result
        df_clean["Cluster"] = labels

        c1, c2, c3 = st.columns(3)
        c1.metric("Model", name)
        c2.metric("Clusters", len(set(labels)))
        c3.metric("Silhouette Score", round(score, 3))

        # Cluster distribution
        st.plotly_chart(px.bar(df_clean["Cluster"].value_counts(), title="Cluster Distribution"))

        # PCA
        X_pca = PCA(2).fit_transform(X)
        st.plotly_chart(px.scatter(
            x=X_pca[:,0],
            y=X_pca[:,1],
            color=df_clean["Cluster"].astype(str),
            title="PCA Cluster Visualization"
        ))

        # Insights
        summary, important = get_insights(df_clean)

        st.subheader("Cluster Summary")
        st.dataframe(summary, width="stretch")

        st.subheader("Top Differentiating Features")
        st.write(important)

        # Downloads
        st.download_button("⬇ Download Data", df_clean.to_csv(index=False), "clusters.csv")
        st.download_button("💾 Download Model", pickle.dumps(model), "model.pkl")

    else:
        st.warning("No valid clustering found. Try different dataset.")

# =========================================
# MANUAL MODE
# =========================================
else:

    st.header("🎓 Manual Clustering")

    df_clean = clean_data(df)
    X = preprocess(df_clean)

    model_type = st.selectbox("Select Model", ["KMeans", "Agglomerative", "DBSCAN"])

    if model_type == "KMeans":
        k = st.slider("Clusters (K)", 2, 10, 3)

    elif model_type == "DBSCAN":
        eps = st.slider("eps", 0.1, 1.0, 0.5)
        min_samples = st.slider("min_samples", 2, 10, 5)

    if st.button("Train Model"):

        if model_type == "KMeans":
            model = KMeans(n_clusters=k)
            labels = model.fit_predict(X)

        elif model_type == "Agglomerative":
            model = AgglomerativeClustering(n_clusters=3)
            labels = model.fit_predict(X)

        else:
            model = DBSCAN(eps=eps, min_samples=min_samples)
            labels = model.fit_predict(X)

        st.session_state.labels = labels

    if st.session_state.labels is not None:

        labels = st.session_state.labels
        score = safe_score(X, labels)

        if score:
            st.metric("Silhouette Score", round(score, 3))
        else:
            st.warning("Score not valid for this clustering")

        X_pca = PCA(2).fit_transform(X)
        st.plotly_chart(px.scatter(
            x=X_pca[:,0],
            y=X_pca[:,1],
            color=pd.Series(labels).astype(str),
            title="Cluster Visualization"
        ))
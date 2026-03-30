import streamlit as st
import pandas as pd
import numpy as np
import pickle

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

import plotly.express as px
import shap

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Elite ML Dashboard", layout="wide")

st.title("🚀 Elite ML Clustering Dashboard")

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data():
    return pd.read_csv("cleaned_data.csv")

try:
    df = load_data()
    df = df.dropna()
except:
    st.error("❌ cleaned_data.csv not found")
    st.stop()

# =========================
# DATA OVERVIEW
# =========================
st.subheader("📊 Data Overview")

c1, c2, c3 = st.columns(3)
c1.metric("Rows", df.shape[0])
c2.metric("Columns", df.shape[1])
c3.metric("Missing", df.isnull().sum().sum())

st.dataframe(df.head())

# =========================
# FILTERS
# =========================
st.sidebar.header("🔍 Filters")

filtered_df = df.copy()

for col in df.select_dtypes(include=['int64','float64']).columns:
    min_val = float(df[col].min())
    max_val = float(df[col].max())
    val = st.sidebar.slider(col, min_val, max_val, (min_val, max_val))
    filtered_df = filtered_df[(filtered_df[col] >= val[0]) & (filtered_df[col] <= val[1])]

df = filtered_df

# =========================
# FEATURE ENGINEERING
# =========================
if 'weight' in df.columns and 'height' in df.columns:
    df['BMI'] = df['weight'] / ((df['height']/100) ** 2)

if 'ap_hi' in df.columns:
    df['BP_Category'] = pd.cut(df['ap_hi'],
                              bins=[0,120,140,200],
                              labels=['Normal','Elevated','High'])

# =========================
# TARGET
# =========================
target_col = st.selectbox("Select Target Column", df.columns)
X = df.drop(columns=[target_col])

# =========================
# PREPROCESSING
# =========================
num_cols = X.select_dtypes(include=['int64','float64']).columns
cat_cols = X.select_dtypes(include=['object','category']).columns

transformers = []

if len(num_cols) > 0:
    transformers.append(('num', StandardScaler(), num_cols))

if len(cat_cols) > 0:
    transformers.append(('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols))

preprocessor = ColumnTransformer(transformers)
X_processed = preprocessor.fit_transform(X)

# =========================
# SIDEBAR CONTROLS
# =========================
st.sidebar.header("🎛️ Model Controls")

mode = st.sidebar.radio("Mode", ["Single Model", "Compare Models"])

model_name = st.sidebar.selectbox("Model", ["KMeans", "DBSCAN", "Hierarchical"])

k = st.sidebar.slider("Clusters (K)", 2, 10, 3)
eps = st.sidebar.slider("DBSCAN eps", 0.1, 5.0, 0.5)
min_samples = st.sidebar.slider("Min Samples", 2, 10, 5)

run = st.sidebar.button("🚀 Run")

# =========================
# FUNCTION
# =========================
def run_model(model):
    labels = model.fit_predict(X_processed)

    if len(set(labels)) > 1 and -1 not in set(labels):
        idx = np.random.choice(len(X_processed), size=min(3000, len(X_processed)), replace=False)
        score = silhouette_score(X_processed[idx], labels[idx])
    else:
        score = -1

    return labels, score

# =========================
# TABS
# =========================
tab1, tab2, tab3 = st.tabs(["📊 Single Model", "⚖️ Comparison", "🔍 Explainability"])

# =========================
# SINGLE MODEL
# =========================
with tab1:
    if run and mode == "Single Model":

        if model_name == "KMeans":
            model = KMeans(n_clusters=k, random_state=42, n_init=10)
        elif model_name == "DBSCAN":
            model = DBSCAN(eps=eps, min_samples=min_samples)
        else:
            model = AgglomerativeClustering(n_clusters=k)

        labels, score = run_model(model)
        df['Cluster'] = labels

        # METRICS
        c1, c2, c3 = st.columns(3)
        c1.metric("Model", model_name)
        c2.metric("Clusters", len(set(labels)))
        c3.metric("Score", f"{score:.4f}" if score != -1 else "N/A")

        # PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_processed)

        pca_df = pd.DataFrame({
            "PC1": X_pca[:,0],
            "PC2": X_pca[:,1],
            "Cluster": labels.astype(str)
        })

        fig = px.scatter(pca_df, x="PC1", y="PC2", color="Cluster")
        st.plotly_chart(fig, use_container_width=True)

# =========================
# COMPARISON
# =========================
with tab2:
    if run and mode == "Compare Models":

        models = {
            "KMeans": KMeans(n_clusters=k, random_state=42, n_init=10),
            "DBSCAN": DBSCAN(eps=eps, min_samples=min_samples),
            "Hierarchical": AgglomerativeClustering(n_clusters=k)
        }

        results = {}

        for name, model in models.items():
            try:
                _, score = run_model(model)
                results[name] = score
            except:
                results[name] = -1

        scores_df = pd.DataFrame({
            "Model": list(results.keys()),
            "Score": list(results.values())
        })

        st.dataframe(scores_df)

        best = scores_df.sort_values(by="Score", ascending=False).iloc[0]
        st.success(f"🏆 Best Model: {best['Model']}")

# =========================
# SHAP EXPLAINABILITY
# =========================
with tab3:
    if run and mode == "Single Model":

        st.subheader("🔍 SHAP Feature Importance")

        try:
            sample = X_processed[:100]

            explainer = shap.Explainer(lambda x: model.fit_predict(x), sample)
            shap_values = explainer(sample)

            shap.summary_plot(shap_values, show=False)
            st.pyplot(bbox_inches='tight')

        except Exception as e:
            st.warning("SHAP not supported for this model")

# =========================
# DOWNLOAD
# =========================
st.subheader("⬇️ Download")

csv = df.to_csv(index=False).encode('utf-8')
st.download_button("Download Results", csv, "clustered_data.csv")

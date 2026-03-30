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

# =========================
# PAGE CONFIG (DARK UI)
# =========================
st.set_page_config(
    page_title="Advanced ML Dashboard",
    layout="wide"
)

st.title("🚀 Advanced ML Clustering Dashboard")

# =========================
# LOAD DATA (AUTO)
# =========================
@st.cache_data
def load_data():
    return pd.read_csv("cleaned_data.csv")

try:
    df = load_data()
    st.success("✅ Dataset loaded: cleaned_data.csv")
except:
    st.error("❌ cleaned_data.csv not found. Make sure it's in the same folder.")
    st.stop()

st.subheader("📊 Dataset Preview")
st.dataframe(df.head())

# =========================
# FEATURE ENGINEERING
# =========================
st.subheader("⚙️ Feature Engineering")

if 'weight' in df.columns and 'height' in df.columns:
    df['BMI'] = df['weight'] / ((df['height']/100) ** 2)

if 'ap_hi' in df.columns:
    df['BP_Category'] = pd.cut(df['ap_hi'],
                              bins=[0,120,140,200],
                              labels=['Normal','Elevated','High'])

# =========================
# TARGET SELECTION
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
    transformers.append(('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols))

preprocessor = ColumnTransformer(transformers)
X_processed = preprocessor.fit_transform(X)

# =========================
# SIDEBAR SETTINGS
# =========================
st.sidebar.header("⚙️ Model Controls")

k = st.sidebar.slider("Clusters (KMeans / Hierarchical)", 2, 10, 3)
eps = st.sidebar.slider("DBSCAN eps", 0.1, 5.0, 0.5)
min_samples = st.sidebar.slider("DBSCAN min_samples", 2, 10, 5)

# =========================
# TRAIN MODELS
# =========================
models = {
    "KMeans": KMeans(n_clusters=k, random_state=42, n_init=10),
    "DBSCAN": DBSCAN(eps=eps, min_samples=min_samples),
    "Hierarchical": AgglomerativeClustering(n_clusters=k)
}

results = {}

st.subheader("📊 Model Comparison")

for name, model in models.items():
    try:
        labels = model.fit_predict(X_processed)

        if len(set(labels)) > 1 and -1 not in set(labels):
            sample_idx = np.random.choice(len(X_processed), size=min(3000, len(X_processed)), replace=False)
            score = silhouette_score(X_processed[sample_idx], labels[sample_idx])
        else:
            score = -1

        results[name] = (labels, score)

    except Exception as e:
        results[name] = (None, -1)
        st.error(f"{name} failed: {e}")

scores_df = pd.DataFrame({
    "Model": results.keys(),
    "Silhouette Score": [v[1] for v in results.values()]
})

st.dataframe(scores_df)

best_model_name = scores_df.sort_values(by="Silhouette Score", ascending=False).iloc[0]["Model"]
st.success(f"🏆 Best Model: {best_model_name}")

labels = results[best_model_name][0]
df['Cluster'] = labels

# =========================
# PCA VISUALIZATION (PLOTLY)
# =========================
st.subheader("📉 Interactive Cluster Visualization")

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_processed)

pca_df = pd.DataFrame({
    "PC1": X_pca[:,0],
    "PC2": X_pca[:,1],
    "Cluster": labels.astype(str)
})

fig = px.scatter(
    pca_df,
    x="PC1",
    y="PC2",
    color="Cluster",
    title="Cluster Visualization (PCA)",
)

st.plotly_chart(fig, use_container_width=True)

# =========================
# CLUSTER PROFILE
# =========================
st.subheader("📊 Cluster Profile")

profile = df.groupby('Cluster').mean(numeric_only=True)
st.dataframe(profile)

# =========================
# AUTO INSIGHTS
# =========================
st.subheader("🧠 Auto Insights")

for cluster in profile.index:
    st.write(f"### Cluster {cluster}")
    row = profile.loc[cluster]

    if 'cardio' in df.columns:
        if row['cardio'] > 0.7:
            st.error("🚨 High Risk Group")
        elif row['cardio'] > 0.4:
            st.warning("⚠️ Medium Risk Group")
        else:
            st.success("✅ Low Risk Group")

    if 'BMI' in row and row['BMI'] > 25:
        st.warning("⚠️ Overweight cluster")

    if 'ap_hi' in row and row['ap_hi'] > 140:
        st.error("🚨 High Blood Pressure")

# =========================
# SAVE MODEL
# =========================
st.subheader("💾 Save Model")

if st.button("Save Best Model"):
    with open("best_model.pkl", "wb") as f:
        pickle.dump(models[best_model_name], f)
    st.success("Model saved as best_model.pkl")

# =========================
# DOWNLOAD DATA
# =========================
st.subheader("⬇️ Download Results")

csv = df.to_csv(index=False).encode('utf-8')
st.download_button("Download Clustered Data", csv, "clustered_data.csv")
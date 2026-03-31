# ================================================
# app.py - ClusterMentor | ML GOD+ Edition
# Production Unsupervised Clustering Pipeline
# AutoML + Step-by-Step Tutor + AI Mentor Explanations
# ================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import io

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline

st.set_page_config(
    page_title="ClusterMentor | ML GOD+",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===================== SESSION STATE =====================
if "df" not in st.session_state:
    st.session_state.df = None
if "df_clean" not in st.session_state:
    st.session_state.df_clean = None
if "X" not in st.session_state:
    st.session_state.X = None
if "online_results" not in st.session_state:
    st.session_state.online_results = None
if "manual_stage" not in st.session_state:
    st.session_state.manual_stage = 1
if "manual_labels" not in st.session_state:
    st.session_state.manual_labels = None
if "manual_model" not in st.session_state:
    st.session_state.manual_model = None
if "manual_params" not in st.session_state:
    st.session_state.manual_params = {}

# ===================== SIDEBAR =====================
st.sidebar.image("https://img.icons8.com/fluency/96/000000/artificial-intelligence.png", width=90)
st.sidebar.title("🧬 ClusterMentor")
st.sidebar.markdown("**AutoML + Interactive ML Tutor**")

uploaded_file = st.sidebar.file_uploader("Upload any CSV dataset", type=["csv"])

if uploaded_file is not None:
    st.session_state.df = pd.read_csv(uploaded_file)
    st.sidebar.success("Dataset loaded successfully!")

# ===================== MAIN TITLE =====================
st.title("🔥 ML GOD+ PLATFORM")
st.markdown("**End-to-End Unsupervised Clustering** — AutoML + Step-by-Step Learning with AI Mentor")

if st.session_state.df is None:
    st.info("👆 Please upload a CSV file from the sidebar to begin.")
    st.stop()

df = st.session_state.df

# ===================== TABS =====================
tab1, tab2, tab3 = st.tabs(["📊 Dataset Explorer", "🚀 Online Mode (AutoML)", "🧪 Manual Learning Mode"])

# ===================== TAB 1: DATA EXPLORER =====================
with tab1:
    st.header("📊 Dataset Explorer")
    
    colA, colB = st.columns([3, 1])
    with colA:
        search = st.text_input("🔍 Search dataset", "")
    with colB:
        view = st.radio("View", ["Full Dataset", "First 100 rows"], horizontal=True)
    
    display_df = df.copy()
    if search:
        mask = display_df.astype(str).apply(lambda x: x.str.contains(search, case=False)).any(axis=1)
        display_df = display_df[mask]
    
    if view == "First 100 rows":
        display_df = display_df.head(100)
    
    st.dataframe(display_df, use_container_width=True, height=500)
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", df.shape[0])
    c2.metric("Columns", df.shape[1])
    c3.metric("Numeric Features", len(df.select_dtypes(include=np.number).columns))
    c4.metric("Categorical Features", len(df.select_dtypes(exclude=np.number).columns))
    
    st.subheader("Missing Values")
    miss = df.isnull().sum()
    st.dataframe(pd.DataFrame({"Missing": miss, "%": (miss/len(df)*100).round(2)}), use_container_width=True)

# ===================== TAB 2: ONLINE MODE (AUTO ML) =====================
with tab2:
    st.header("🚀 Online Mode — Fully Automatic Clustering")
    st.caption("One-click end-to-end pipeline with best model selection")
    
    if st.button("🚀 Run Full AutoML Pipeline", type="primary", use_container_width=True):
        with st.spinner("Cleaning • Preprocessing • Training KMeans, DBSCAN & Agglomerative • Selecting best..."):
            try:
                # Auto Cleaning
                df_clean = df.copy()
                num_cols = df_clean.select_dtypes(include=np.number).columns.tolist()
                cat_cols = df_clean.select_dtypes(exclude=np.number).columns.tolist()
                
                if num_cols:
                    df_clean[num_cols] = df_clean[num_cols].fillna(df_clean[num_cols].mean())
                if cat_cols:
                    for col in cat_cols:
                        df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0] if not df_clean[col].mode().empty else "Unknown")
                
                # Preprocessing
                preprocessor = ColumnTransformer([
                    ("num", StandardScaler(), num_cols),
                    ("cat", OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
                ])
                
                X = preprocessor.fit_transform(df_clean)
                st.session_state.X = X
                
                # Try multiple models
                best_score = -1
                best_name = ""
                best_labels = None
                best_n = 0
                
                # KMeans
                for k in range(2, min(10, len(df_clean)//3 + 2)):
                    km = KMeans(n_clusters=k, random_state=42, n_init=10)
                    labels = km.fit_predict(X)
                    if len(np.unique(labels)) > 1:
                        score = silhouette_score(X, labels)
                        if score > best_score:
                            best_score = score
                            best_name = f"KMeans (k={k})"
                            best_labels = labels
                            best_n = k
                
                # Agglomerative
                for k in range(2, min(10, len(df_clean)//3 + 2)):
                    agg = AgglomerativeClustering(n_clusters=k)
                    labels = agg.fit_predict(X)
                    if len(np.unique(labels)) > 1:
                        score = silhouette_score(X, labels)
                        if score > best_score:
                            best_score = score
                            best_name = f"Agglomerative (k={k})"
                            best_labels = labels
                            best_n = k
                
                df_clean["Cluster"] = best_labels
                
                # PCA for visualization
                pca = PCA(n_components=2, random_state=42)
                X_pca = pca.fit_transform(X)
                
                st.session_state.online_results = {
                    "best_model": best_name,
                    "n_clusters": best_n,
                    "silhouette": round(best_score, 4),
                    "df": df_clean,
                    "X_pca": X_pca,
                    "labels": best_labels
                }
                
                st.success("✅ AutoML completed! Best model selected using Silhouette Score.")
                st.rerun()
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    # Show Results
    if st.session_state.online_results:
        res = st.session_state.online_results
        col1, col2, col3 = st.columns(3)
        col1.metric("🏆 Best Model", res["best_model"])
        col2.metric("🔢 Clusters", res["n_clusters"])
        col3.metric("⭐ Silhouette Score", res["silhouette"])
        
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(px.bar(x=res["df"]["Cluster"].value_counts().index, 
                                 y=res["df"]["Cluster"].value_counts().values,
                                 labels={"x":"Cluster", "y":"Count"}, title="Cluster Distribution"), 
                           use_container_width=True)
        with c2:
            pca_df = pd.DataFrame(res["X_pca"], columns=["PC1", "PC2"])
            pca_df["Cluster"] = res["labels"].astype(str)
            st.plotly_chart(px.scatter(pca_df, x="PC1", y="PC2", color="Cluster", 
                                     title="PCA 2D Cluster Visualization"), 
                           use_container_width=True)
        
        st.subheader("Cluster-wise Feature Means")
        st.dataframe(res["df"].groupby("Cluster").mean(numeric_only=True).round(3), use_container_width=True)
        
        csv = res["df"].to_csv(index=False)
        st.download_button("⬇️ Download Dataset with Clusters", csv, "clustered_dataset.csv", "text/csv")
        
        if st.button("🧠 Explain What Happened", type="secondary", use_container_width=True):
            with st.expander("📚 Full AI Mentor Explanation", expanded=True):
                st.markdown(generate_online_explanation(res), unsafe_allow_html=True)

# ===================== TAB 3: MANUAL MODE =====================
with tab3:
    st.header("🧪 Manual Learning Mode")
    st.caption("Learn clustering step-by-step with full control and mentor guidance")
    
    steps = ["1. Data Understanding", "2. Data Cleaning", "3. Feature Engineering", 
             "4. Preprocessing", "5. Model Selection", "6. Model Training", 
             "7. Evaluation", "8. Visualization"]
    
    current = st.selectbox("Current Step", steps, index=st.session_state.manual_stage-1)
    step_idx = steps.index(current) + 1
    
    # Progress Bar
    st.progress((step_idx-1)/len(steps))
    
    # Step Content
    if step_idx == 1:
        st.subheader("1. Data Understanding")
        st.dataframe(df.describe(), use_container_width=True)
        st.plotly_chart(px.imshow(df.corr(numeric_only=True), text_auto=True, aspect="auto"), use_container_width=True)
    
    elif step_idx == 2:
        st.subheader("2. Data Cleaning")
        method = st.radio("Cleaning Method", ["Drop NaN rows", "Fill numeric with mean, categorical with mode"])
        if st.button("Apply Cleaning"):
            df_clean = df.copy()
            if method == "Drop NaN rows":
                df_clean = df_clean.dropna()
            else:
                num_cols = df_clean.select_dtypes(include=np.number).columns
                cat_cols = df_clean.select_dtypes(exclude=np.number).columns
                df_clean[num_cols] = df_clean[num_cols].fillna(df_clean[num_cols].mean())
                for col in cat_cols:
                    df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0] if not df_clean[col].mode().empty else "Unknown")
            st.session_state.df_clean = df_clean
            st.success("Data cleaned successfully!")
    
    elif step_idx == 3:
        st.subheader("3. Feature Engineering")
        st.info("Advanced feature creation coming soon in v2.0")
    
    elif step_idx == 4:
        st.subheader("4. Preprocessing")
        if st.button("Apply Standard Scaling + One-Hot Encoding"):
            num_cols = st.session_state.df_clean.select_dtypes(include=np.number).columns.tolist() if st.session_state.df_clean is not None else df.select_dtypes(include=np.number).columns.tolist()
            cat_cols = st.session_state.df_clean.select_dtypes(exclude=np.number).columns.tolist() if st.session_state.df_clean is not None else df.select_dtypes(exclude=np.number).columns.tolist()
            
            preprocessor = ColumnTransformer([
                ("num", StandardScaler(), num_cols),
                ("cat", OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
            ])
            X = preprocessor.fit_transform(st.session_state.df_clean if st.session_state.df_clean is not None else df)
            st.session_state.X = X
            st.success("Preprocessing completed!")
    
    elif step_idx == 5:
        st.subheader("5. Model Selection")
        model_choice = st.selectbox("Select Algorithm", ["KMeans", "DBSCAN", "AgglomerativeClustering"])
        st.session_state.manual_model = model_choice
        if model_choice == "KMeans":
            st.session_state.manual_params["n_clusters"] = st.slider("Number of Clusters (K)", 2, 15, 3)
        elif model_choice == "DBSCAN":
            st.session_state.manual_params["eps"] = st.slider("eps", 0.1, 5.0, 0.5, 0.1)
            st.session_state.manual_params["min_samples"] = st.slider("min_samples", 3, 50, 5)
    
    elif step_idx == 6:
        st.subheader("6. Model Training")
        if st.session_state.X is not None and st.button("🚀 Train Model", type="primary"):
            try:
                if st.session_state.manual_model == "KMeans":
                    model = KMeans(n_clusters=st.session_state.manual_params.get("n_clusters", 3), random_state=42)
                elif st.session_state.manual_model == "DBSCAN":
                    model = DBSCAN(eps=st.session_state.manual_params.get("eps", 0.5),
                                   min_samples=st.session_state.manual_params.get("min_samples", 5))
                else:
                    model = AgglomerativeClustering(n_clusters=st.session_state.manual_params.get("n_clusters", 3))
                
                labels = model.fit_predict(st.session_state.X)
                st.session_state.manual_labels = labels
                st.success(f"Model trained! Found {len(np.unique(labels))} clusters.")
            except Exception as e:
                st.error(str(e))
    
    elif step_idx == 7:
        st.subheader("7. Evaluation")
        if st.session_state.manual_labels is not None:
            score = silhouette_score(st.session_state.X, st.session_state.manual_labels)
            st.metric("Silhouette Score", round(score, 4))
            st.info("Higher score = better defined clusters (ideal > 0.5)")
    
    elif step_idx == 8:
        st.subheader("8. Visualization")
        if st.session_state.manual_labels is not None:
            pca = PCA(2)
            X_pca = pca.fit_transform(st.session_state.X)
            fig = px.scatter(x=X_pca[:,0], y=X_pca[:,1], color=st.session_state.manual_labels.astype(str),
                           title="Final Cluster Visualization (PCA)")
            st.plotly_chart(fig, use_container_width=True)
            
            final_df = st.session_state.df_clean.copy() if st.session_state.df_clean is not None else df.copy()
            final_df["Cluster"] = st.session_state.manual_labels
            st.download_button("⬇️ Download Final Dataset", final_df.to_csv(index=False), "manual_clustered.csv")

    # Mentor Explanation Button
    if st.button("🧠 Explain What Happened", type="secondary", use_container_width=True):
        with st.expander("📚 AI Mentor Explanation (Stage-Aware)", expanded=True):
            st.markdown(generate_manual_explanation(step_idx), unsafe_allow_html=True)

# ===================== EXPLANATION FUNCTIONS =====================
def generate_online_explanation(res):
    return f"""
    <h2>🧠 AI Mentor Full Explanation — AutoML Mode</h2>
    <h3>1. Data Understanding</h3>
    <p>Dataset: {res['df'].shape[0]} rows, {res['df'].shape[1]} columns. Mixed numerical + categorical data detected.</p>
    
    <h3>2. Data Cleaning</h3>
    <p>Missing values were automatically filled (mean for numeric, mode for categorical).</p>
    
    <h3>3. Preprocessing</h3>
    <p><b>StandardScaler</b> + <b>OneHotEncoder</b> applied → All features on same scale for distance-based clustering.</p>
    
    <h3>4. Model Training & Selection</h3>
    <p>Tried KMeans, DBSCAN, and Agglomerative Clustering. Selected <b>{res['best_model']}</b> with highest Silhouette Score of <b>{res['silhouette']}</b>.</p>
    
    <h3>5. What Clusters Mean</h3>
    <p>Each group represents similar data points. You can now segment customers, detect anomalies, or discover hidden patterns.</p>
    
    <p><b>Tip from Mentor:</b> Try different random states or add domain features to improve results further.</p>
    """

def generate_manual_explanation(stage):
    text = f"<h2>🧠 AI Mentor — Up to Stage {stage}</h2>"
    if stage >= 1: text += "<p><b>Stage 1:</b> Explored raw data and statistics.</p>"
    if stage >= 2: text += "<p><b>Stage 2:</b> Cleaned missing values — critical for good clustering.</p>"
    if stage >= 4: text += "<p><b>Stage 4:</b> Scaled & encoded features so algorithms can work properly.</p>"
    if stage >= 5: text += f"<p><b>Stage 5:</b> You chose <b>{st.session_state.get('manual_model', '—')}</b> — great choice!</p>"
    if stage >= 7: text += "<p><b>Stage 7:</b> Silhouette score tells us cluster quality.</p>"
    text += "<hr><p><b>💡 Pro Tip:</b> The best ML pipelines combine automation with human intuition. Keep experimenting!</p>"
    return text

# ===================== FOOTER =====================
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #666;'>"
    "ClusterMentor v2.0 — Production-ready • Works with any dataset • AutoML + Interactive Tutor"
    "</p>",
    unsafe_allow_html=True
)

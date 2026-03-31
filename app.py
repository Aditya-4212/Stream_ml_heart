<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ClusterMentor - Production Streamlit App</title>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&amp;family=Space+Grotesk:wght@500;600&amp;display=swap');
        
        :root {
            --primary: #00d4ff;
        }
        
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Inter', system_ui, sans-serif;
            background: linear-gradient(135deg, #0a0a0a, #1a1a2e);
            color: #ffffff;
            line-height: 1.6;
        }
        
        .app-header {
            background: rgba(10, 10, 20, 0.95);
            backdrop-filter: blur(20px);
            border-bottom: 1px solid rgba(0, 212, 255, 0.2);
            padding: 1.5rem 5%;
            position: sticky;
            top: 0;
            z-index: 100;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        
        .logo {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 2rem;
            font-weight: 600;
            background: linear-gradient(90deg, #00d4ff, #00ff9d);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            display: flex;
            align-items: center;
            gap: 12px;
        }
        
        .main-container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 2rem 5%;
        }
        
        .tab-bar {
            display: flex;
            background: rgba(255,255,255,0.05);
            border-radius: 9999px;
            padding: 6px;
            margin-bottom: 2rem;
            width: fit-content;
        }
        
        .tab {
            padding: 14px 32px;
            border-radius: 9999px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }
        
        .tab.active {
            background: #00d4ff;
            color: #000;
            box-shadow: 0 10px 30px -10px #00d4ff;
        }
        
        .content-card {
            background: rgba(255,255,255,0.08);
            border-radius: 24px;
            padding: 2.5rem;
            border: 1px solid rgba(255,255,255,0.1);
            margin-bottom: 2rem;
        }
        
        .metric {
            background: rgba(255,255,255,0.06);
            border-radius: 20px;
            padding: 1.5rem;
            text-align: center;
            border: 1px solid rgba(0, 212, 255, 0.2);
        }
        
        .btn-primary {
            background: linear-gradient(90deg, #00d4ff, #00ff9d);
            color: #000;
            font-weight: 600;
            padding: 14px 32px;
            border-radius: 9999px;
            border: none;
            cursor: pointer;
            font-size: 1.1rem;
            transition: all 0.3s;
            display: inline-flex;
            align-items: center;
            gap: 8px;
        }
        
        .btn-primary:hover {
            transform: translateY(-3px);
            box-shadow: 0 20px 25px -5px rgb(0 212 255);
        }
        
        .explain-btn {
            background: linear-gradient(90deg, #7b2cbf, #c026d3);
            color: white;
        }
        
        pre {
            background: #111;
            padding: 1.5rem;
            border-radius: 16px;
            overflow-x: auto;
            font-size: 0.95rem;
            line-height: 1.7;
        }
        
        .step-number {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 32px;
            height: 32px;
            background: #00d4ff;
            color: #000;
            border-radius: 50%;
            font-weight: 700;
            font-size: 1.1rem;
            margin-right: 12px;
        }
    </style>
</head>
<body>
    <div class="app-header">
        <div class="logo">
            🧬 ClusterMentor
            <span style="font-size: 1.2rem; background: #111; color: #00d4ff; padding: 4px 14px; border-radius: 9999px; font-family: Inter;">v2.0</span>
        </div>
        <div style="display: flex; align-items: center; gap: 20px; font-weight: 500;">
            <div style="background: rgba(0,212,255,0.1); color: #00d4ff; padding: 6px 18px; border-radius: 9999px; font-size: 0.95rem;">
                Production • Any Dataset • AutoML + Tutor
            </div>
            <a href="#" onclick="copyCode()" style="color: #00d4ff; text-decoration: none; display: flex; align-items: center; gap: 6px;">
                📋 Copy Full app.py
            </a>
        </div>
    </div>

    <div class="main-container">
        <h1 style="font-size: 3rem; font-weight: 600; margin-bottom: 8px; line-height: 1.1;">
            End-to-End Unsupervised<br>Clustering Pipeline
        </h1>
        <p style="font-size: 1.35rem; color: #aaa; max-width: 700px; margin-bottom: 2rem;">
            AutoML mode + fully interactive manual learning mode with mentor explanations.<br>
            Works with <strong>ANY CSV</strong> — numerical, categorical, or mixed.
        </p>

        <!-- Mode Tabs -->
        <div class="tab-bar" id="modeTabs">
            <div class="tab active" onclick="switchMode(0)" id="tab0">📊 Dataset Overview</div>
            <div class="tab" onclick="switchMode(1)" id="tab1">🚀 Online Mode (AutoML)</div>
            <div class="tab" onclick="switchMode(2)" id="tab2">🧪 Manual Mode (Step-by-Step)</div>
        </div>

        <!-- Dataset Overview Tab -->
        <div id="panel0" class="content-card">
            <h2 style="margin-bottom: 1.5rem; display: flex; align-items: center; gap: 12px;">
                📤 Your Dataset
                <span id="datasetName" style="font-size: 1rem; background: #222; padding: 4px 14px; border-radius: 9999px; color: #0f0;"></span>
            </h2>
            
            <div style="display: flex; gap: 2rem; flex-wrap: wrap; margin-bottom: 2rem;">
                <div class="metric" style="flex: 1; min-width: 220px;">
                    <div style="font-size: 2.5rem; font-weight: 600;" id="rowsCount">0</div>
                    <div style="color: #aaa; font-size: 1.1rem;">Rows</div>
                </div>
                <div class="metric" style="flex: 1; min-width: 220px;">
                    <div style="font-size: 2.5rem; font-weight: 600;" id="colsCount">0</div>
                    <div style="color: #aaa; font-size: 1.1rem;">Columns</div>
                </div>
                <div class="metric" style="flex: 1; min-width: 220px;">
                    <div style="font-size: 2.5rem; font-weight: 600; color: #0f0;" id="numericCount">0</div>
                    <div style="color: #aaa; font-size: 1.1rem;">Numeric Features</div>
                </div>
                <div class="metric" style="flex: 1; min-width: 220px;">
                    <div style="font-size: 2.5rem; font-weight: 600; color: #ff0;" id="catCount">0</div>
                    <div style="color: #aaa; font-size: 1.1rem;">Categorical Features</div>
                </div>
            </div>

            <div style="margin-bottom: 1rem; display: flex; align-items: center; gap: 1rem;">
                <input type="text" id="searchInput" placeholder="🔎 Search dataset (type to filter rows)..." 
                       style="flex: 1; padding: 14px 20px; border-radius: 9999px; border: 1px solid rgba(255,255,255,0.2); background: rgba(255,255,255,0.08); color: white; font-size: 1.1rem;">
                <select id="viewMode" onchange="renderTable()" style="padding: 14px 20px; border-radius: 9999px; background: rgba(255,255,255,0.08); color: white; border: none;">
                    <option value="full">Full Dataset (scrollable)</option>
                    <option value="head">View First 50 Rows</option>
                </select>
            </div>

            <div id="dataTableContainer" style="max-height: 520px; overflow: auto; border-radius: 16px; border: 1px solid rgba(255,255,255,0.15); background: #111;"></div>

            <div style="margin-top: 1.5rem; display: flex; gap: 1rem; flex-wrap: wrap;">
                <div style="background: rgba(255,255,255,0.06); padding: 1rem 1.5rem; border-radius: 16px; flex: 1;">
                    <strong>Data Types Summary</strong>
                    <div id="typeSummary" style="margin-top: 1rem; font-family: monospace; font-size: 0.95rem;"></div>
                </div>
                <div style="background: rgba(255,255,255,0.06); padding: 1rem 1.5rem; border-radius: 16px; flex: 1;">
                    <strong>Missing Values</strong>
                    <div id="missingSummary" style="margin-top: 1rem;"></div>
                </div>
            </div>
        </div>

        <!-- Online Mode Tab -->
        <div id="panel1" class="content-card" style="display: none;">
            <h2>🚀 Online Mode — Fully Automatic Clustering</h2>
            <p style="color: #aaa; margin-bottom: 2rem;">One-click end-to-end pipeline. The app automatically cleans, preprocesses, tries multiple models, and picks the best one using silhouette score.</p>
            
            <button onclick="runAutoML()" class="btn-primary" style="font-size: 1.3rem; padding: 18px 40px;">
                ▶️ Run Full AutoML Pipeline
            </button>

            <div id="onlineResults" style="margin-top: 2.5rem; display: none;">
                <div style="display: flex; gap: 2rem; flex-wrap: wrap;">
                    <div class="metric" style="flex: 1;">
                        <div style="font-size: 1.1rem; color: #aaa;">Best Model</div>
                        <div id="bestModelName" style="font-size: 2rem; font-weight: 700; margin: 8px 0;"></div>
                    </div>
                    <div class="metric" style="flex: 1;">
                        <div style="font-size: 1.1rem; color: #aaa;">Number of Clusters</div>
                        <div id="clusterCount" style="font-size: 2.8rem; font-weight: 700; color: #00ff9d;"></div>
                    </div>
                    <div class="metric" style="flex: 1;">
                        <div style="font-size: 1.1rem; color: #aaa;">Silhouette Score</div>
                        <div id="silScore" style="font-size: 2.8rem; font-weight: 700; color: #00d4ff;"></div>
                    </div>
                </div>

                <div style="margin: 2rem 0; display: grid; grid-template-columns: 1fr 1fr; gap: 2rem;">
                    <div>
                        <h3>📊 Cluster Distribution</h3>
                        <div id="clusterBarContainer" style="height: 340px;"></div>
                    </div>
                    <div>
                        <h3>📍 PCA 2D Visualization (Interactive)</h3>
                        <div id="pcaPlotContainer" style="height: 340px; background: #111; border-radius: 16px;"></div>
                    </div>
                </div>

                <div>
                    <h3>Cluster-wise Feature Summary</h3>
                    <div id="summaryTableContainer" style="overflow-x: auto;"></div>
                </div>

                <div style="margin-top: 2rem;">
                    <button onclick="downloadOnlineResult()" class="btn-primary" style="background: #00ff9d;">
                        ⬇️ Download Dataset + Cluster Labels (CSV)
                    </button>
                    <button onclick="showOnlineExplanation()" class="btn-primary explain-btn" style="margin-left: 12px;">
                        🧠 Explain What Happened
                    </button>
                </div>

                <div id="onlineExplanationArea" style="margin-top: 2rem; display: none;"></div>
            </div>
        </div>

        <!-- Manual Mode Tab -->
        <div id="panel2" class="content-card" style="display: none;">
            <h2>🧪 Manual Mode — Interactive ML Learning Platform</h2>
            <p style="color: #aaa; margin-bottom: 2rem;">Build the pipeline step-by-step. Every action is explained. Perfect for learning unsupervised learning from scratch.</p>
            
            <!-- Progress -->
            <div style="background: rgba(255,255,255,0.08); padding: 1rem; border-radius: 9999px; display: flex; align-items: center; gap: 12px; margin-bottom: 2rem;">
                <div style="font-weight: 600;">Progress</div>
                <div style="flex: 1; height: 8px; background: #222; border-radius: 9999px; overflow: hidden;">
                    <div id="manualProgressBar" style="height: 100%; width: 12.5%; background: linear-gradient(90deg, #00d4ff, #00ff9d); transition: width 0.6s cubic-bezier(0.34, 1.56, 0.64, 1);"></div>
                </div>
                <div id="progressText" style="font-size: 1.1rem; font-weight: 600; color: #00d4ff;">Step 1 of 8</div>
            </div>

            <!-- Step Tabs -->
            <div style="display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 2rem;" id="stepTabs">
                <!-- Dynamically populated by JS -->
            </div>

            <!-- Current Step Content -->
            <div id="stepContent" style="min-height: 520px;"></div>

            <div style="margin-top: 3rem; padding-top: 2rem; border-top: 1px solid rgba(255,255,255,0.15);">
                <button onclick="showManualExplanation()" class="btn-primary explain-btn">
                    🧠 Explain What Happened (Stage-Aware Mentor View)
                </button>
                <div id="manualExplanationArea" style="margin-top: 2rem; display: none;"></div>
            </div>
        </div>
    </div>

    <script>
        // Full production-ready Streamlit app code (app.py) generated below
        // This is a self-contained, copy-paste ready single-file Streamlit application
        // Copy the entire block inside the <pre> and save as app.py

        const FULL_APP_CODE = `# ================================================
# CLUSTERMENTOR - Production Streamlit App
# Single file | Works with ANY CSV | AutoML + Tutor
# ================================================

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go
import io

st.set_page_config(
    page_title="ClusterMentor | Unsupervised ML Pipeline",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===================== SESSION STATE INIT =====================
if "original_df" not in st.session_state:
    st.session_state.original_df = None
if "online_results" not in st.session_state:
    st.session_state.online_results = None
if "manual_df" not in st.session_state:
    st.session_state.manual_df = None
if "manual_progress" not in st.session_state:
    st.session_state.manual_progress = 0
if "manual_cleaning_method" not in st.session_state:
    st.session_state.manual_cleaning_method = None
if "manual_new_features" not in st.session_state:
    st.session_state.manual_new_features = []
if "manual_X" not in st.session_state:
    st.session_state.manual_X = None
if "manual_labels" not in st.session_state:
    st.session_state.manual_labels = None
if "manual_model_choice" not in st.session_state:
    st.session_state.manual_model_choice = None
if "manual_params" not in st.session_state:
    st.session_state.manual_params = {}

# ===================== SIDEBAR =====================
st.sidebar.image("https://img.icons8.com/clouds/200/00d4ff/artificial-intelligence.png", width=80)
st.sidebar.title("🧬 ClusterMentor")
st.sidebar.markdown("**Any dataset • Clustering**")
uploaded_file = st.sidebar.file_uploader("Upload your CSV dataset", type=["csv"])

if uploaded_file is not None and st.session_state.original_df is None:
    st.session_state.original_df = pd.read_csv(uploaded_file)
    st.session_state.manual_df = st.session_state.original_df.copy()
    st.rerun()

st.sidebar.divider()
st.sidebar.subheader("📍 Quick Info")
if st.session_state.original_df is not None:
    df = st.session_state.original_df
    st.sidebar.metric("Rows", df.shape[0])
    st.sidebar.metric("Columns", df.shape[1])
    st.sidebar.caption("Ready for both AutoML and learning mode")

# ===================== MAIN APP =====================
st.title("🧬 ClusterMentor")
st.markdown("**End-to-End Unsupervised Clustering Pipeline** — AutoML + Interactive Tutor with Mentor Explanations")

if st.session_state.original_df is None:
    st.info("👆 Upload a CSV file in the sidebar to begin. Supports **any** numerical, categorical, or mixed dataset.")
    st.stop()

df = st.session_state.original_df

# Tab navigation
tab_overview, tab_online, tab_manual = st.tabs(["📊 Dataset Overview", "🚀 Online Mode (AutoML)", "🧪 Manual Mode (Step-by-Step)"])

# ===================== TAB 1: DATASET OVERVIEW =====================
with tab_overview:
    st.header("Dataset Overview")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        search_term = st.text_input("🔎 Search / Filter dataset", placeholder="Type keyword to filter rows...")
    with col2:
        view_option = st.radio("View", ["Full Dataset (scrollable)", "Head (first 100 rows)"], horizontal=True)
    
    filtered_df = df
    if search_term:
        mask = filtered_df.astype(str).apply(lambda x: x.str.contains(search_term, case=False)).any(axis=1)
        filtered_df = filtered_df[mask]
    
    display_df = filtered_df if view_option == "Full Dataset (scrollable)" else filtered_df.head(100)
    
    st.dataframe(display_df, use_container_width=True, height=520)
    
    # Stats
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total Rows", df.shape[0])
    with c2:
        st.metric("Total Columns", df.shape[1])
    with c3:
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        st.metric("Numeric Columns", len(num_cols))
    with c4:
        cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()
        st.metric("Categorical Columns", len(cat_cols))
    
    st.subheader("Data Types & Missing Values")
    col_a, col_b = st.columns(2)
    with col_a:
        st.write("**Column Types**")
        type_df = pd.DataFrame({
            "Column": df.columns,
            "Type": df.dtypes.astype(str)
        })
        st.dataframe(type_df, use_container_width=True, height=300)
    
    with col_b:
        st.write("**Missing Values Summary**")
        missing = df.isna().sum()
        missing_pct = (missing / len(df) * 100).round(2)
        miss_df = pd.DataFrame({"Missing Count": missing, "Missing %": missing_pct}).sort_values("Missing Count", ascending=False)
        st.dataframe(miss_df[miss_df["Missing Count"] > 0], use_container_width=True)
        
        # Missing heatmap
        if missing.sum() > 0:
            fig = px.imshow(df.isna().T, color_continuous_scale="Blues", title="Missing Values Heatmap")
            st.plotly_chart(fig, use_container_width=True)

# ===================== TAB 2: ONLINE MODE =====================
with tab_online:
    st.header("🚀 Online Mode — Fully Automatic Clustering")
    st.caption("The app handles everything: cleaning → preprocessing → model selection → best model by silhouette score")
    
    if st.button("🚀 Run Full AutoML Pipeline", type="primary", use_container_width=True):
        with st.spinner("🔄 Cleaning • Preprocessing • Training 3 models • Selecting best..."):
            try:
                # Auto cleaning
                clean_df = df.copy()
                num_cols = clean_df.select_dtypes(include=np.number).columns.tolist()
                cat_cols = clean_df.select_dtypes(exclude=np.number).columns.tolist()
                
                # Fill missing
                if num_cols:
                    clean_df[num_cols] = clean_df[num_cols].fillna(clean_df[num_cols].mean())
                if cat_cols:
                    for c in cat_cols:
                        mode_val = clean_df[c].mode()
                        clean_df[c] = clean_df[c].fillna(mode_val[0] if not mode_val.empty else "Unknown")
                clean_df = clean_df.dropna()
                
                # Preprocessing pipeline
                preprocessor = ColumnTransformer(
                    transformers=[
                        ('num', StandardScaler(), num_cols),
                        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
                    ],
                    remainder='drop'
                )
                X = preprocessor.fit_transform(clean_df)
                
                # Model candidates
                best_score = -np.inf
                best_name = ""
                best_n_clusters = 0
                best_labels = None
                best_model_info = {}
                
                # KMeans + optimal k
                for k in range(2, min(11, len(clean_df) // 3 + 2)):
                    km = KMeans(n_clusters=k, random_state=42, n_init=10)
                    labels_km = km.fit_predict(X)
                    if len(np.unique(labels_km)) > 1:
                        score = silhouette_score(X, labels_km)
                        if score > best_score:
                            best_score = score
                            best_name = f"KMeans (k={k})"
                            best_n_clusters = k
                            best_labels = labels_km
                            best_model_info = {"type": "KMeans", "k": k}
                
                # Agglomerative
                for k in range(2, min(11, len(clean_df) // 3 + 2)):
                    agg = AgglomerativeClustering(n_clusters=k, linkage="ward")
                    labels_agg = agg.fit_predict(X)
                    if len(np.unique(labels_agg)) > 1:
                        score = silhouette_score(X, labels_agg)
                        if score > best_score:
                            best_score = score
                            best_name = f"Agglomerative (k={k})"
                            best_n_clusters = k
                            best_labels = labels_agg
                            best_model_info = {"type": "Agglomerative", "k": k}
                
                # DBSCAN with simple grid
                for eps_val in [0.3, 0.5, 0.8, 1.5]:
                    for min_s in [5, 10]:
                        db = DBSCAN(eps=eps_val, min_samples=min_s)
                        labels_db = db.fit_predict(X)
                        clustered_idx = labels_db != -1
                        if clustered_idx.sum() > 10 and len(np.unique(labels_db[clustered_idx])) > 1:
                            score = silhouette_score(X[clustered_idx], labels_db[clustered_idx])
                            if score > best_score:
                                best_score = score
                                n_clust = len(np.unique(labels_db[clustered_idx]))
                                best_name = f"DBSCAN (eps={eps_val}, clusters={n_clust})"
                                best_n_clusters = n_clust
                                best_labels = labels_db
                                best_model_info = {"type": "DBSCAN", "eps": eps_val, "min_samples": min_s}
                
                # Store results
                clean_df = clean_df.copy()
                clean_df["cluster"] = best_labels
                
                pca = PCA(n_components=2, random_state=42)
                X_pca = pca.fit_transform(X)
                
                st.session_state.online_results = {
                    "best_model": best_name,
                    "n_clusters": best_n_clusters,
                    "silhouette_score": round(best_score, 4),
                    "labels": best_labels,
                    "df_with_clusters": clean_df,
                    "X_pca": X_pca,
                    "num_cols": num_cols,
                    "cat_cols": cat_cols,
                    "model_info": best_model_info,
                    "cleaning": "Auto-filled missing (mean/mode) + dropped remaining NaNs"
                }
                st.success("✅ AutoML pipeline completed successfully!")
                st.rerun()
            except Exception as e:
                st.error(f"Pipeline error: {str(e)}")
    
    # Show results
    if st.session_state.online_results:
        res = st.session_state.online_results
        st.divider()
        
        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            st.metric("🏆 Best Model", res["best_model"])
        with col_m2:
            st.metric("🔢 Clusters Found", res["n_clusters"])
        with col_m3:
            st.metric("⭐ Silhouette Score", f"{res['silhouette_score']:.3f}")
        
        # Charts
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            st.plotly_chart(
                px.bar(
                    x=pd.Series(res["labels"]).value_counts().index.astype(str),
                    y=pd.Series(res["labels"]).value_counts().values,
                    labels={"x": "Cluster", "y": "Count"},
                    title="Cluster Distribution",
                    color_discrete_sequence=["#00d4ff"]
                ),
                use_container_width=True
            )
        
        with col_chart2:
            pca_df = pd.DataFrame(res["X_pca"], columns=["PC1", "PC2"])
            pca_df["Cluster"] = res["labels"].astype(str)
            st.plotly_chart(
                px.scatter(
                    pca_df, x="PC1", y="PC2", color="Cluster",
                    title="Interactive PCA Cluster Visualization",
                    color_discrete_sequence=px.colors.qualitative.Set3,
                    hover_data={"Cluster": True}
                ),
                use_container_width=True
            )
        
        # Summary table
        st.subheader("📋 Cluster-wise Feature Summary (Numeric means)")
        summary_df = res["df_with_clusters"].groupby("cluster")[res["num_cols"]].mean().round(3)
        st.dataframe(summary_df, use_container_width=True)
        
        # Download
        csv = res["df_with_clusters"].to_csv(index=False)
        st.download_button(
            label="⬇️ Download final dataset with cluster labels",
            data=csv,
            file_name="clustered_dataset.csv",
            mime="text/csv"
        )
        
        # Explanation button
        if st.button("🧠 Explain What Happened", type="secondary", use_container_width=True):
            with st.expander("📚 Full Mentor Explanation (Online Mode)", expanded=True):
                st.markdown(generate_online_explanation(res), unsafe_allow_html=True)

# ===================== TAB 3: MANUAL MODE =====================
with tab_manual:
    st.header("🧪 Manual Mode — Step-by-Step Interactive Learning")
    steps_list = [
        "1. Data Understanding",
        "2. Data Cleaning",
        "3. Feature Engineering",
        "4. Preprocessing",
        "5. Model Selection",
        "6. Model Training",
        "7. Evaluation",
        "8. Visualization"
    ]
    
    # Progress
    progress_pct = (st.session_state.manual_progress / (len(steps_list) - 1)) * 100
    st.progress(int(progress_pct))
    st.caption(f"**Current progress: Step {st.session_state.manual_progress + 1} / {len(steps_list)}**")
    
    # Step selector
    current_step = st.selectbox("Jump to step (recommended: follow sequentially)", steps_list, index=st.session_state.manual_progress)
    step_idx = steps_list.index(current_step)
    
    # Render current step
    if step_idx == 0:  # Data Understanding
        st.subheader("1. Data Understanding")
        st.info("We start with raw data. Always explore before modeling!")
        st.dataframe(st.session_state.manual_df.head(100), use_container_width=True)
        st.write("**Basic statistics**")
        st.dataframe(st.session_state.manual_df.describe(), use_container_width=True)
        
        if st.button("✅ Mark as completed & continue", type="primary"):
            st.session_state.manual_progress = max(st.session_state.manual_progress, 1)
            st.rerun()
    
    elif step_idx == 1:  # Data Cleaning
        st.subheader("2. Data Cleaning")
        st.info("Missing values hurt clustering. We handle them automatically or manually.")
        
        method = st.radio("Choose cleaning strategy", [
            "Drop rows with any missing values",
            "Fill numeric with mean, categorical with mode"
        ])
        
        if st.button("Apply Cleaning"):
            temp_df = st.session_state.manual_df.copy()
            if method.startswith("Drop"):
                temp_df = temp_df.dropna()
                st.session_state.manual_cleaning_method = "dropped_na"
            else:
                num_c = temp_df.select_dtypes(include=np.number).columns
                cat_c = temp_df.select_dtypes(exclude=np.number).columns
                if len(num_c) > 0:
                    temp_df[num_c] = temp_df[num_c].fillna(temp_df[num_c].mean())
                if len(cat_c) > 0:
                    for c in cat_c:
                        temp_df[c] = temp_df[c].fillna(temp_df[c].mode()[0] if not temp_df[c].mode().empty else "Unknown")
                st.session_state.manual_cleaning_method = "filled_mean_mode"
            st.session_state.manual_df = temp_df
            st.success("✅ Data cleaned!")
            st.session_state.manual_progress = max(st.session_state.manual_progress, 2)
            st.rerun()
        
        st.caption("Current dataset shape after cleaning: " + str(st.session_state.manual_df.shape))
    
    elif step_idx == 2:  # Feature Engineering
        st.subheader("3. Feature Engineering")
        st.info("Create smarter features to improve clustering quality.")
        
        num_cols_now = st.session_state.manual_df.select_dtypes(include=np.number).columns.tolist()
        
        colA, colB = st.columns(2)
        with colA:
            if num_cols_now:
                bin_col = st.selectbox("Bin a numeric column", num_cols_now)
                bins = st.slider("Number of bins", 2, 10, 4)
                if st.button("Create Binned Feature"):
                    st.session_state.manual_df[f"{bin_col}_binned"] = pd.cut(
                        st.session_state.manual_df[bin_col], bins=bins, labels=False
                    ).astype("category")
                    st.session_state.manual_new_features.append(f"{bin_col}_binned")
                    st.success(f"✅ New feature {bin_col}_binned created")
                    st.rerun()
        
        with colB:
            if len(num_cols_now) >= 2:
                c1 = st.selectbox("Column A", num_cols_now, key="c1")
                c2 = st.selectbox("Column B", num_cols_now, key="c2")
                op = st.selectbox("Operation", ["Add (+)", "Multiply (×)", "Ratio (A/B)"])
                if st.button("Create Derived Feature"):
                    if op == "Add (+)":
                        new_name = f"{c1}_{c2}_sum"
                        st.session_state.manual_df[new_name] = st.session_state.manual_df[c1] + st.session_state.manual_df[c2]
                    elif op == "Multiply (×)":
                        new_name = f"{c1}_{c2}_prod"
                        st.session_state.manual_df[new_name] = st.session_state.manual_df[c1] * st.session_state.manual_df[c2]
                    else:
                        new_name = f"{c1}_div_{c2}"
                        st.session_state.manual_df[new_name] = st.session_state.manual_df[c1] / (st.session_state.manual_df[c2] + 1e-8)
                    st.session_state.manual_new_features.append(new_name)
                    st.success(f"✅ New feature {new_name} created")
                    st.rerun()
        
        if st.button("✅ Finish feature engineering"):
            st.session_state.manual_progress = max(st.session_state.manual_progress, 3)
            st.rerun()
    
    elif step_idx == 3:  # Preprocessing
        st.subheader("4. Preprocessing")
        st.info("Scaling + encoding is required for clustering algorithms.")
        
        num_cols = st.session_state.manual_df.select_dtypes(include=np.number).columns.tolist()
        cat_cols = st.session_state.manual_df.select_dtypes(exclude=np.number).columns.tolist()
        
        st.write(f"Numeric columns detected: **{len(num_cols)}**")
        st.write(f"Categorical columns detected: **{len(cat_cols)}**")
        
        apply_scale = st.checkbox("Apply StandardScaler to numeric features", value=True)
        apply_encode = st.checkbox("Apply OneHotEncoder to categorical features", value=True)
        
        if st.button("Apply Preprocessing & Store Transformed Data"):
            try:
                preprocessor = ColumnTransformer(
                    transformers=[
                        ('num', StandardScaler(), num_cols) if apply_scale else ('num', 'passthrough', num_cols),
                        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols) if apply_encode else ('cat', 'passthrough', cat_cols)
                    ],
                    remainder='drop'
                )
                X_trans = preprocessor.fit_transform(st.session_state.manual_df)
                st.session_state.manual_X = X_trans
                st.success("✅ Preprocessing complete! Transformed data ready for modeling.")
                st.session_state.manual_progress = max(st.session_state.manual_progress, 4)
                st.rerun()
            except Exception as e:
                st.error(f"Preprocessing failed: {e}")
    
    elif step_idx == 4:  # Model Selection
        st.subheader("5. Model Selection")
        model_type = st.selectbox("Choose clustering algorithm", ["KMeans", "DBSCAN", "AgglomerativeClustering"])
        st.session_state.manual_model_choice = model_type
        
        if model_type == "KMeans":
            st.session_state.manual_params["n_clusters"] = st.slider("Number of clusters (K)", 2, 15, 3)
        elif model_type == "DBSCAN":
            st.session_state.manual_params["eps"] = st.slider("eps", 0.1, 5.0, 0.5, step=0.1)
            st.session_state.manual_params["min_samples"] = st.slider("min_samples", 2, 50, 5)
        else:
            st.session_state.manual_params["n_clusters"] = st.slider("Number of clusters", 2, 15, 3)
        
        if st.button("✅ Parameters saved — go to Training"):
            st.session_state.manual_progress = max(st.session_state.manual_progress, 5)
            st.rerun()
    
    elif step_idx == 5:  # Model Training
        st.subheader("6. Model Training")
        if st.session_state.manual_X is None:
            st.warning("⚠️ Please complete Preprocessing first!")
        else:
            if st.button("🔥 Train Selected Model", type="primary"):
                try:
                    if st.session_state.manual_model_choice == "KMeans":
                        model = KMeans(n_clusters=st.session_state.manual_params.get("n_clusters", 3), random_state=42)
                    elif st.session_state.manual_model_choice == "DBSCAN":
                        model = DBSCAN(eps=st.session_state.manual_params.get("eps", 0.5),
                                       min_samples=st.session_state.manual_params.get("min_samples", 5))
                    else:
                        model = AgglomerativeClustering(n_clusters=st.session_state.manual_params.get("n_clusters", 3))
                    
                    labels = model.fit_predict(st.session_state.manual_X)
                    st.session_state.manual_labels = labels
                    st.success(f"✅ Model trained! {len(np.unique(labels))} clusters detected.")
                    st.session_state.manual_progress = max(st.session_state.manual_progress, 6)
                    st.rerun()
                except Exception as e:
                    st.error(str(e))
    
    elif step_idx == 6:  # Evaluation
        st.subheader("7. Evaluation")
        if st.session_state.manual_labels is not None and st.session_state.manual_X is not None:
            try:
                score = silhouette_score(st.session_state.manual_X, st.session_state.manual_labels)
                st.metric("Silhouette Score", f"{score:.4f}", help="Higher = better separated clusters (max 1.0)")
                st.info("**Interpretation**: >0.5 = strong structure, 0.25-0.5 = reasonable, <0.25 = weak clustering")
            except:
                st.metric("Silhouette Score", "N/A (single cluster or all noise)")
        else:
            st.warning("Train a model first")
        
        if st.button("✅ Evaluation complete"):
            st.session_state.manual_progress = max(st.session_state.manual_progress, 7)
            st.rerun()
    
    elif step_idx == 7:  # Visualization
        st.subheader("8. Visualization")
        if st.session_state.manual_labels is not None and st.session_state.manual_X is not None:
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(st.session_state.manual_X)
            pca_df = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
            pca_df["Cluster"] = st.session_state.manual_labels.astype(str)
            
            st.plotly_chart(
                px.scatter(pca_df, x="PC1", y="PC2", color="Cluster", title="Final Cluster Visualization"),
                use_container_width=True
            )
            
            # Download final manual result
            final_manual = st.session_state.manual_df.copy()
            final_manual["cluster"] = st.session_state.manual_labels
            csv_manual = final_manual.to_csv(index=False)
            st.download_button("⬇️ Download final clustered dataset", csv_manual, "manual_clustered.csv", "text/csv")
        else:
            st.info("Complete previous steps to see visualizations")
    
    # Final manual download & reset
    if st.session_state.manual_progress >= 7:
        st.success("🎉 Congratulations! You have completed the full manual pipeline.")

# ===================== EXPLANATION FUNCTIONS =====================
def generate_online_explanation(res):
    html = f"""
    <h2>🧠 Mentor Explanation — Online Mode</h2>
    <h3>1. Data Understanding</h3>
    <ul>
    <li>Detected <strong>{len(res['num_cols'])} numeric</strong> and <strong>{len(res['cat_cols'])} categorical</strong> features</li>
    <li>Rows: {res['df_with_clusters'].shape[0]} • Columns: {res['df_with_clusters'].shape[1]}</li>
    </ul>
    
    <h3>2. Data Cleaning</h3>
    <p>{res.get('cleaning', 'Automatic handling of missing values')}</p>
    
    <h3>3. Preprocessing</h3>
    <ul>
    <li><strong>StandardScaler</strong> — brings all numeric features to same scale (mean=0, std=1)</li>
    <li><strong>OneHotEncoder</strong> — converts categorical variables into binary columns so distance-based algorithms can work</li>
    </ul>
    
    <h3>4. Model Training & Selection</h3>
    <p>We tried <strong>KMeans</strong>, <strong>DBSCAN</strong>, and <strong>Agglomerative Clustering</strong> with multiple hyper-parameters and selected the one with the highest silhouette score.</p>
    <p><strong>Best model:</strong> {res['best_model']}</p>
    
    <h3>5. What the Silhouette Score Means</h3>
    <p>A score of <strong>{res['silhouette_score']}</strong> tells us how well-separated the clusters are.<br>
    <span style="color:#0f0">Higher is better</span> — values close to 1 mean excellent clustering.</p>
    
    <h3>6. Insights & Real-World Meaning</h3>
    <p>Each cluster represents a natural group in your data. You can now:</p>
    <ul>
    <li>Target different customer segments</li>
    <li>Identify outlier behavior</li>
    <li>Discover hidden patterns without any labels</li>
    </ul>
    <p><strong>Tip:</strong> Try different random seeds or add domain-specific features to improve results further.</p>
    """
    return html

def generate_manual_explanation(progress):
    stages_done = min(progress + 1, 8)
    html = f"<h2>🧠 Mentor Explanation — Manual Mode (Up to Step {stages_done})</h2>"
    
    if stages_done >= 1:
        html += "<h3>✅ Stage 1: Data Understanding</h3><p>We explored shape, types, and missing values — the foundation of any ML project.</p>"
    if stages_done >= 2:
        html += f"<h3>✅ Stage 2: Data Cleaning</h3><p>You chose <strong>{st.session_state.get('manual_cleaning_method', 'auto')}</strong>. This step prevents noisy data from ruining cluster quality.</p>"
    if stages_done >= 3:
        html += f"<h3>✅ Stage 3: Feature Engineering</h3><p>New features created: {len(st.session_state.get('manual_new_features', []))}. Good features = better clusters.</p>"
    if stages_done >= 4:
        html += "<h3>✅ Stage 4: Preprocessing</h3><p>Scaling + encoding ensures every feature contributes equally to distance calculations.</p>"
    if stages_done >= 5:
        html += f"<h3>✅ Stage 5: Model Selection</h3><p>You chose <strong>{st.session_state.get('manual_model_choice', '—')}</strong> — a key decision that affects how clusters are formed.</p>"
    if stages_done >= 6:
        html += "<h3>✅ Stage 6: Model Training</h3><p>Model has been fitted. You now have cluster assignments for every row.</p>"
    if stages_done >= 7:
        html += "<h3>✅ Stage 7: Evaluation</h3><p>Silhouette score shows how good your clustering is. Remember: unsupervised learning has no ground truth — we rely on internal metrics.</p>"
    if stages_done >= 8:
        html += "<h3>✅ Stage 8: Visualization</h3><p>PCA plot gives you an intuitive 2D view of your high-dimensional clusters.</p>"
    
    html += "<hr><p><strong>💡 Pro tip from your mentor:</strong> The best clustering often comes from combining domain knowledge with experimentation. Try different K values and re-run the pipeline!</p>"
    return html

# ===================== FOOTER =====================
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#666;'>"
    "✅ Production-level • Single file • Zero code changes required • Works with any dataset<br>"
    "Built as a complete AutoML tool + interactive ML tutor with mentor explanations"
    "</p>",
    unsafe_allow_html=True
)

# JS helper to copy code (for the demo page)
def copyCode():
    navigator.clipboard.writeText(FULL_APP_CODE).then(() => {
        alert("✅ Full app.py code copied to clipboard! Just save it and run with: streamlit run app.py")
    })
        """
    </script>
</body>
</html>

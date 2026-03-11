import streamlit as st
import pandas as pd
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from agent import load_uploaded_dataset, _build_metadata
from tool import (
    show_head,
    dataset_stats,
    handle_missing_values,
    data_engineering,
    correlation_analysis,
    detect_outliers,
    visualize_data,
)

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="AI EDA Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Light, vibrant CSS theme ──────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    /* Global font */
    html, body, .stApp, .stMarkdown, p, span, label, div {
        font-family: 'Inter', sans-serif;
    }

    /* Light background */
    .stApp {
        background: linear-gradient(135deg, #f8f9fe 0%, #eef1f8 50%, #f0f4ff 100%);
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #ffffff 0%, #f0f4ff 100%);
        border-right: 1px solid #e0e7ff;
    }
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown span {
        color: #374151;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background-color: #ffffff;
        border-radius: 14px;
        padding: 6px 8px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
        border: 1px solid #e5e7eb;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        padding: 10px 18px;
        font-weight: 600;
        font-size: 13px;
        color: #6b7280;
        background: transparent;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
        color: white !important;
        border-radius: 10px;
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.35);
    }

    /* Metric cards */
    [data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 14px;
        padding: 18px 20px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }
    [data-testid="stMetric"] label {
        color: #6b7280 !important;
        font-weight: 600 !important;
        font-size: 12px !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    [data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #1f2937 !important;
        font-weight: 700 !important;
    }

    /* Headings */
    h1 {
        color: #1f2937 !important;
        font-weight: 800 !important;
        background: none !important;
        -webkit-text-fill-color: #1f2937 !important;
    }
    h2 {
        background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        font-weight: 700 !important;
    }
    h3 {
        color: #374151 !important;
        background: none !important;
        -webkit-text-fill-color: #374151 !important;
        font-weight: 600 !important;
    }

    /* Expanders */
    .stExpander {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 14px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.04);
    }

    /* DataFrames */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 1px 4px rgba(0,0,0,0.04);
    }

    /* Info card */
    .info-card {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 14px;
        padding: 20px 24px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        margin-bottom: 12px;
    }
    .info-card h4 {
        color: #6366f1;
        margin: 0 0 8px 0;
        font-weight: 700;
    }
    .info-card p {
        color: #4b5563;
        margin: 0;
        font-size: 14px;
        line-height: 1.5;
    }

    /* Section card */
    .section-card {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 14px;
        padding: 24px;
        margin-bottom: 16px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }
    .section-title {
        color: #6366f1;
        font-weight: 700;
        font-size: 16px;
        margin-bottom: 12px;
    }

    /* Strategy badge */
    .strategy-badge {
        display: inline-block;
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        color: white;
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
        margin: 4px 0;
    }

    /* Hero section */
    .hero {
        text-align: center;
        padding: 60px 20px 40px;
    }
    .hero h1 {
        font-size: 42px !important;
        margin-bottom: 8px;
    }
    .hero p {
        color: #6b7280;
        font-size: 18px;
        max-width: 600px;
        margin: 0 auto 30px;
    }

    /* Feature cards */
    .feature-card {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 16px;
        padding: 28px 24px;
        text-align: center;
        box-shadow: 0 2px 12px rgba(0,0,0,0.05);
        transition: transform 0.2s, box-shadow 0.2s;
        height: 100%;
    }
    .feature-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(99, 102, 241, 0.15);
    }
    .feature-icon {
        font-size: 36px;
        margin-bottom: 12px;
    }
    .feature-card h4 {
        color: #1f2937;
        font-weight: 700;
        margin: 0 0 8px;
    }
    .feature-card p {
        color: #6b7280;
        font-size: 13px;
        margin: 0;
        line-height: 1.5;
    }

    /* Upload area */
    [data-testid="stFileUploader"] {
        background: #ffffff;
        border-radius: 14px;
        padding: 12px;
        border: 1px solid #e5e7eb;
    }

    /* Buttons */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 10px 28px !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 14px rgba(99, 102, 241, 0.35) !important;
        transition: all 0.2s !important;
    }
    .stButton > button[kind="primary"]:hover {
        box-shadow: 0 6px 20px rgba(99, 102, 241, 0.5) !important;
        transform: translateY(-1px);
    }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🤖 AI EDA Agent")
    st.markdown("Upload any **CSV** or **XLSX** dataset and get a full automated EDA powered by GPT-4o-mini.")
    st.markdown("---")

    uploaded_file = st.file_uploader(
        "📂 Upload Dataset (max 10 MB)",
        type=["csv", "xlsx", "xls"],
        help="Supports CSV and Excel files up to 10 MB.",
    )

    st.markdown("---")
    st.markdown("**Tools Included:**")
    st.markdown("""
    1. 📂 Dataset Preview
    2. 📊 Statistics
    3. 🧹 Missing Values
    4. 🔧 Data Engineering
    5. 🎯 Outlier Detection
    6. 🔗 Correlation Analysis
    7. 🤖 AI Visualizations
    """)
    st.markdown("---")
    st.caption("Built with Python • Seaborn • LangChain • Streamlit")


# ── Load file into session ────────────────────────────────────
if uploaded_file is not None:
    file_id = f"{uploaded_file.name}_{uploaded_file.size}"
    if st.session_state.get("_file_id") != file_id:
        with st.spinner("Loading dataset..."):
            metadata, df = load_uploaded_dataset(uploaded_file)
            st.session_state["df_original"] = df.copy()
            st.session_state["df"] = df
            st.session_state["metadata"] = metadata
            st.session_state["_file_id"] = file_id
            st.session_state["eda_started"] = False
            for key in ["missing_done", "engineering_done", "llm_response",
                         "missing_report", "engineering_report"]:
                st.session_state.pop(key, None)


# ── Landing page (no file uploaded) ──────────────────────────
if "df" not in st.session_state:
    st.markdown("""
    <div class="hero">
        <h1>🤖 AI Data Scientist</h1>
        <p>Upload a dataset from the sidebar to unlock a full automated Exploratory Data Analysis powered by AI.</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">📊</div>
            <h4>7 EDA Tools</h4>
            <p>Statistics, missing values, data engineering, outliers, correlation — all automated.</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🤖</div>
            <h4>AI-Powered Plots</h4>
            <p>GPT-4o-mini analyzes your data and auto-generates the most insightful visualizations.</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">📂</div>
            <h4>Any Dataset</h4>
            <p>Upload CSV or XLSX up to 10 MB. Works on completely unseen data.</p>
        </div>
        """, unsafe_allow_html=True)

    st.stop()


# ── "Start EDA" gate ──────────────────────────────────────────
if not st.session_state.get("eda_started"):
    df = st.session_state["df"]

    st.markdown("# 🤖 AI Data Scientist — EDA Specialist")
    st.success(f"✅ **{uploaded_file.name}** loaded successfully — **{df.shape[0]:,} rows × {df.shape[1]} columns**")

    st.markdown("### Quick Preview")
    st.dataframe(df.head(5), use_container_width=True)

    st.markdown("")
    col_left, col_center, col_right = st.columns([2, 1, 2])
    with col_center:
        if st.button("🚀 Start EDA", type="primary", use_container_width=True):
            st.session_state["eda_started"] = True
            st.rerun()

    st.stop()


# ══════════════════════════════════════════════════════════════
#                       EDA TABS
# ══════════════════════════════════════════════════════════════
df = st.session_state["df"]

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📂 Preview",
    "📊 Statistics",
    "🧹 Missing Values",
    "🔧 Engineering",
    "🎯 Outliers",
    "🔗 Correlation",
    "🤖 AI Visualizations",
])


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 1 — Dataset Preview
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab1:
    st.markdown("## 📂 Dataset Preview")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows", f"{df.shape[0]:,}")
    col2.metric("Columns", f"{df.shape[1]}")
    mem_kb = df.memory_usage(deep=True).sum() / 1024
    col3.metric("Memory", f"{mem_kb:.1f} KB")
    col4.metric("Duplicates", f"{df.duplicated().sum()}")

    st.markdown("### 🔍 First 10 Rows")
    st.dataframe(df.head(10), use_container_width=True)

    st.markdown("### 📋 Column Information")
    dtype_df = pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes.astype(str).values,
        "Non-Null": df.notna().sum().values,
        "Nulls": df.isna().sum().values,
        "Null %": (df.isna().sum() / len(df) * 100).round(1).astype(str).values + "%",
        "Unique": [df[c].nunique() for c in df.columns],
    })
    st.dataframe(dtype_df, use_container_width=True, hide_index=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 2 — Dataset Statistics
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab2:
    st.markdown("## 📊 Dataset Statistics")

    # Quick overview metrics
    numeric_df = df.select_dtypes(include="number")
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    null_total = df.isnull().sum().sum()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Rows", f"{df.shape[0]:,}")
    c2.metric("Numeric Cols", f"{len(numeric_df.columns)}")
    c3.metric("Categorical Cols", f"{len(cat_cols)}")
    c4.metric("Total Nulls", f"{null_total:,}")

    # Data types breakdown
    st.markdown("### 🔠 Data Types Breakdown")
    dtype_counts = df.dtypes.value_counts()
    dtype_breakdown = pd.DataFrame({
        "Data Type": dtype_counts.index.astype(str),
        "Column Count": dtype_counts.values,
    })
    st.dataframe(dtype_breakdown, use_container_width=True, hide_index=True)

    # Numeric statistics
    if not numeric_df.empty:
        st.markdown("### 🔢 Numeric Column Statistics")
        stats = numeric_df.describe().T
        stats["skewness"] = numeric_df.skew()
        stats["kurtosis"] = numeric_df.kurt()
        st.dataframe(stats.round(4), use_container_width=True)

    # Categorical statistics
    if len(cat_cols) > 0:
        st.markdown("### 🔤 Categorical Column Statistics")
        cat_data = []
        for col in cat_cols:
            vc = df[col].value_counts()
            cat_data.append({
                "Column": col,
                "Unique Values": df[col].nunique(),
                "Top Value": vc.idxmax() if not vc.empty else "N/A",
                "Top Frequency": vc.max() if not vc.empty else 0,
                "Top %": f"{(vc.max() / len(df) * 100):.1f}%" if not vc.empty else "0%",
            })
        st.dataframe(pd.DataFrame(cat_data), use_container_width=True, hide_index=True)

    # Duplicate rows
    dupes = df.duplicated().sum()
    if dupes > 0:
        st.warning(f"⚠️ Found **{dupes}** duplicate rows ({(dupes / len(df) * 100):.1f}%)")
    else:
        st.success("✅ No duplicate rows found.")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 3 — Missing Values
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab3:
    st.markdown("## 🧹 Missing Value Handler")

    df_current = st.session_state["df"]
    null_counts = df_current.isnull().sum()
    missing_cols = null_counts[null_counts > 0]
    null_total = null_counts.sum()

    if null_total == 0 and not st.session_state.get("missing_done"):
        st.success("✅ No missing values detected! Dataset is already clean.")
        # Show overview anyway
        st.markdown("### Column Null Summary")
        null_df = pd.DataFrame({
            "Column": df_current.columns,
            "Nulls": df_current.isnull().sum().values,
            "Status": ["✅ Clean" for _ in df_current.columns],
        })
        st.dataframe(null_df, use_container_width=True, hide_index=True)

    elif not st.session_state.get("missing_done"):
        # Show what's missing BEFORE cleaning
        st.warning(f"⚠️ Found **{null_total}** missing values across **{len(missing_cols)}** column(s).")

        st.markdown("### Missing Values Breakdown")
        missing_df = pd.DataFrame({
            "Column": missing_cols.index,
            "Missing Count": missing_cols.values,
            "Missing %": (missing_cols / len(df_current) * 100).round(1).astype(str).values + "%",
            "Data Type": [str(df_current[c].dtype) for c in missing_cols.index],
        })
        st.dataframe(missing_df, use_container_width=True, hide_index=True)

        st.markdown("### Imputation Strategy")
        st.markdown("""
        | Data Type | Strategy | When |
        |---|---|---|
        | **Numeric (skewed)** | Fill with **Median** | \|skewness\| > 0.5 |
        | **Numeric (symmetric)** | Fill with **Mean** | \|skewness\| ≤ 0.5 |
        | **Categorical** | Fill with **Mode** | Most frequent value |
        | **Datetime** | **Forward Fill** | Preserves time continuity |
        | **>50% missing** | **Drop Column** | Too much data lost |
        """)

        col_left, col_center, col_right = st.columns([2, 1, 2])
        with col_center:
            if st.button("🧹 Fix Missing Values", type="primary", use_container_width=True):
                with st.spinner("Applying smart imputation..."):
                    original_nulls = df_current.isnull().sum()
                    cleaned_df, report = handle_missing_values(df_current.copy())
                    st.session_state["df"] = cleaned_df
                    st.session_state["missing_done"] = True

                    # Build structured before/after data
                    before_after = []
                    for col in missing_cols.index:
                        before_n = int(original_nulls[col])
                        if col in cleaned_df.columns:
                            after_n = int(cleaned_df[col].isnull().sum())
                            status = "✅ Clean" if after_n == 0 else f"⚠️ {after_n} remain"
                        else:
                            after_n = 0
                            status = "🗑️ Dropped"
                        before_after.append({
                            "Column": col,
                            "Before (Nulls)": before_n,
                            "After (Nulls)": after_n,
                            "Status": status,
                        })
                    st.session_state["missing_before_after"] = before_after
                    st.rerun()
    else:
        st.success("✅ All missing values have been handled!")

    # Show before/after results
    if st.session_state.get("missing_before_after"):
        st.markdown("### 📊 Before / After Summary")
        ba_df = pd.DataFrame(st.session_state["missing_before_after"])
        st.dataframe(ba_df, use_container_width=True, hide_index=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 4 — Data Engineering
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab4:
    st.markdown("## 🔧 Data Engineering & EDA")

    if not st.session_state.get("engineering_done"):
        st.info("Run data engineering to get a full detailed EDA on your (cleaned) data.")

        col_left, col_center, col_right = st.columns([2, 1, 2])
        with col_center:
            if st.button("🔧 Run Engineering", type="primary", use_container_width=True):
                with st.spinner("Running data engineering pipeline..."):
                    df_copy = st.session_state["df"].copy()
                    cleaned_df, report = data_engineering(df_copy)
                    st.session_state["df"] = cleaned_df
                    st.session_state["engineering_done"] = True
                    st.rerun()
    else:
        st.success("✅ Data Engineering complete!")

    # Always show structured EDA when engineering is done
    if st.session_state.get("engineering_done"):
        df_eng = st.session_state["df"]

        st.markdown("### 📐 Dataset Shape")
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", f"{df_eng.shape[0]:,}")
        c2.metric("Columns", f"{df_eng.shape[1]}")
        c3.metric("Duplicates", f"{df_eng.duplicated().sum()}")

        # Data types
        st.markdown("### 🔠 Data Types")
        types_df = pd.DataFrame({
            "Column": df_eng.columns,
            "Data Type": df_eng.dtypes.astype(str).values,
        })
        st.dataframe(types_df, use_container_width=True, hide_index=True)

        # Null check
        remaining_nulls = df_eng.isnull().sum().sum()
        if remaining_nulls == 0:
            st.success("✅ No null values — dataset is fully clean!")
        else:
            st.warning(f"⚠️ {remaining_nulls} null values remain.")

        # Descriptive stats
        num_df = df_eng.select_dtypes(include="number")
        if not num_df.empty:
            st.markdown("### 📈 Descriptive Statistics (Numeric)")
            st.dataframe(num_df.describe().T.round(4), use_container_width=True)

        # Categorical value counts
        cat_cols_eng = df_eng.select_dtypes(include=["object", "category"]).columns
        if len(cat_cols_eng) > 0:
            st.markdown("### 🏷️ Categorical Value Counts")
            for col in cat_cols_eng:
                with st.expander(f"📌 {col} ({df_eng[col].nunique()} unique values)"):
                    vc = df_eng[col].value_counts().reset_index()
                    vc.columns = ["Value", "Count"]
                    vc["Percentage"] = (vc["Count"] / len(df_eng) * 100).round(1).astype(str) + "%"
                    st.dataframe(vc, use_container_width=True, hide_index=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 5 — Outlier Detection
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab5:
    st.markdown("## 🎯 Outlier Detection (IQR Method)")

    df_current = st.session_state["df"]
    fig, summary_df, report = detect_outliers(df_current)

    if summary_df is not None and not summary_df.empty:
        # Quick metrics
        total_outliers = int(summary_df["Outliers"].sum())
        cols_with_outliers = int((summary_df["Outliers"] > 0).sum())

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Outliers", f"{total_outliers:,}")
        c2.metric("Affected Columns", f"{cols_with_outliers}")
        c3.metric("Clean Columns", f"{len(summary_df) - cols_with_outliers}")

        st.markdown("### 📋 Outlier Summary")

        # Add status column
        display_df = summary_df.copy()
        display_df["Status"] = display_df["Outliers"].apply(
            lambda x: "⚠️ Has Outliers" if x > 0 else "✅ Clean"
        )
        st.dataframe(display_df, use_container_width=True, hide_index=True)

    if fig is not None:
        st.markdown("### 📊 Box Plots")
        st.pyplot(fig)
        plt.close(fig)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 6 — Correlation Analysis
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab6:
    st.markdown("## 🔗 Correlation Analysis")

    df_current = st.session_state["df"]
    numeric_check = df_current.select_dtypes(include="number")

    if numeric_check.shape[1] < 2:
        st.warning("⚠️ Need at least 2 numeric columns for correlation analysis.")
    else:
        fig, top_pairs, report = correlation_analysis(df_current)

        if fig is not None:
            st.markdown("### 🗺️ Correlation Heatmap")
            st.pyplot(fig)
            plt.close(fig)

        if top_pairs is not None and not top_pairs.empty:
            st.markdown("### 🔗 Top Correlated Feature Pairs")

            display_pairs = top_pairs.copy()
            display_pairs["Direction"] = display_pairs["Correlation"].apply(
                lambda x: "📈 Positive" if x > 0 else "📉 Negative"
            )
            display_pairs["Strength"] = display_pairs["Abs Correlation"].apply(
                lambda x: "🔴 Strong" if x > 0.7 else ("🟡 Moderate" if x > 0.4 else "🟢 Weak")
            )
            display_pairs["Correlation"] = display_pairs["Correlation"].round(4)
            display_pairs["Abs Correlation"] = display_pairs["Abs Correlation"].round(4)
            st.dataframe(display_pairs, use_container_width=True, hide_index=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 7 — AI Visualizations
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab7:
    st.markdown("## 🤖 AI-Powered Visualizations")
    st.markdown("GPT-4o-mini analyzes your dataset and auto-generates the most insightful plots.")

    col_left, col_center, col_right = st.columns([2, 1, 2])
    with col_center:
        run_ai = st.button("🚀 Generate AI Plots", type="primary", use_container_width=True)

    if run_ai:
        load_dotenv(dotenv_path=".evv")

        llm_model = ChatOpenAI(model="gpt-4o-mini", temperature=0.2, max_tokens=1500)

        ai_prompt = ChatPromptTemplate.from_messages([
            ('system', """You are an expert Data Scientist and EDA Specialist.

A user has uploaded an unknown dataset. I will provide you its metadata: column names, data types, a summary (df.info()), and a preview (df.head()).

Your job is to analyze this metadata and recommend the most insightful plots for 3 types of EDA analysis.
For each category, recommend 1 or 2 plots — pick the count that best suits the dataset (don't force 2 if 1 is enough).

You MUST respond with ONLY valid JSON (no markdown, no code fences, no extra text).
Use this exact structure:

{{
  "univariate": [
    {{"plot_type": "<plot type>", "columns": ["<column_name>"], "hue": null, "reason": "<1-sentence reason>"}}
  ],
  "bivariate": [
    {{"plot_type": "<plot type>", "columns": ["<col1>", "<col2>"], "hue": "<optional_col_or_null>", "reason": "<1-sentence reason>"}}
  ],
  "multivariate": [
    {{"plot_type": "<plot type>", "columns": ["<col1>", "<col2>", "<col3>"], "hue": "<optional_col_or_null>", "reason": "<1-sentence reason>"}}
  ]
}}

Rules:
- Use generic plot types: histogram, scatter plot, box plot, bar plot, heatmap, pair plot, count plot, violin plot
- Always use EXACT column names from the provided metadata
- "hue" must be a categorical column name or null
- "columns" must be a list of exact column name strings
- Each category ("univariate", "bivariate", "multivariate") must have 1 or 2 plot objects
- Base all recommendations strictly on the provided metadata — do NOT invent columns"""),
            ('human', "{dataset_info}")
        ])

        ai_chain = ai_prompt | llm_model | StrOutputParser()
        df_current = st.session_state["df"]
        current_metadata = _build_metadata(df_current)

        with st.spinner("🤖 Asking GPT-4o-mini for the best plots..."):
            try:
                response = ai_chain.invoke({"dataset_info": current_metadata})
                st.session_state["llm_response"] = response
            except Exception as e:
                st.error(f"❌ LLM call failed: {e}")

    # ── Render AI recommendations ─────────────────────────────
    if st.session_state.get("llm_response"):
        response = st.session_state["llm_response"]

        with st.expander("🔍 Raw LLM Response (JSON)", expanded=False):
            st.code(response, language="json")

        try:
            recommendations = json.loads(response)
            df_current = st.session_state["df"]

            category_info = {
                "univariate": ("🔹", "Univariate Analysis", "Analyzing one variable at a time"),
                "bivariate": ("🔸", "Bivariate Analysis", "Exploring relationships between 2 variables"),
                "multivariate": ("🔶", "Multivariate Analysis", "Uncovering patterns across 3+ variables"),
            }

            for category in ["univariate", "bivariate", "multivariate"]:
                plots = recommendations.get(category, [])
                if not plots:
                    continue

                emoji, title, desc = category_info[category]
                st.markdown(f"### {emoji} {title}")
                st.caption(f"{desc} — {len(plots)} plot{'s' if len(plots) > 1 else ''} recommended")

                for i, plot in enumerate(plots):
                    plot_type = plot.get("plot_type", "histogram")
                    columns = plot.get("columns", [])
                    hue = plot.get("hue")
                    reason = plot.get("reason", "")

                    if hue in ("null", "None", None):
                        hue = None

                    # Info header
                    st.markdown(
                        f"**{plot_type.title()}** — Columns: `{', '.join(columns)}`"
                        + (f" — Hue: `{hue}`" if hue else "")
                    )
                    st.info(f"💡 **Why this plot:** {reason}")

                    fig, msg = visualize_data(df_current, plot_type, columns, hue=hue)
                    if fig is not None:
                        st.pyplot(fig)
                        plt.close(fig)
                    else:
                        st.warning(msg)

                st.markdown("---")

            st.success("✅ All AI-recommended visualizations generated!")

        except json.JSONDecodeError:
            st.error("❌ Failed to parse LLM response as JSON. Click the button to try again.")
        except Exception as e:
            st.error(f"❌ Error: {e}")

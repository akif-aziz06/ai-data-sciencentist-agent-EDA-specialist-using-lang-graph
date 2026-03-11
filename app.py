import streamlit as st
import pandas as pd
import re
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dotenv import load_dotenv
from agent import load_uploaded_dataset, _build_metadata
from graph import build_eda_graph, run_eda, resume_eda, get_store, extract_options

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

    html, body, .stApp, .stMarkdown, p, span, label, div {
        font-family: 'Inter', sans-serif;
    }
    .stApp {
        background: linear-gradient(135deg, #f8f9fe 0%, #eef1f8 50%, #f0f4ff 100%);
    }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #ffffff 0%, #f0f4ff 100%);
        border-right: 1px solid #e0e7ff;
    }
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown span {
        color: #374151;
    }
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
    h1 { color: #1f2937 !important; font-weight: 800 !important; background: none !important; -webkit-text-fill-color: #1f2937 !important; }
    h2 { background: linear-gradient(135deg, #6366f1, #8b5cf6) !important; -webkit-background-clip: text !important; -webkit-text-fill-color: transparent !important; font-weight: 700 !important; }
    h3 { color: #374151 !important; background: none !important; -webkit-text-fill-color: #374151 !important; font-weight: 600 !important; }
    .stExpander { background: #ffffff; border: 1px solid #e5e7eb; border-radius: 14px; box-shadow: 0 1px 4px rgba(0,0,0,0.04); }
    .stDataFrame { border-radius: 12px; overflow: hidden; box-shadow: 0 1px 4px rgba(0,0,0,0.04); }
    .hero { text-align: center; padding: 60px 20px 40px; }
    .hero h1 { font-size: 42px !important; margin-bottom: 8px; }
    .hero p { color: #6b7280; font-size: 18px; max-width: 600px; margin: 0 auto 30px; }
    .feature-card { background: #ffffff; border: 1px solid #e5e7eb; border-radius: 16px; padding: 28px 24px; text-align: center; box-shadow: 0 2px 12px rgba(0,0,0,0.05); transition: transform 0.2s, box-shadow 0.2s; height: 100%; }
    .feature-card:hover { transform: translateY(-2px); box-shadow: 0 8px 24px rgba(99, 102, 241, 0.15); }
    .feature-icon { font-size: 36px; margin-bottom: 12px; }
    .feature-card h4 { color: #1f2937; font-weight: 700; margin: 0 0 8px; }
    .feature-card p { color: #6b7280; font-size: 13px; margin: 0; line-height: 1.5; }
    [data-testid="stFileUploader"] { background: #ffffff; border-radius: 14px; padding: 12px; border: 1px solid #e5e7eb; }
    .stButton > button[kind="primary"] { background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important; color: white !important; border: none !important; border-radius: 10px !important; padding: 10px 28px !important; font-weight: 600 !important; box-shadow: 0 4px 14px rgba(99, 102, 241, 0.35) !important; }
    .stButton > button[kind="primary"]:hover { box-shadow: 0 6px 20px rgba(99, 102, 241, 0.5) !important; transform: translateY(-1px); }
    .hitl-card { background: #fffbeb; border: 2px solid #f59e0b; border-radius: 14px; padding: 24px; margin: 16px 0; }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🤖 AI EDA Agent")
    st.markdown("Upload any **CSV** or **XLSX** dataset and get a full automated EDA powered by a **LangGraph Agent** with GPT-4o-mini.")
    st.markdown("---")

    uploaded_file = st.file_uploader(
        "📂 Upload Dataset (max 10 MB)",
        type=["csv", "xlsx", "xls"],
        help="Supports CSV and Excel files up to 10 MB.",
    )

    st.markdown("---")
    st.markdown("**Agent Pipeline:**")
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
    st.markdown("**Powered by:**")
    st.caption("LangGraph • @tool • Human-in-the-Loop • GPT-4o-mini")


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
            # Reset EDA state
            for key in ["eda_started", "eda_complete", "eda_interrupted",
                        "eda_store", "eda_messages", "interrupt_data", "eda_graph"]:
                st.session_state.pop(key, None)


# ── Landing page (no file uploaded) ──────────────────────────
if "df" not in st.session_state:
    st.markdown("""
    <div class="hero">
        <h1>🤖 AI Data Scientist</h1>
        <p>Upload a dataset from the sidebar to unlock a full automated EDA powered by a LangGraph Agent.</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🔗</div>
            <h4>LangGraph Agent</h4>
            <p>A stateful graph pipeline that orchestrates the entire EDA — each tool is an @tool node called by the LLM.</p>
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
            <div class="feature-icon">🧑‍💻</div>
            <h4>Human-in-the-Loop</h4>
            <p>The agent asks YOU when it encounters ambiguous decisions — you stay in control.</p>
        </div>
        """, unsafe_allow_html=True)

    st.stop()


# ── Initialize graph ──────────────────────────────────────────
if "eda_graph" not in st.session_state:
    st.session_state["eda_graph"] = build_eda_graph()


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
#  Run Graph / Handle HITL
# ══════════════════════════════════════════════════════════════

# Run the graph if EDA started but not complete and not interrupted
if (st.session_state.get("eda_started")
    and not st.session_state.get("eda_complete")
    and not st.session_state.get("eda_interrupted")):

    with st.spinner("🤖 AI Agent is running the EDA pipeline... This may take a minute."):
        try:
            result, store, interrupted, interrupt_data = run_eda(
                st.session_state["df"],
                st.session_state["metadata"],
                st.session_state["eda_graph"],
            )
            st.session_state["eda_store"] = store
            st.session_state["eda_messages"] = result.get("messages", [])

            if interrupted:
                st.session_state["eda_interrupted"] = True
                st.session_state["interrupt_data"] = interrupt_data
            else:
                st.session_state["eda_complete"] = True
                if store.get("df") is not None:
                    st.session_state["df"] = store["df"]
            st.rerun()
        except Exception as e:
            st.error(f"❌ Error running EDA pipeline: {e}")
            import traceback
            st.code(traceback.format_exc())
            st.stop()


# ── Human-in-the-Loop UI ─────────────────────────────────────
if st.session_state.get("eda_interrupted"):
    st.markdown("# 🤖 AI Data Scientist — EDA Specialist")

    interrupt_data = st.session_state.get("interrupt_data", {})
    question = interrupt_data.get("question", "The agent needs your input.") if interrupt_data else "The agent needs your input."

    st.warning("🤖 **The AI Agent needs your input to continue:**")
    st.markdown(f"{question}")

    options = extract_options(question)

    if len(options) > 1:
        choice = st.radio("Select an option:", options, key="hitl_radio")
    else:
        choice = st.text_input("Your response:", key="hitl_text")

    col_left, col_center, col_right = st.columns([2, 1, 2])
    with col_center:
        if st.button("Continue ▶", type="primary", use_container_width=True):
            with st.spinner("🤖 Resuming analysis..."):
                try:
                    result, store, interrupted, interrupt_data = resume_eda(
                        choice,
                        st.session_state["eda_graph"],
                    )
                    st.session_state["eda_store"] = store

                    if interrupted:
                        st.session_state["interrupt_data"] = interrupt_data
                    else:
                        st.session_state["eda_interrupted"] = False
                        st.session_state["eda_complete"] = True
                        if store.get("df") is not None:
                            st.session_state["df"] = store["df"]
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error: {e}")
    st.stop()


# ══════════════════════════════════════════════════════════════
#  EDA RESULTS — TABS
# ══════════════════════════════════════════════════════════════

if not st.session_state.get("eda_complete"):
    st.stop()

st.markdown("# 🤖 AI Data Scientist — EDA Specialist")
st.success("✅ EDA pipeline complete! Explore your results below.")

store = st.session_state.get("eda_store", {})
df = st.session_state["df"]
results = store.get("results", {})
figures = store.get("figures", {})
viz_figs = store.get("viz_figures", [])

# Show agent conversation in an expander
messages = st.session_state.get("eda_messages", [])
if messages:
    with st.expander("💬 Agent Conversation Log", expanded=False):
        for msg in messages:
            role = getattr(msg, "type", "unknown")
            content = getattr(msg, "content", "")
            if role == "ai" and content:
                st.markdown(f"🤖 **Agent:** {content[:500]}")
            elif role == "human" and content and "Run the complete" not in content:
                st.markdown(f"👤 **You:** {content[:300]}")
            elif role == "tool" and content:
                st.markdown(f"🔧 **Tool Result:** `{content[:200]}...`")


tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📂 Preview",
    "📊 Statistics",
    "🧹 Missing Values",
    "🔧 Engineering",
    "🎯 Outliers",
    "🔗 Correlation",
    "🤖 AI Visualizations",
])


# ━━ TAB 1 — Preview ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
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
        "Unique": [df[c].nunique() for c in df.columns],
    })
    st.dataframe(dtype_df, use_container_width=True, hide_index=True)


# ━━ TAB 2 — Statistics ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab2:
    st.markdown("## 📊 Dataset Statistics")

    numeric_df = df.select_dtypes(include="number")
    cat_cols = df.select_dtypes(include=["object", "category"]).columns

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Rows", f"{df.shape[0]:,}")
    c2.metric("Numeric Cols", f"{len(numeric_df.columns)}")
    c3.metric("Categorical Cols", f"{len(cat_cols)}")
    c4.metric("Total Nulls", f"{df.isnull().sum().sum():,}")

    if not numeric_df.empty:
        st.markdown("### 🔢 Numeric Column Statistics")
        stats = numeric_df.describe().T
        stats["skewness"] = numeric_df.skew()
        stats["kurtosis"] = numeric_df.kurt()
        st.dataframe(stats.round(4), use_container_width=True)

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
            })
        st.dataframe(pd.DataFrame(cat_data), use_container_width=True, hide_index=True)

    dupes = df.duplicated().sum()
    if dupes > 0:
        st.warning(f"⚠️ Found **{dupes}** duplicate rows ({(dupes / len(df) * 100):.1f}%)")
    else:
        st.success("✅ No duplicate rows found.")


# ━━ TAB 3 — Missing Values ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab3:
    st.markdown("## 🧹 Missing Value Handler")

    missing_data = results.get("missing")
    if missing_data and missing_data.get("column_details"):
        st.success("✅ Missing values handled by the AI Agent automatically!")

        # Quick metrics
        total_cols = missing_data.get("total_missing_cols", 0)
        total_vals = missing_data.get("total_missing_values", 0)
        c1, c2, c3 = st.columns(3)
        c1.metric("Columns Affected", f"{total_cols}")
        c2.metric("Total Missing Values", f"{total_vals:,}")
        c3.metric("Status", "All Fixed ✅")

        # Strategy table
        st.markdown("### 🔧 Imputation Strategy Applied")
        col_details = missing_data.get("column_details", [])
        if col_details:
            st.dataframe(
                pd.DataFrame(col_details),
                use_container_width=True,
                hide_index=True,
            )

        # Per-column detail cards
        st.markdown("### 📌 Column Details")
        for detail in col_details:
            with st.expander(f"📌 {detail['Column']} — {detail['Missing']} missing ({detail['Missing %']})"):
                dc1, dc2 = st.columns(2)
                dc1.markdown(f"**Data Type:** `{detail['Data Type']}`")
                dc2.markdown(f"**Missing:** {detail['Missing']} / {len(df)} rows")
                st.markdown(f"**Strategy:** `{detail['Strategy']}`")
                st.info(f"💡 **Why:** {detail['Reason']}")

        # Before / After summary
        before_after = missing_data.get("before_after", [])
        if before_after:
            st.markdown("### 📊 Before / After Summary")
            st.dataframe(
                pd.DataFrame(before_after),
                use_container_width=True,
                hide_index=True,
            )

    elif missing_data:
        # Fallback: no structured data, show report text
        st.success("✅ Missing values handled by the AI Agent automatically!")
        st.text(missing_data.get("report", "No report available."))
    else:
        null_total = df.isnull().sum().sum()
        if null_total == 0:
            st.success("✅ No missing values — dataset is clean!")
        else:
            st.info(f"Found {null_total} missing values.")



# ━━ TAB 4 — Data Engineering ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab4:
    st.markdown("## 🔧 Data Engineering & EDA")

    eng_data = results.get("engineering")
    if eng_data:
        st.success("✅ Data Engineering complete!")
        df_eng = eng_data.get("cleaned_df", df)

        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", f"{df_eng.shape[0]:,}")
        c2.metric("Columns", f"{df_eng.shape[1]}")
        c3.metric("Duplicates", f"{df_eng.duplicated().sum()}")

        num_df = df_eng.select_dtypes(include="number")
        if not num_df.empty:
            st.markdown("### 📈 Descriptive Statistics")
            st.dataframe(num_df.describe().T.round(4), use_container_width=True)

        cat_cols_eng = df_eng.select_dtypes(include=["object", "category"]).columns
        if len(cat_cols_eng) > 0:
            st.markdown("### 🏷️ Categorical Value Counts")
            for col in cat_cols_eng:
                with st.expander(f"📌 {col} ({df_eng[col].nunique()} unique values)"):
                    vc = df_eng[col].value_counts().reset_index()
                    vc.columns = ["Value", "Count"]
                    st.dataframe(vc, use_container_width=True, hide_index=True)
    else:
        st.info("Data engineering results not available.")


# ━━ TAB 5 — Outlier Detection ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab5:
    st.markdown("## 🎯 Outlier Detection (IQR Method)")

    outlier_data = results.get("outliers")
    if outlier_data:
        summary_df = outlier_data.get("summary_df")
        if summary_df is not None and not summary_df.empty:
            total_outliers = int(summary_df["Outliers"].sum())
            cols_with = int((summary_df["Outliers"] > 0).sum())

            c1, c2, c3 = st.columns(3)
            c1.metric("Total Outliers", f"{total_outliers:,}")
            c2.metric("Affected Columns", f"{cols_with}")
            c3.metric("Clean Columns", f"{len(summary_df) - cols_with}")

            st.markdown("### 📋 Outlier Summary")
            display_df = summary_df.copy()
            display_df["Status"] = display_df["Outliers"].apply(
                lambda x: "⚠️ Has Outliers" if x > 0 else "✅ Clean"
            )
            st.dataframe(display_df, use_container_width=True, hide_index=True)

    fig_outlier = figures.get("outliers")
    if fig_outlier is not None:
        st.markdown("### 📊 Box Plots")
        st.pyplot(fig_outlier)
        plt.close(fig_outlier)


# ━━ TAB 6 — Correlation ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab6:
    st.markdown("## 🔗 Correlation Analysis")

    fig_corr = figures.get("correlation")
    if fig_corr is not None:
        st.markdown("### 🗺️ Correlation Heatmap")
        st.pyplot(fig_corr)
        plt.close(fig_corr)

    corr_data = results.get("correlation")
    if corr_data:
        top_pairs = corr_data.get("top_pairs")
        if top_pairs is not None and not top_pairs.empty:
            st.markdown("### 🔗 Top Correlated Feature Pairs")
            display_pairs = top_pairs.copy()
            display_pairs["Direction"] = display_pairs["Correlation"].apply(
                lambda x: "📈 Positive" if x > 0 else "📉 Negative"
            )
            display_pairs["Strength"] = display_pairs["Abs Correlation"].apply(
                lambda x: "🔴 Strong" if x > 0.7 else ("🟡 Moderate" if x > 0.4 else "🟢 Weak")
            )
            st.dataframe(display_pairs.round(4), use_container_width=True, hide_index=True)
    else:
        numeric_check = df.select_dtypes(include="number")
        if numeric_check.shape[1] < 2:
            st.warning("⚠️ Need at least 2 numeric columns for correlation analysis.")


# ━━ TAB 7 — AI Visualizations ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab7:
    st.markdown("## 🤖 AI-Powered Visualizations")
    st.markdown("These plots were automatically recommended and generated by the AI Agent.")

    if viz_figs:
        for i, viz in enumerate(viz_figs):
            plot_type = viz.get("plot_type", "Plot")
            columns = viz.get("columns", [])
            hue = viz.get("hue")
            msg = viz.get("message", "")

            st.markdown(
                f"**{plot_type.title()}** — Columns: `{', '.join(columns)}`"
                + (f" — Hue: `{hue}`" if hue else "")
            )

            fig = viz.get("fig")
            if fig is not None:
                st.pyplot(fig)
                plt.close(fig)
            else:
                st.warning(msg)

            if i < len(viz_figs) - 1:
                st.markdown("---")

        st.success(f"✅ {len(viz_figs)} AI-recommended visualizations generated!")
    else:
        st.info("No AI visualizations were generated. The agent may not have reached this step.")

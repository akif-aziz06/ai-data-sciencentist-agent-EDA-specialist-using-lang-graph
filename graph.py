"""
LangGraph-based EDA pipeline with @tool bindings and human-in-the-loop.

Replaces the simple LangChain chain with a full StateGraph where:
- Each EDA function is wrapped with @tool for LLM tool calling
- The LLM agent decides which tools to call in sequence
- Human-in-the-loop via interrupt() when the agent encounters ambiguity
"""

import re
import pandas as pd
from typing import TypedDict, Annotated
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver

from tool import (
    show_head as _show_head,
    dataset_stats as _dataset_stats,
    handle_missing_values as _handle_missing_values,
    data_engineering as _data_engineering,
    correlation_analysis as _correlation_analysis,
    detect_outliers as _detect_outliers,
    visualize_data as _visualize_data,
)

load_dotenv(dotenv_path=".evv")

# Clear any invalid proxy env vars (e.g. leftover placeholder "http://your-proxy-address:port")
import os
for proxy_var in ("HTTPS_PROXY", "HTTP_PROXY", "https_proxy", "http_proxy"):
    val = os.environ.get(proxy_var, "")
    if "your-proxy" in val or val.endswith(":port"):
        os.environ.pop(proxy_var, None)


# ══════════════════════════════════════════════════════════════
#  Shared DataFrame Store
# ══════════════════════════════════════════════════════════════

_store = {
    "df": None,
    "figures": {},
    "results": {},
    "viz_figures": [],
}


def set_working_df(df: pd.DataFrame):
    """Initialize the store with a fresh DataFrame."""
    _store["df"] = df.copy()
    _store["figures"] = {}
    _store["results"] = {}
    _store["viz_figures"] = []


def get_working_df() -> pd.DataFrame:
    return _store["df"]


def get_store() -> dict:
    """Return a copy of the current store."""
    return {
        "df": _store["df"],
        "figures": dict(_store["figures"]),
        "results": dict(_store["results"]),
        "viz_figures": list(_store["viz_figures"]),
    }


# ══════════════════════════════════════════════════════════════
#  @tool Wrapped EDA Functions
# ══════════════════════════════════════════════════════════════

@tool
def tool_show_head(n: int = 5) -> str:
    """Preview the first n rows of the dataset. Call this first to understand the data."""
    df = get_working_df()
    head_df, info = _show_head(df, n)
    _store["results"]["head"] = {"df": head_df, "info": info}
    return f"{info}\n\n{head_df.to_string()}"


@tool
def tool_dataset_stats() -> str:
    """Generate comprehensive statistics: shape, memory, dtypes, numeric stats with skewness and kurtosis, categorical stats, duplicates."""
    df = get_working_df()
    report = _dataset_stats(df)
    _store["results"]["stats"] = report
    return report


@tool
def tool_handle_missing_values() -> str:
    """Detect and fix missing values using smart imputation: median for skewed numeric, mean for symmetric, mode for categorical, forward-fill for datetime, drop if more than 50 percent missing."""
    df = get_working_df()
    original_nulls = df.isnull().sum()
    total_rows = len(df)

    # Build structured column-level data BEFORE cleaning
    column_details = []
    missing_cols = original_nulls[original_nulls > 0]
    for col in missing_cols.index:
        null_count = int(missing_cols[col])
        null_pct = round((null_count / total_rows) * 100, 1)
        dtype = str(df[col].dtype)

        # Determine strategy
        if null_pct > 50:
            strategy, reason = "Drop Column", f"{null_pct}% missing — too much data lost"
        elif pd.api.types.is_datetime64_any_dtype(df[col].dtype):
            strategy, reason = "Forward Fill", "Datetime — preserves time continuity"
        elif pd.api.types.is_numeric_dtype(df[col].dtype):
            skewness = round(df[col].skew(), 2)
            if abs(skewness) > 0.5:
                strategy = f"Median ({df[col].median():.4f})"
                reason = f"Skewness = {skewness} — median is robust to skewed data"
            else:
                strategy = f"Mean ({df[col].mean():.4f})"
                reason = f"Skewness = {skewness} — symmetric, mean is appropriate"
        else:
            mode_val = df[col].mode()[0] if not df[col].dropna().empty else "Unknown"
            strategy = f"Mode ('{mode_val}')"
            reason = "Categorical — mode is the safest imputation"

        column_details.append({
            "Column": col,
            "Data Type": dtype,
            "Missing": null_count,
            "Missing %": f"{null_pct}%",
            "Strategy": strategy,
            "Reason": reason,
        })

    # Now actually clean
    cleaned_df, report = _handle_missing_values(df.copy())
    _store["df"] = cleaned_df

    # Build before/after summary
    before_after = []
    for detail in column_details:
        col = detail["Column"]
        if col in cleaned_df.columns:
            after_nulls = int(cleaned_df[col].isnull().sum())
            status = "✅ Clean" if after_nulls == 0 else f"⚠️ {after_nulls} remain"
        else:
            after_nulls = 0
            status = "🗑️ Dropped"
        before_after.append({
            "Column": col,
            "Before (Nulls)": detail["Missing"],
            "After (Nulls)": after_nulls,
            "Status": status,
        })

    _store["results"]["missing"] = {
        "report": report,
        "original_nulls": original_nulls,
        "cleaned_df": cleaned_df,
        "column_details": column_details,
        "before_after": before_after,
        "total_missing_cols": len(missing_cols),
        "total_missing_values": int(missing_cols.sum()),
    }
    return report



@tool
def tool_data_engineering() -> str:
    """Run data engineering: fill remaining nulls, build full EDA report with descriptive stats, categorical value counts, and duplicate analysis."""
    df = get_working_df()
    cleaned_df, report = _data_engineering(df.copy())
    _store["df"] = cleaned_df
    _store["results"]["engineering"] = {
        "report": report,
        "cleaned_df": cleaned_df,
    }
    return report


@tool
def tool_detect_outliers() -> str:
    """Detect outliers in all numeric columns using the IQR method. Returns outlier counts, percentages, bounds, and generates box plots."""
    df = get_working_df()
    fig, summary_df, report = _detect_outliers(df)
    _store["figures"]["outliers"] = fig
    _store["results"]["outliers"] = {
        "summary_df": summary_df,
        "report": report,
    }
    return report


@tool
def tool_correlation_analysis(top_n: int = 5) -> str:
    """Generate a correlation heatmap and rank the top N most correlated feature pairs with direction."""
    df = get_working_df()
    fig, top_pairs, report = _correlation_analysis(df, top_n)
    _store["figures"]["correlation"] = fig
    _store["results"]["correlation"] = {
        "top_pairs": top_pairs,
        "report": report,
    }
    return report


@tool
def tool_visualize_data(plot_type: str, columns: str, hue: str = "") -> str:
    """Generate a plot. plot_type: histogram, scatter plot, box plot, bar plot, heatmap, pair plot, count plot, violin plot. columns: comma-separated column names. hue: optional categorical column for color grouping, leave empty for none."""
    df = get_working_df()
    col_list = [c.strip() for c in columns.split(",")]
    hue_val = None if hue in ("", "null", "None", "none") else hue
    fig, msg = _visualize_data(df, plot_type, col_list, hue=hue_val)
    if fig:
        _store["viz_figures"].append({
            "fig": fig,
            "plot_type": plot_type,
            "columns": col_list,
            "hue": hue_val,
            "message": msg,
        })
    return msg


# ══════════════════════════════════════════════════════════════
#  System Prompt
# ══════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are an expert AI Data Scientist and EDA Specialist Agent.
A user has uploaded a dataset. Run a complete Exploratory Data Analysis pipeline using your tools.

## Your Tools
1. tool_show_head — Preview the dataset
2. tool_dataset_stats — Comprehensive statistics
3. tool_handle_missing_values — Detect and fix missing values
4. tool_data_engineering — Data engineering + EDA report
5. tool_detect_outliers — Outlier detection (IQR method)
6. tool_correlation_analysis — Correlation heatmap + top pairs
7. tool_visualize_data — Generate specific plots

## Workflow
Call tools in this order:
1. tool_show_head
2. tool_dataset_stats
3. tool_handle_missing_values
4. tool_data_engineering
5. tool_detect_outliers
6. tool_correlation_analysis
7. Then generate 4-6 insightful plots using tool_visualize_data (mix of univariate, bivariate, multivariate). Use EXACT column names from the dataset.

## Human-in-the-Loop
Run autonomously by default. But if you encounter these ambiguous situations, STOP and ask the user by presenting numbered options:
- Column with 40-60% missing values: ask whether to drop or impute
- High-cardinality categorical (>50 unique values): ask how to handle
- Many outliers (>10% of rows in a column): ask whether to keep, cap, or remove
- Column looks numeric but stored as text: ask whether to convert

Format questions as:
"[Situation]. What would you like to do?
1. [Option A]
2. [Option B]
3. [Option C]"

ONLY ask when genuinely uncertain. For clear-cut cases, proceed automatically.
After completing all analysis, provide a brief final summary of key findings.

## Dataset Metadata
{metadata}"""


# ══════════════════════════════════════════════════════════════
#  LangGraph State & Nodes
# ══════════════════════════════════════════════════════════════

class EDAState(TypedDict):
    messages: Annotated[list, add_messages]


ALL_TOOLS = [
    tool_show_head,
    tool_dataset_stats,
    tool_handle_missing_values,
    tool_data_engineering,
    tool_detect_outliers,
    tool_correlation_analysis,
    tool_visualize_data,
]


def _get_llm():
    """Create the LLM with tools bound."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2, max_tokens=4000)
    return llm.bind_tools(ALL_TOOLS)


def agent_node(state: EDAState) -> dict:
    """LLM agent node — decides which tool to call or asks the human."""
    llm_with_tools = _get_llm()
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


HITL_MARKERS = [
    "what would you like",
    "what should i do",
    "how would you like",
    "would you prefer",
    "please choose",
    "select an option",
]


def should_continue(state: EDAState) -> str:
    """Route: tools, human_review, or end."""
    last_msg = state["messages"][-1]

    # If LLM wants to call a tool
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        return "tools"

    # If LLM is asking the user a question (HITL trigger)
    content = getattr(last_msg, "content", "").lower()
    if any(marker in content for marker in HITL_MARKERS):
        return "human_review"

    return END


def human_review_node(state: EDAState) -> dict:
    """Pauses the graph and waits for human input via interrupt()."""
    last_msg = state["messages"][-1]
    user_choice = interrupt({"question": last_msg.content})
    return {"messages": [HumanMessage(content=user_choice)]}


# ══════════════════════════════════════════════════════════════
#  Build & Compile the Graph
# ══════════════════════════════════════════════════════════════

def build_eda_graph():
    """Build and compile the EDA StateGraph with checkpointing."""
    graph = StateGraph(EDAState)

    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(ALL_TOOLS))
    graph.add_node("human_review", human_review_node)

    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", should_continue, {
        "tools": "tools",
        "human_review": "human_review",
        END: END,
    })
    graph.add_edge("tools", "agent")
    graph.add_edge("human_review", "agent")

    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)


# ══════════════════════════════════════════════════════════════
#  Public API
# ══════════════════════════════════════════════════════════════

def run_eda(df: pd.DataFrame, metadata: str, graph, thread_id: str = "eda-session"):
    """
    Start the EDA pipeline.
    Returns: (result, store, interrupted, interrupt_data)
    """
    set_working_df(df)

    initial_state = {
        "messages": [
            SystemMessage(content=SYSTEM_PROMPT.format(metadata=metadata)),
            HumanMessage(content="Run the complete EDA pipeline on this dataset now. Start with tool_show_head."),
        ]
    }

    config = {"configurable": {"thread_id": thread_id}}
    result = graph.invoke(initial_state, config=config)

    # Check if graph was interrupted
    graph_state = graph.get_state(config)
    if graph_state.next:
        interrupt_data = None
        if graph_state.tasks and graph_state.tasks[0].interrupts:
            interrupt_data = graph_state.tasks[0].interrupts[0].value
        return result, get_store(), True, interrupt_data

    return result, get_store(), False, None


def resume_eda(user_choice: str, graph, thread_id: str = "eda-session"):
    """Resume the pipeline after human input."""
    config = {"configurable": {"thread_id": thread_id}}
    result = graph.invoke(Command(resume=user_choice), config=config)

    graph_state = graph.get_state(config)
    if graph_state.next:
        interrupt_data = None
        if graph_state.tasks and graph_state.tasks[0].interrupts:
            interrupt_data = graph_state.tasks[0].interrupts[0].value
        return result, get_store(), True, interrupt_data

    return result, get_store(), False, None


def extract_options(text: str) -> list:
    """Extract numbered options from LLM text (e.g., '1. Drop column')."""
    pattern = r'^\s*\d+[\.\)]\s*(.+)$'
    options = re.findall(pattern, text, re.MULTILINE)
    return options if options else [text]

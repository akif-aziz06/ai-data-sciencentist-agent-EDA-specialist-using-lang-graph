import json
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from agent import load_dataset
from tool import (
    show_head, dataset_stats, handle_missing_values,
    data_engineering, correlation_analysis, detect_outliers, visualize_data,
)

load_dotenv(dotenv_path=".evv")

SYSTEM_PROMPT = """You are an expert Data Scientist and EDA Specialist.

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
- Base all recommendations strictly on the provided metadata — do NOT invent columns"""

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2, max_tokens=1500)

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{dataset_info}"),
])

chain = prompt | llm | StrOutputParser()


def _normalize_hue(hue):
    """Convert string 'null'/'None' to Python None."""
    return None if hue in ("null", "None", None) else hue


def run_auto_visualizations(df, response_text):
    """Parse LLM JSON response and generate all recommended plots."""
    recommendations = json.loads(response_text)

    for category in ["univariate", "bivariate", "multivariate"]:
        plots = recommendations.get(category, [])
        if not plots:
            print(f"⚠️  No {category} plots recommended.")
            continue

        print(f"\n📊 Generating {category.upper()} plots ({len(plots)} recommended)...\n")

        for i, plot in enumerate(plots, 1):
            plot_type = plot.get("plot_type", "histogram")
            columns = plot.get("columns", [])
            hue = _normalize_hue(plot.get("hue"))
            reason = plot.get("reason", "")

            print(f"  🎨 [{category.title()} #{i}] {plot_type.title()} — Columns: {columns}"
                  + (f" — Hue: {hue}" if hue else ""))
            print(f"     Reason: {reason}")

            fig, msg = visualize_data(df, plot_type, columns, hue=hue)
            print(f"     {msg}")
            if fig:
                plt.show()


if __name__ == "__main__":
    metadata, df = load_dataset()

    steps = [
        ("Dataset Preview", lambda: show_head(df)),
        ("Dataset Statistics", lambda: (dataset_stats(df),)),
    ]

    # Step 1–2: Preview & Stats
    for title, fn in steps:
        print(f"\n{'=' * 50}\n{title}\n{'=' * 50}")
        result = fn()
        if isinstance(result, tuple) and len(result) == 2:
            print(result[0].to_string() if hasattr(result[0], "to_string") else result[0])
            print(result[1])
        else:
            print(result[0])

    # Step 3: Missing Values
    print(f"\n{'=' * 50}\nMissing Value Handler\n{'=' * 50}")
    df, report = handle_missing_values(df)
    print(report)

    # Step 4: Data Engineering
    print(f"\n{'=' * 50}\nData Engineering & EDA\n{'=' * 50}")
    df, report = data_engineering(df)
    print(report)

    # Step 5: Outlier Detection
    print(f"\n{'=' * 50}\nOutlier Detection\n{'=' * 50}")
    fig, summary_df, report = detect_outliers(df)
    print(report)
    if fig:
        plt.show()

    # Step 6: Correlation Analysis
    print(f"\n{'=' * 50}\nCorrelation Analysis\n{'=' * 50}")
    fig, top_pairs, report = correlation_analysis(df)
    print(report)
    if fig:
        plt.show()

    # Step 7: LLM Recommendations
    print(f"\n{'=' * 50}\nAsking GPT-4o-mini for Plot Advice...\n{'=' * 50}")
    response = chain.invoke({"dataset_info": metadata})
    print(response)

    # Step 8: Auto-Visualize
    print(f"\n{'=' * 50}\nAuto-Generating Visualizations\n{'=' * 50}")
    try:
        run_auto_visualizations(df, response)
        print(f"\n{'=' * 50}\n✅ All visualizations generated!\n{'=' * 50}")
    except json.JSONDecodeError as e:
        print(f"\n❌ Failed to parse LLM response as JSON: {e}")
        print(f"Raw response:\n{response}")
    except Exception as e:
        print(f"\n❌ Error during auto-visualization: {e}")
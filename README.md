# 🤖 AI Data Scientist — EDA Specialist Agent

An AI-powered Exploratory Data Analysis (EDA) agent that automates the entire EDA pipeline. Upload any CSV or Excel dataset, and the agent runs **7 analysis tools** — from data cleaning to outlier detection — then uses **GPT-4o-mini** to recommend and auto-generate the most insightful visualizations for your data.

### 🔗 [Live Demo — Try It Now](https://akif-ai-data-scientist-agent.streamlit.app/)

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 📂 **Dataset Upload** | Upload any CSV or XLSX file (up to 10 MB) |
| 📊 **Dataset Statistics** | Shape, memory, dtypes, skewness, kurtosis, categorical breakdowns |
| 🧹 **Smart Missing Value Handler** | Auto-selects imputation strategy: median (skewed), mean (symmetric), mode (categorical), forward-fill (datetime), or drops columns with >50% nulls |
| 🔧 **Data Engineering** | Full EDA report on cleaned data — descriptive stats, value counts, duplicates |
| 🎯 **Outlier Detection** | IQR-based outlier detection with per-column summary + box plots |
| 🔗 **Correlation Analysis** | Heatmap + ranked list of top correlated feature pairs with strength labels |
| 🤖 **AI-Powered Visualizations** | GPT-4o-mini analyzes your dataset metadata and recommends the best plots (1–2 per category: univariate, bivariate, multivariate), then auto-generates them |

---

## 🏗️ Architecture

```
app.py          ← Streamlit frontend (7 tabs, file upload, UI rendering)
├── agent.py    ← Data loading (user uploads + metadata builder for LLM)
├── tool.py     ← 7 EDA tool functions (stats, cleaning, plotting, etc.)
└── llm.py      ← LangChain chain (GPT-4o-mini prompt + auto-visualization)
```

### How It Works

1. **User uploads a dataset** via the Streamlit sidebar
2. **Clicks "Start EDA"** to begin the analysis
3. **7 tabs** let the user explore each EDA step independently:
   - Preview → Statistics → Missing Values → Engineering → Outliers → Correlation → AI Visualizations
4. In the **AI Visualizations** tab, the dataset metadata is sent to **GPT-4o-mini**
5. The LLM returns **structured JSON** recommending the best plots
6. The app **automatically generates** each recommended plot using Seaborn/Matplotlib

---

## 📁 Project Structure

```
ai-data-sciencentist-agent-EDA-specialist/
│
├── app.py                  # Streamlit frontend — main entry point
├── agent.py                # Dataset loader + metadata builder
├── tool.py                 # 7 EDA tool functions
├── llm.py                  # LangChain LLM chain (CLI version)
│
├── .evv                    # OpenAI API key (not committed)
├── .evv.example            # Template showing required env variable
├── requirements.txt        # Python dependencies
├── runtime.txt             # Python version pin for Streamlit Cloud
│
├── .streamlit/
│   └── config.toml         # Streamlit config (10 MB upload limit)
│
├── .gitignore              # Git ignore rules
├── README.md               # This file
└── SETUP.md                # How to run locally
```

---

## 🛠️ Tech Stack

| Technology | Purpose |
|------------|---------|
| **Python** | Core language |
| **Streamlit** | Web frontend |
| **Pandas / NumPy** | Data manipulation |
| **Seaborn / Matplotlib** | Visualization |
| **LangChain** | LLM orchestration |
| **GPT-4o-mini (OpenAI)** | AI-powered plot recommendations |

---

## 📋 EDA Tool Details

### Tool 1 — `visualize_data()`
Generates plots based on a given plot type and column list. Supports: **histogram, scatter, box, bar, heatmap, pair plot, count plot, violin plot**. Returns a matplotlib figure for flexible rendering.

### Tool 2 — `data_engineering()`
Fills null values (median for numeric, mode for categorical) and produces a full EDA report covering shape, data types, descriptive statistics, categorical value counts, and duplicate rows.

### Tool 3 — `show_head()`
Returns the first N rows of the dataset for quick preview.

### Tool 4 — `correlation_analysis()`
Computes the correlation matrix, renders a heatmap, and ranks the top N most correlated feature pairs with direction (positive/negative) and strength labels.

### Tool 5 — `detect_outliers()`
Uses the **IQR (Interquartile Range) method** to detect outliers in every numeric column. Reports outlier count, percentage, and bounds. Renders box plots for visual inspection.

### Tool 6 — `dataset_stats()`
Comprehensive overview: shape, memory usage, dtype breakdown, per-column null/unique counts, numeric statistics (including skewness and kurtosis), and categorical statistics.

### Tool 7 — `handle_missing_values()`
**Smart imputation** — automatically selects the best strategy per column:

| Condition | Strategy | Reason |
|-----------|----------|--------|
| Numeric, skewed (abs(skew) > 0.5) | **Median** | Robust to outliers |
| Numeric, symmetric | **Mean** | Best central estimate |
| Categorical / Object | **Mode** | Most frequent value |
| Datetime | **Forward Fill** | Preserves time continuity |
| >50% missing | **Drop Column** | Too much data lost |

---

## 🤖 AI Visualization Engine

The LLM receives your dataset's metadata (column names, types, `df.info()`, `df.head()`) and returns a **structured JSON** response like:

```json
{
  "univariate": [
    {"plot_type": "histogram", "columns": ["total_bill"], "hue": null, "reason": "Shows distribution of bill amounts"}
  ],
  "bivariate": [
    {"plot_type": "scatter plot", "columns": ["total_bill", "tip"], "hue": "sex", "reason": "Reveals tipping patterns by gender"}
  ],
  "multivariate": [
    {"plot_type": "pair plot", "columns": ["total_bill", "tip", "size"], "hue": "sex", "reason": "Multi-variable relationship overview"}
  ]
}
```

Each recommendation is then **automatically rendered** as a Seaborn plot — no manual intervention needed.

---

## 🚀 Quick Start

See **[SETUP.md](SETUP.md)** for full local setup instructions.

**TL;DR:**
```bash
git clone https://github.com/akif-aziz06/ai-data-sciencentist-agent-EDA-specialist.git
cd ai-data-sciencentist-agent-EDA-specialist
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
# Add your OpenAI API key to .evv
streamlit run app.py
```

---

## 📄 License

This project is open source and available for educational and personal use.

---

**Built by [Akif Aziz](https://github.com/akif-aziz06)** — AI/ML enthusiast exploring the intersection of data science and large language models.
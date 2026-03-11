import pandas as pd
import io

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False


def load_dataset():
    """Load the built-in seaborn 'tips' dataset (CLI demo fallback)."""
    if not HAS_SEABORN:
        raise ImportError("seaborn is required for the demo dataset.")
    df = sns.load_dataset("tips")
    return _build_metadata(df), df


def load_uploaded_dataset(uploaded_file):
    """
    Load a user-uploaded CSV or XLSX file.

    Returns:
        (metadata_string, DataFrame)
    """
    filename = getattr(uploaded_file, "name", "uploaded")

    if filename.endswith((".xlsx", ".xls")):
        df = pd.read_excel(uploaded_file)
    else:
        df = pd.read_csv(uploaded_file)

    return _build_metadata(df), df


def _build_metadata(df: pd.DataFrame) -> str:
    """Build a metadata string from a DataFrame for the LLM."""
    metadata_dict = df.dtypes.astype(str).to_dict()
    dataset_info = f"Dataset Columns and Types: {metadata_dict}"

    buffer = io.StringIO()
    df.info(buf=buffer)
    df_info = buffer.getvalue()

    df_head = df.head().to_string()

    return (
        f"{dataset_info}\n\n"
        f"--- df.info() ---\n{df_info}\n"
        f"--- df.head() ---\n{df_head}"
    )


if __name__ == "__main__":
    print("Here is what we will send to the LLM:")
    print("-" * 20)
    metadata, _ = load_dataset()
    print(metadata)

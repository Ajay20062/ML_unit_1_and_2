from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
DATASET_PATH = ROOT / "corpus.csv"


def load_dataset(path: Path) -> pd.DataFrame:
    """Load a CSV dataset and validate it exists."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path.name}")
    return pd.read_csv(path)


def print_dataset_summary(filename: str, data: pd.DataFrame) -> None:
    """Display a clean overview of the dataset."""
    print("\n" + "=" * 70)
    print(f"Dataset: {filename}")
    print("=" * 70)
    print("\nFirst 5 rows:")
    print(data.head())
    print("\nLast 5 rows:")
    print(data.tail())
    print("\nShape:", data.shape)
    print("\nColumns:")
    print(data.columns.tolist())
    print("\nData types:")
    print(data.dtypes)


def main() -> None:
    corpus = load_dataset(DATASET_PATH)
    print_dataset_summary(DATASET_PATH.name, corpus)


if __name__ == "__main__":
    main()
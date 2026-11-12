from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
DATASET_PATH = ROOT / "corpus.csv"


def load_dataset(path: Path) -> pd.DataFrame:
    """Load the corpus and raise a clear error if it is missing."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path.name}")
    return pd.read_csv(path)


def main() -> None:
    corpus = load_dataset(DATASET_PATH)
    print("\n" + "=" * 70)
    print("Dataset Summary")
    print("=" * 70)
    print(f"\nFile: {DATASET_PATH.name}")
    corpus.info()
    print("\nStatistical Summary:")
    print(corpus.describe(include="all"))
    print("\nMissing Values:")
    print(corpus.isnull().sum())
    print("\nUnique Values:")
    print(corpus.nunique())


if __name__ == "__main__":
    main()
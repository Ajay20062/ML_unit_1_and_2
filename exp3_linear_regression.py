"""Experiment 3: predict question-passage relevance with linear regression."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ROOT = Path(__file__).resolve().parent
RANDOM_SEED = 42
REQUIRED_FILES = (
    "corpus.csv",
    "question_train.csv",
    "question_valid.csv",
    "question_test.csv",
    "ground_truth.json",
)


def load_data():
    missing = [name for name in REQUIRED_FILES if not (ROOT / name).exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required dataset files for Experiment 3: " + ", ".join(missing)
        )

    questions = {
        split: pd.read_csv(ROOT / f"question_{split}.csv")
        for split in ("train", "valid", "test")
    }
    corpus = pd.read_csv(ROOT / "corpus.csv")
    with open(ROOT / "ground_truth.json", encoding="utf-8") as file:
        ground_truth = {
            int(item_key): set(item_value)
            for item in json.load(file)
            for item_key, item_value in item.items()
        }
    return questions, corpus, ground_truth


def build_features(question_frame, corpus, ground_truth, vectorizer, negatives=2):
    """Create positive and sampled-negative question/passage examples."""
    corpus_ids = corpus["cid"].to_numpy()
    corpus_text = corpus["truncated_text"].fillna("").to_numpy()
    corpus_index = {int(cid): index for index, cid in enumerate(corpus_ids)}
    question_text = question_frame["question"].fillna("").tolist()
    question_vectors = vectorizer.transform(question_text)
    features = []
    labels = []
    rng = np.random.default_rng(RANDOM_SEED)

    for row_index, (_, row) in enumerate(question_frame.iterrows()):
        relevant = ground_truth.get(int(row["qid"]), set())
        positive_ids = [cid for cid in relevant if cid in corpus_index]
        if not positive_ids:
            continue
        candidate_ids = [positive_ids[0]]
        while len(candidate_ids) < negatives + 1:
            candidate = int(rng.choice(corpus_ids))
            if candidate not in relevant and candidate not in candidate_ids:
                candidate_ids.append(candidate)

        for candidate_id in candidate_ids:
            passage_index = corpus_index[candidate_id]
            passage_vector = vectorizer.transform([corpus_text[passage_index]])
            cosine = float(question_vectors[row_index].multiply(passage_vector).sum())
            question_tokens = set(str(row["question"]).lower().split())
            passage_tokens = set(str(corpus_text[passage_index]).lower().split())
            overlap = len(question_tokens & passage_tokens) / max(len(question_tokens), 1)
            features.append([
                cosine,
                overlap,
                np.log1p(len(question_tokens)),
                np.log1p(len(passage_tokens)),
            ])
            labels.append(float(candidate_id in relevant))
    return np.asarray(features), np.asarray(labels)


def main():
    questions, corpus, ground_truth = load_data()
    vectorizer = TfidfVectorizer(stop_words="english", max_features=50000)
    vectorizer.fit(pd.concat([questions["train"]["question"], corpus["truncated_text"]]))

    train_x, train_y = build_features(questions["train"], corpus, ground_truth, vectorizer)
    valid_x, valid_y = build_features(questions["valid"], corpus, ground_truth, vectorizer)

    model = LinearRegression().fit(train_x, train_y)
    predictions = model.predict(valid_x)
    print("Linear regression relevance prediction")
    print(f"Training examples: {len(train_y):,}")
    print(f"Validation examples: {len(valid_y):,}")
    print(f"MAE: {mean_absolute_error(valid_y, predictions):.4f}")
    print(f"RMSE: {mean_squared_error(valid_y, predictions) ** 0.5:.4f}")
    print(f"R2: {r2_score(valid_y, predictions):.4f}")
    print("Sample predictions (actual, predicted):")
    for actual, predicted in zip(valid_y[:5], predictions[:5]):
        print(f"  {actual:.0f}, {predicted:.4f}")


if __name__ == "__main__":
    main()
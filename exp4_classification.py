"""Experiment 4: Bayesian logistic regression and SVM classification."""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from exp3_linear_regression import build_features, load_data


class BayesianLogisticRegression:
    """Binary logistic regression with a Gaussian prior (MAP/Laplace fit)."""

    def __init__(self, prior_precision=1.0, max_iter=100, tolerance=1e-6):
        self.prior_precision = prior_precision
        self.max_iter = max_iter
        self.tolerance = tolerance

    def fit(self, features, labels):
        design = np.column_stack([np.ones(len(features)), features])
        weights = np.zeros(design.shape[1])
        prior = np.eye(design.shape[1]) * self.prior_precision
        prior[0, 0] = 0.0

        for _ in range(self.max_iter):
            probabilities = self._sigmoid(design @ weights)
            gradient = design.T @ (labels - probabilities) - prior @ weights
            diagonal = probabilities * (1.0 - probabilities)
            hessian = design.T @ (diagonal[:, None] * design) + prior
            update = np.linalg.solve(hessian, gradient)
            weights += update
            if np.linalg.norm(update) < self.tolerance:
                break
        self.weights_ = weights
        return self

    @staticmethod
    def _sigmoid(values):
        return 1.0 / (1.0 + np.exp(-np.clip(values, -40, 40)))

    def predict_proba(self, features):
        design = np.column_stack([np.ones(len(features)), features])
        positive = self._sigmoid(design @ self.weights_)
        return np.column_stack([1.0 - positive, positive])

    def predict(self, features):
        return (self.predict_proba(features)[:, 1] >= 0.5).astype(int)


def main():
    questions, corpus, ground_truth = load_data()
    vectorizer = TfidfVectorizer(stop_words="english", max_features=50000)
    vectorizer.fit(pd.concat([questions["train"]["question"], corpus["truncated_text"]]))

    train_x, train_y = build_features(questions["train"], corpus, ground_truth, vectorizer)
    valid_x, valid_y = build_features(questions["valid"], corpus, ground_truth, vectorizer)
    scaler = StandardScaler()
    train_x = scaler.fit_transform(train_x)
    valid_x = scaler.transform(valid_x)
    train_y = train_y.astype(int)
    valid_y = valid_y.astype(int)

    models = {
        "Bayesian logistic regression": BayesianLogisticRegression(),
        "SVM": SVC(kernel="linear", C=1.0),
    }
    print("Classification of relevant and non-relevant passages")
    print(f"Training examples: {len(train_y):,}")
    print(f"Validation examples: {len(valid_y):,}")
    for name, model in models.items():
        model.fit(train_x, train_y)
        predictions = model.predict(valid_x)
        print(f"\n{name}")
        print(f"Accuracy: {accuracy_score(valid_y, predictions):.4f}")
        print(classification_report(valid_y, predictions, target_names=["non-relevant", "relevant"], zero_division=0))


if __name__ == "__main__":
    main()
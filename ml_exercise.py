"""Train and evaluate a binary classifier on synthetic data."""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler


def split_data():
    """Return a stratified train/test split with features on different scales."""
    features, labels = make_classification(
        n_samples=1200,
        n_features=6,
        n_informative=4,
        n_redundant=0,
        weights=[0.9, 0.1],
        flip_y=0.02,
        random_state=42,
    )
    features *= np.array([1000.0, 0.01, 100.0, 1.0, 0.1, 10.0])
    return train_test_split(
        features, labels, test_size=0.25, stratify=labels, random_state=42
    )


def train_model(features, labels):
    model = make_pipeline(StandardScaler(), LogisticRegression())
    return model.fit(features, labels)


def predict_positive(model: Pipeline, features: np.ndarray) -> np.ndarray:
    """Return class-1 probabilities for a 2D batch of raw feature rows."""
    return model.predict_proba(features)[:, 1]


def main():
    train_x, test_x, train_y, test_y = split_data()
    model = train_model(train_x, train_y)
    baseline = DummyClassifier(strategy="prior").fit(train_x, train_y)
    scores = predict_positive(model, test_x)
    baseline_scores = baseline.predict_proba(test_x)[:, 1]
    predictions = scores >= 0.5
    tn, fp, fn, tp = confusion_matrix(test_y, predictions, labels=[0, 1]).ravel()

    print(f"Held-out rows: {len(test_y)} (positive: {test_y.sum()})")
    print(f"Baseline average precision: {average_precision_score(test_y, baseline_scores):.3f}")
    print(f"Model average precision:    {average_precision_score(test_y, scores):.3f}")
    print(f"At threshold 0.50: TN={tn}, FP={fp}, FN={fn}, TP={tp}")

    # Deliberately broken comparison: the fitted classifier expects scaled inputs.
    broken_scores = model.named_steps["logisticregression"].predict_proba(test_x)[:, 1]
    print(f"Broken average precision:   {average_precision_score(test_y, broken_scores):.3f}")
    print(f"First row, correct vs broken: {scores[0]:.6f} vs {broken_scores[0]:.6f}")


if __name__ == "__main__":
    main()

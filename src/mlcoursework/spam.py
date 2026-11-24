"""Task 1: spam detection with scikit-learn text pipelines.

Two pipelines are tuned with RandomizedSearchCV and compared on a held-out
validation split: TF-IDF + Multinomial Naive Bayes, and binary bag-of-words +
Logistic Regression.
"""
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

_NON_ALNUM = re.compile(r"[^a-zA-Z0-9]")
_SPACES = re.compile(r"\s+")


def preprocess_text(batch):
    """Lower-case, replace non-alphanumerics with spaces and collapse whitespace."""
    return [_SPACES.sub(" ", _NON_ALNUM.sub(" ", str(line))).strip().lower() for line in batch]


def load_dataset(path):
    """Load a spam CSV into (texts, labels) with labels as 0 = ham, 1 = spam.

    Accepts the course format (columns ``text``, ``label`` as 0/1) and the public
    Enron-1 ``spam_ham_dataset.csv`` format (``text``, ``label`` as ham/spam,
    ``label_num``).
    """
    df = pd.read_csv(path).dropna(subset=["text"])
    if "label_num" in df.columns:
        labels = df["label_num"]
    else:
        labels = df["label"].replace({"ham": 0, "spam": 1})
    return df["text"].astype(str), labels.astype(int)


def model_configs():
    """Pipelines and search spaces for the two classifiers."""
    cleaner = ("text_cleaning", FunctionTransformer(preprocess_text, validate=False))
    return {
        "BayesClassifier": {
            "workflow": Pipeline([
                cleaner,
                ("vectorizer", TfidfVectorizer(sublinear_tf=True)),
                ("model", MultinomialNB()),
            ]),
            "search_space": {
                "vectorizer__ngram_range": [(1, 1), (2, 2)],
                "vectorizer__max_df": [0.85, 0.9],
                "vectorizer__min_df": [1, 3],
                "model__alpha": [0.5, 1.5, 3.0],
            },
        },
        "LogisticClassifier": {
            "workflow": Pipeline([
                cleaner,
                ("vectorizer", CountVectorizer(binary=True)),
                ("model", LogisticRegression(solver="liblinear")),
            ]),
            "search_space": {
                "vectorizer__ngram_range": [(1, 2), (2, 3)],
                "vectorizer__max_features": [1500, 2500, 4000],
                "model__C": [0.05, 0.5, 5.0],
                "model__penalty": ["l1", "l2"],
            },
        },
    }


def model_path(model_label, out_dir="."):
    """Single source of truth for where a model is saved, so save and load can't drift apart."""
    return Path(out_dir) / f"{model_label}_final_model.pkl"


def evaluate(y_true, y_pred):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1 Score": f1_score(y_true, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]),
    }


def train_and_evaluate(X_train, y_train, X_valid, y_valid, configs=None, out_dir=".",
                       n_iter=5, cv=3, random_state=42, n_jobs=-1):
    """Tune each pipeline, save its best estimator, and return validation metrics per model."""
    results = []
    for label, cfg in (configs or model_configs()).items():
        search = RandomizedSearchCV(cfg["workflow"], cfg["search_space"], n_iter=n_iter, cv=cv,
                                    scoring="f1", random_state=random_state, n_jobs=n_jobs)
        search.fit(X_train, y_train)
        joblib.dump(search.best_estimator_, model_path(label, out_dir))
        metrics = evaluate(y_valid, search.predict(X_valid))
        results.append({"Model": label, **metrics, "best_params": search.best_params_})
    return results


def split(texts, labels, test_size=0.2, random_state=42):
    return train_test_split(texts, labels, test_size=test_size, stratify=labels,
                            random_state=random_state)


def predict_file(model_label, texts, out_dir="."):
    model = joblib.load(model_path(model_label, out_dir))
    return np.asarray(model.predict(texts)).astype(int)

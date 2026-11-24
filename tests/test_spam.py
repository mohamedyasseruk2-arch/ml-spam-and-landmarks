import numpy as np
import pandas as pd
import pytest

from mlcoursework import spam

HAM = ["meeting moved to 3pm, see attached agenda", "please review the gas volumes spreadsheet",
       "lunch tomorrow with the trading team?", "updated nominations for wellhead volumes",
       "can you send the contract draft by friday", "notes from today's pipeline call"]
SPAM = ["win a free prize now click here", "cheap meds online no prescription",
        "you have won $1,000,000 claim now", "limited offer!!! buy cheap watches",
        "click here to claim your free gift card", "earn money fast from home, free offer"]


def toy_data(repeat=6):
    texts = (HAM + SPAM) * repeat
    labels = ([0] * len(HAM) + [1] * len(SPAM)) * repeat
    return pd.Series(texts), pd.Series(labels)


def test_preprocess_text_normalises():
    assert spam.preprocess_text(["Hello,   WORLD!!\n$100"]) == ["hello world 100"]


def test_load_dataset_accepts_course_and_enron_formats(tmp_path):
    course = tmp_path / "course.csv"
    pd.DataFrame({"text": ["a", "b", None], "label": [0, 1, 1]}).to_csv(course, index=False)
    texts, labels = spam.load_dataset(course)
    assert list(texts) == ["a", "b"] and list(labels) == [0, 1]  # missing text dropped

    enron = tmp_path / "spam_ham_dataset.csv"
    pd.DataFrame({"label": ["ham", "spam"], "text": ["x", "y"], "label_num": [0, 1]}).to_csv(enron)
    texts, labels = spam.load_dataset(enron)
    assert list(labels) == [0, 1]


def test_evaluate_uses_sklearn_confusion_matrix():
    # Regression: the original notebook redefined confusion_matrix() as a function that called itself.
    m = spam.evaluate([0, 0, 1, 1], [0, 1, 1, 1])
    assert m["confusion_matrix"].tolist() == [[1, 1], [0, 2]]
    assert m["Precision"] == pytest.approx(2 / 3) and m["Recall"] == 1.0


def test_train_saves_models_that_predict_file_can_load(tmp_path):
    # Regression: the notebook saved "LogisticClassifier_final_model.pkl" but loaded
    # "LogRegModel_final_model.pkl", so a fresh run crashed.
    texts, labels = toy_data()
    X_tr, X_va, y_tr, y_va = spam.split(texts, labels, test_size=0.25)
    results = spam.train_and_evaluate(X_tr, y_tr, X_va, y_va, out_dir=tmp_path, n_iter=2, cv=2, n_jobs=1)

    assert [r["Model"] for r in results] == ["BayesClassifier", "LogisticClassifier"]
    for r in results:
        assert spam.model_path(r["Model"], tmp_path).exists()
        assert r["F1 Score"] >= 0.8, r  # trivially separable toy data
    preds = spam.predict_file("LogisticClassifier", pd.Series(["claim your free prize now"]), tmp_path)
    assert preds.dtype.kind == "i" and set(np.unique(preds)) <= {0, 1}

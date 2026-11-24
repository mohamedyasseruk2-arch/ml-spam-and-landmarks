# Spam detection and facial landmark regression

[![tests](https://github.com/mohamedyasseruk2-arch/ml-spam-and-landmarks/actions/workflows/tests.yml/badge.svg)](https://github.com/mohamedyasseruk2-arch/ml-spam-and-landmarks/actions/workflows/tests.yml)

Two supervised-learning tasks from my Machine Learning coursework (University of Sussex, 2025), built with
scikit-learn and OpenCV. I later refactored them from Colab notebooks into a tested package. That surfaced
four bugs in the original work, described below.

| Task | Approach |
|---|---|
| **Spam detection** (Enron email corpus) | Text cleaning → TF-IDF + Multinomial Naive Bayes vs binary bag-of-words + Logistic Regression, tuned with `RandomizedSearchCV` (3-fold, F1). Evaluated on accuracy, precision, recall, F1 and confusion matrices. |
| **Facial landmarks** (5 points per 256×256 face) | Dense-grid SIFT descriptors → standardised features → Ridge regression, plus a custom stage-wise residual ensemble. Evaluated on Euclidean pixel error (mean, median, Q3). |

## Results (spam detection, validation set)

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Naive Bayes (TF-IDF) | 0.925 | 0.993 | 0.752 | 0.856 |
| **Logistic Regression** (binary BoW, 1–2 grams) | **0.968** | 0.944 | 0.948 | **0.946** |

Naive Bayes almost never flags a real email as spam, but it misses a quarter of the spam. Logistic
Regression balances the two, so it was the model used for the test-set predictions.

## Bugs found when refactoring, and the tests that now guard them

| # | Bug in the original notebook | Effect | Regression test |
|---|---|---|---|
| 1 | Landmark predictions were converted back to pixels, but compared against ground truth that was still standardised (the Ridge cell even standardised it twice) | The reported landmark errors (214 px and 194 px on a 256 px image) were not meaningful | `test_perfect_model_scores_zero_pixel_error`, `test_validation_targets_stay_in_pixels` |
| 2 | Model saved as `LogisticClassifier_final_model.pkl`, loaded as `LogRegModel_final_model.pkl` | A clean run crashed; it only worked because an old file was on disk | `test_train_saves_models_that_predict_file_can_load` |
| 3 | `confusion_matrix` redefined as a wrapper that called itself | Infinite recursion, hidden only because a later import shadowed it again | `test_evaluate_uses_sklearn_confusion_matrix` |
| 4 | The face notebook printed "saved to landmarks_output.csv" but wrote `results_task2.csv` | Misleading output | Removed; the paths now come from one function |

Bug 1 means the original landmark error figures can't be trusted. The course's face dataset is no longer
available to re-run them, so the tests check the fixed evaluation on synthetic faces instead.

## Layout

```
src/mlcoursework/spam.py       text pipelines, tuning, evaluation, save/load
src/mlcoursework/landmarks.py  SIFT grid features, Ridge, stage-wise ensemble, pixel-error metrics
tests/                         13 tests, synthetic data only (they run in CI without the datasets)
scripts/run_spam.py            re-run the spam experiment on the real CSV
notebooks/                     the original coursework notebooks, outputs cleared
```

## Run it

```bash
pip install -r requirements.txt
python -m pytest -v
python scripts/run_spam.py data/spam_ham_dataset.csv   # see data/README.md for the dataset
```

"""Re-run the spam experiment on a CSV and print validation metrics.

Usage: python scripts/run_spam.py data/spam_ham_dataset.csv
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mlcoursework import spam  # noqa: E402

path = Path(sys.argv[1] if len(sys.argv) > 1 else "data/spam_ham_dataset.csv")
out = Path("results")
out.mkdir(exist_ok=True)

texts, labels = spam.load_dataset(path)
print(f"{len(texts)} emails, {labels.mean():.1%} spam")
X_tr, X_va, y_tr, y_va = spam.split(texts, labels)
results = spam.train_and_evaluate(X_tr, y_tr, X_va, y_va, out_dir=out)

rows = []
for r in results:
    print(f"\n{r['Model']}: best params {r['best_params']}")
    print(f"  accuracy {r['Accuracy']:.3f}  precision {r['Precision']:.3f}  "
          f"recall {r['Recall']:.3f}  F1 {r['F1 Score']:.3f}")
    print(f"  confusion matrix [[TN FP] [FN TP]] = {r['confusion_matrix'].tolist()}")
    rows.append({k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in r.items()})
(out / "spam_metrics.json").write_text(json.dumps(rows, indent=2, default=str))

from __future__ import annotations
import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, average_precision_score
import joblib

def main():
    out_dir = os.path.join(os.path.dirname(__file__), "out")

    Xtr = np.load(os.path.join(out_dir, "X_train.npy"), mmap_mode="r")
    ytr = np.load(os.path.join(out_dir, "y_train.npy"), mmap_mode="r")

    Xva = np.load(os.path.join(out_dir, "X_val.npy"), mmap_mode="r")
    yva = np.load(os.path.join(out_dir, "y_val.npy"), mmap_mode="r")

    clf = LogisticRegression(
        max_iter=2000,
        n_jobs=-1,
        class_weight="balanced",
        solver="lbfgs",
    )
    clf.fit(Xtr, ytr)

    cal = CalibratedClassifierCV(clf, method="isotonic", cv="prefit")
    cal.fit(Xva, yva)

    p = cal.predict_proba(Xva)[:, 1]
    auc = roc_auc_score(yva, p)
    ap = average_precision_score(yva, p)
    print(f"VAL AUC: {auc:.4f}")
    print(f"VAL PR-AUC: {ap:.4f}")

    artifact_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "backend", "artifacts")
    os.makedirs(artifact_dir, exist_ok=True)

    joblib.dump(clf, os.path.join(artifact_dir, "clf.joblib"))
    joblib.dump(cal, os.path.join(artifact_dir, "calibrator.joblib"))

    print("Saved backend artifacts:", artifact_dir)

if __name__ == "__main__":
    main()

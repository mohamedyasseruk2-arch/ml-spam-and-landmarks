"""Task 2: facial landmark regression from dense-grid SIFT descriptors.

Each image is described by SIFT descriptors computed on a fixed grid of
keypoints. Ridge regression, and a stage-wise residual ensemble of Ridge
models, predict 5 (x, y) landmarks per image. Errors are measured as the
Euclidean distance in pixels between predicted and true points.
"""
import cv2
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def grid_keypoints(height, width, spacing=32):
    """Keypoints on a uniform grid, offset by half a cell from the image edges."""
    return [cv2.KeyPoint(float(x), float(y), spacing)
            for y in range(spacing // 2, height, spacing)
            for x in range(spacing // 2, width, spacing)]


def sift_grid_features(images, spacing=32):
    """Flattened SIFT descriptors (one row per image) on a fixed keypoint grid."""
    sift = cv2.SIFT_create()
    height, width = images.shape[1:3]
    keypoints = grid_keypoints(height, width, spacing)
    rows = []
    for image in images:
        if image.dtype != np.uint8:
            image = (np.clip(image, 0, 1) * 255).astype(np.uint8)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
        # compute() can drop keypoints, so pass a copy and check the count
        kept, desc = sift.compute(gray, list(keypoints))
        if desc is None or len(kept) != len(keypoints):
            desc = np.zeros((len(keypoints), 128), dtype=np.float32)
        rows.append(desc.flatten())
    return np.asarray(rows, dtype=np.float32)


def point_errors(pred_pts, true_pts):
    """Euclidean distance per landmark, in the units of the inputs (pixels)."""
    pred = np.reshape(pred_pts, (-1, 2))
    true = np.reshape(true_pts, (-1, 2))
    return np.sqrt(np.sum((pred - true) ** 2, axis=1))


def summarise(errors, name):
    errors = np.asarray(errors)
    return {"model_name": name, "avg_error": float(errors.mean()),
            "median_error": float(np.median(errors)), "q3_error": float(np.percentile(errors, 75))}


class StageWiseRegressor:
    """Additive ensemble: each stage fits the residual left by the stages before it."""

    def __init__(self, base_estimator=Ridge, stages=3, estimator_params=None):
        self.stages = stages
        self.estimator_params = estimator_params or {}
        self.pipeline = [base_estimator(**self.estimator_params) for _ in range(stages)]

    def fit(self, X, y):
        prediction = np.zeros_like(y, dtype=float)
        for model in self.pipeline:
            model.fit(X, y - prediction)
            prediction += model.predict(X)
        return self

    def predict(self, X):
        return np.sum([m.predict(X) for m in self.pipeline], axis=0)


def prepare(features, points, test_size=0.2, random_state=42):
    """Split and scale. Validation targets are returned in PIXELS so errors stay interpretable."""
    y = points.reshape(points.shape[0], -1)
    X_tr, X_va, y_tr, y_va = train_test_split(features, y, test_size=test_size,
                                              random_state=random_state)
    sx, sy = StandardScaler(), StandardScaler()
    return {"X_train": sx.fit_transform(X_tr), "X_val": sx.transform(X_va),
            "y_train": sy.fit_transform(y_tr), "y_val_px": y_va,
            "scaler_X": sx, "scaler_y": sy}


def fit_and_score(model, data, name):
    """Fit on scaled targets, predict, convert back to pixels, then compare with pixel ground truth."""
    model.fit(data["X_train"], data["y_train"])
    pred_px = data["scaler_y"].inverse_transform(model.predict(data["X_val"]))
    return model, summarise(point_errors(pred_px, data["y_val_px"]), name)


def predict_points(model, features, scaler_X, scaler_y, n_points=5):
    pred = scaler_y.inverse_transform(model.predict(scaler_X.transform(features)))
    return pred.reshape(-1, n_points, 2)

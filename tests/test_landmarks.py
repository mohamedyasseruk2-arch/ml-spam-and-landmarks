import numpy as np
import pytest
from sklearn.linear_model import Ridge

from mlcoursework import landmarks as lm


def synthetic_faces(n=60, size=128, seed=0):
    """Images with 5 bright dots; the dots ARE the landmarks, so features carry real signal."""
    rng = np.random.default_rng(seed)
    images = np.zeros((n, size, size, 3), dtype=np.uint8)
    base = np.array([[40, 45], [88, 45], [64, 70], [48, 95], [80, 95]], dtype=float)
    pts = base + rng.uniform(-10, 10, size=(n, 1, 2)) + rng.normal(0, 1.5, size=(n, 5, 2))
    for img, p in zip(images, pts):
        for x, y in p.astype(int):
            img[max(y - 3, 0):y + 4, max(x - 3, 0):x + 4] = 255
    return images, pts


def test_grid_keypoints_cover_image():
    kps = lm.grid_keypoints(256, 256, spacing=32)
    assert len(kps) == 64
    assert kps[0].pt == (16.0, 16.0) and kps[-1].pt == (240.0, 240.0)


def test_sift_features_have_fixed_width():
    images, _ = synthetic_faces(n=4)
    feats = lm.sift_grid_features(images, spacing=32)
    assert feats.shape == (4, 16 * 128)  # 4x4 grid on 128 px, 128-d SIFT each
    assert np.isfinite(feats).all()


def test_sift_accepts_float_images():
    images, _ = synthetic_faces(n=2)
    as_float = images.astype(np.float32) / 255
    assert np.allclose(lm.sift_grid_features(as_float), lm.sift_grid_features(images), atol=1)


def test_point_errors_are_euclidean_pixels():
    pred = np.array([[[0, 0], [3, 4]]])
    true = np.array([[[0, 0], [0, 0]]])
    assert lm.point_errors(pred, true).tolist() == [0.0, 5.0]


def test_perfect_model_scores_zero_pixel_error():
    # Regression: the notebook compared pixel predictions against *scaled* ground truth,
    # so even a perfect model reported a large error. A perfect model must score 0 px.
    rng = np.random.default_rng(1)
    pts = rng.uniform(0, 256, size=(50, 5, 2))
    data = lm.prepare(pts.reshape(50, -1), pts)  # features = targets -> exactly learnable

    class Identity:
        def fit(self, X, y):
            return self

        def predict(self, X):  # X is scaled pts; map back into scaled-target space
            return data["scaler_y"].transform(data["scaler_X"].inverse_transform(X))

    _, summary = lm.fit_and_score(Identity(), data, "identity")
    assert summary["avg_error"] == pytest.approx(0, abs=1e-9)


def test_validation_targets_stay_in_pixels():
    images, pts = synthetic_faces()
    data = lm.prepare(lm.sift_grid_features(images), pts)
    assert 20 < data["y_val_px"].mean() < 110  # pixel coordinates, not z-scores


def test_models_beat_mean_baseline_and_errors_are_plausible():
    images, pts = synthetic_faces(n=120)
    data = lm.prepare(lm.sift_grid_features(images, spacing=16), pts)

    baseline = lm.point_errors(np.tile(data["scaler_y"].mean_, (len(data["y_val_px"]), 1)),
                               data["y_val_px"]).mean()
    _, ridge = lm.fit_and_score(Ridge(alpha=1.0), data, "Ridge")
    _, stage = lm.fit_and_score(lm.StageWiseRegressor(Ridge, 3, {"alpha": 1.0}), data, "StageWise")

    assert ridge["avg_error"] < baseline and stage["avg_error"] < baseline
    assert stage["avg_error"] < 128  # errors are on the image's own pixel scale


def test_stagewise_reduces_training_residual():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(200, 10))
    y = X @ rng.normal(size=(10, 4)) + 0.1 * rng.normal(size=(200, 4))
    one = lm.StageWiseRegressor(Ridge, 1, {"alpha": 50.0}).fit(X, y)
    three = lm.StageWiseRegressor(Ridge, 3, {"alpha": 50.0}).fit(X, y)
    err = lambda m: np.mean((m.predict(X) - y) ** 2)
    assert err(three) < err(one)


def test_predict_points_shape():
    images, pts = synthetic_faces(n=30)
    feats = lm.sift_grid_features(images)
    data = lm.prepare(feats, pts)
    model, _ = lm.fit_and_score(Ridge(), data, "Ridge")
    out = lm.predict_points(model, feats[:7], data["scaler_X"], data["scaler_y"])
    assert out.shape == (7, 5, 2)

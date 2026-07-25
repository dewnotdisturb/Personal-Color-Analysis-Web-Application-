import numpy as np

from personalcolor.train import generate_synthetic_dataset, train_and_save


def test_generate_synthetic_dataset_shapes():
    X, y = generate_synthetic_dataset(n_per_season=50, seed=1)
    assert X.shape == (200, 5)
    assert y.shape == (200,)
    unique, counts = np.unique(y, return_counts=True)
    assert set(unique) == {"Spring", "Summer", "Autumn", "Winter"}
    assert all(c == 50 for c in counts)


def test_train_and_save_produces_artifacts_and_reasonable_accuracy(tmp_path, monkeypatch):
    model_dir = tmp_path / "model"
    monkeypatch.setattr("personalcolor.train.MODEL_DIR", model_dir)
    monkeypatch.setattr("personalcolor.train.MODEL_PATH", model_dir / "knn_model.joblib")
    monkeypatch.setattr("personalcolor.train.SCALER_PATH", model_dir / "scaler.joblib")
    monkeypatch.setattr("personalcolor.train.REPORT_PATH", model_dir / "eval_report.txt")

    report = train_and_save(n_per_season=100, k=9, seed=1)

    assert (model_dir / "knn_model.joblib").exists()
    assert (model_dir / "scaler.joblib").exists()
    assert "accuracy" in report

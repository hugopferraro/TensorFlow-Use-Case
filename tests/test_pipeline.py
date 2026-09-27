"""Focused tests for the data contract and leakage-sensitive pipeline steps."""

import json

import numpy as np

import dry_bean_study as study


def test_downloaded_archive_and_schema_are_exact() -> None:
    assert study.ARCHIVE_FILE.exists()
    assert study.sha256_file(study.ARCHIVE_FILE) == study.EXPECTED_ARCHIVE_SHA256
    frame = study.load_and_validate_data()
    assert frame.shape == (study.EXPECTED_ROWS, len(study.FEATURES) + 1)
    assert list(frame.columns) == study.FEATURES + [study.TARGET]
    assert int(frame.duplicated().sum()) == 68


def test_split_is_disjoint_stratified_and_scaler_is_train_only() -> None:
    frame = study.load_and_validate_data()
    config = study.ExperimentConfig.from_quick_flag(True)
    data = study.prepare_data(frame, config)
    clean_rows = len(frame.drop_duplicates())

    assert len(data.y_train) + len(data.y_validation) + len(data.y_test) == clean_rows
    assert abs(len(data.y_train) / clean_rows - 0.70) < 0.001
    assert abs(len(data.y_validation) / clean_rows - 0.15) < 0.001
    assert abs(len(data.y_test) / clean_rows - 0.15) < 0.001
    assert np.allclose(data.x_train.mean(axis=0), 0, atol=1e-5)
    assert np.allclose(data.x_train.std(axis=0), 1, atol=1e-5)
    for labels in (data.y_train, data.y_validation, data.y_test):
        assert set(np.unique(labels)) == set(range(7))

    split_table = study.pd.read_csv(study.TABLES_DIR / "data_split.csv")
    assert not split_table["source_row"].duplicated().any()
    assert set(split_table["split"]) == {"train", "validation", "test"}


def test_hypermodel_has_valid_probability_output() -> None:
    study.set_reproducibility(study.SEED)
    hyperparameters = study.kt.HyperParameters()
    model = study.build_hypermodel(len(study.FEATURES), len(study.EXPECTED_CLASSES))(hyperparameters)
    output = model(np.zeros((3, len(study.FEATURES)), dtype=np.float32), training=False).numpy()

    assert output.shape == (3, len(study.EXPECTED_CLASSES))
    assert np.allclose(output.sum(axis=1), 1.0)
    assert model.loss == "sparse_categorical_crossentropy"


def test_saved_full_experiment_artifacts_support_inference() -> None:
    model_path = study.MODEL_DIR / "dry_bean_classifier.keras"
    results_path = study.ARTIFACTS_DIR / "final_results.json"
    manifest_path = study.ARTIFACTS_DIR / "reproducibility_manifest.json"
    assert model_path.exists() and results_path.exists() and manifest_path.exists()

    model = study.tf.keras.models.load_model(model_path, compile=False)
    scaler = study.joblib.load(study.MODEL_DIR / "standard_scaler.joblib")
    encoder = study.joblib.load(study.MODEL_DIR / "label_encoder.joblib")
    frame = study.load_and_validate_data()
    transformed = scaler.transform(frame.loc[:2, study.FEATURES]).astype(np.float32)
    probabilities = model.predict(transformed, verbose=0)
    predicted_labels = encoder.inverse_transform(probabilities.argmax(axis=1))

    results = json.loads(results_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert probabilities.shape == (3, 7)
    assert np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5)
    assert set(predicted_labels).issubset(study.EXPECTED_CLASSES)
    assert results["test_metrics"]["accuracy"] > 0.90
    assert manifest["config"]["quick"] is False

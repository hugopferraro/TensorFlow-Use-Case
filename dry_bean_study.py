# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Classificação de feijões secos com TensorFlow
#
# Este estudo constrói e avalia uma rede neural para classificar sete variedades
# de feijão a partir de 16 medidas geométricas extraídas por visão computacional.
# O fluxo foi projetado para ser reproduzível e impedir vazamento de informação:
# o teste fica isolado, e a padronização é ajustada somente no treino.

# %%
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import shutil
import sys
import urllib.request
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")

import joblib
import keras_tuner as kt
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import seaborn as sns
import sklearn
import tensorflow as tf
from scipy.stats import binomtest, f_oneway
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight

matplotlib.use("Agg")
sns.set_theme(style="whitegrid", context="notebook")

# %% [markdown]
# ## 1. Configuração e reprodutibilidade
#
# Todas as sementes, caminhos e limites experimentais ficam centralizados. O modo
# `--quick` serve apenas como teste de integração; os resultados finais usam a
# configuração completa.

# %%
ROOT = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw" / "DryBeanDataset"
DATA_FILE = RAW_DIR / "Dry_Bean_Dataset.xlsx"
ARCHIVE_FILE = DATA_DIR / "dry_bean_dataset.zip"
ARTIFACTS_DIR = ROOT / "artifacts"
FIGURES_DIR = ARTIFACTS_DIR / "figures"
TABLES_DIR = ARTIFACTS_DIR / "tables"
MODEL_DIR = ARTIFACTS_DIR / "model"
TUNER_DIR = ARTIFACTS_DIR / "tuner"

DATA_URL = "https://archive.ics.uci.edu/static/public/602/dry+bean+dataset.zip"
EXPECTED_ARCHIVE_SHA256 = "0a64eff5be87f48c3dbbfc0a12a56c5d5b5167ef8e61cd45d69b3e7c7130c06f"
EXPECTED_ROWS = 13_611
TARGET = "Class"
SEED = 42

FEATURES = [
    "Area",
    "Perimeter",
    "MajorAxisLength",
    "MinorAxisLength",
    "AspectRation",
    "Eccentricity",
    "ConvexArea",
    "EquivDiameter",
    "Extent",
    "Solidity",
    "roundness",
    "Compactness",
    "ShapeFactor1",
    "ShapeFactor2",
    "ShapeFactor3",
    "ShapeFactor4",
]
EXPECTED_CLASSES = {"BARBUNYA", "BOMBAY", "CALI", "DERMASON", "HOROZ", "SEKER", "SIRA"}


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int = SEED
    test_size: float = 0.15
    validation_size: float = 0.15
    max_trials: int = 12
    search_epochs: int = 60
    final_epochs: int = 120
    patience: int = 10
    bootstrap_iterations: int = 2_000
    permutation_repeats: int = 10
    quick: bool = False

    @classmethod
    def from_quick_flag(cls, quick: bool) -> "ExperimentConfig":
        if not quick:
            return cls()
        return cls(
            max_trials=2,
            search_epochs=8,
            final_epochs=12,
            patience=3,
            bootstrap_iterations=100,
            permutation_repeats=2,
            quick=True,
        )


def set_reproducibility(seed: int) -> None:
    """Seed Python, NumPy, and TensorFlow and request deterministic operations."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except (AttributeError, RuntimeError):
        pass


def ensure_directories() -> None:
    for directory in (DATA_DIR, RAW_DIR, ARTIFACTS_DIR, FIGURES_DIR, TABLES_DIR, MODEL_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def to_builtin(value: Any) -> Any:
    """Convert NumPy/Pandas values recursively into JSON-compatible objects."""
    if isinstance(value, dict):
        return {str(key): to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if pd.isna(value):
        return None
    return value


def save_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_builtin(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

# %% [markdown]
# ## 2. Aquisição e validação da base
#
# A fonte é o UCI Machine Learning Repository (DOI 10.24432/C50S4B). O arquivo
# baixado é autenticado por hash; a extração bloqueia caminhos fora de `data/`.
# Além do esquema esperado, verificam-se classes, ausências, finitude e domínios.

# %%
def safe_extract_zip(archive: Path, destination: Path) -> None:
    destination_resolved = destination.resolve()
    with zipfile.ZipFile(archive) as zipped:
        for member in zipped.infolist():
            target = (destination / member.filename).resolve()
            if destination_resolved not in target.parents and target != destination_resolved:
                raise ValueError(f"Unsafe path in archive: {member.filename}")
        zipped.extractall(destination)


def download_dataset(force: bool = False) -> Path:
    ensure_directories()
    if force or not ARCHIVE_FILE.exists():
        temporary = ARCHIVE_FILE.with_suffix(".download")
        print(f"[download] Obtendo conjunto de dados de {DATA_URL}")
        urllib.request.urlretrieve(DATA_URL, temporary)
        temporary.replace(ARCHIVE_FILE)

    actual_hash = sha256_file(ARCHIVE_FILE)
    if actual_hash != EXPECTED_ARCHIVE_SHA256:
        raise ValueError(
            "O arquivo baixado não corresponde ao artefato validado. "
            f"Esperado {EXPECTED_ARCHIVE_SHA256}; recebido {actual_hash}."
        )
    if force or not DATA_FILE.exists():
        safe_extract_zip(ARCHIVE_FILE, DATA_DIR / "raw")
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Planilha não encontrada apó extração: {DATA_FILE}")
    return DATA_FILE


def load_and_validate_data() -> pd.DataFrame:
    download_dataset()
    frame = pd.read_excel(DATA_FILE, engine="openpyxl")

    expected_columns = FEATURES + [TARGET]
    if list(frame.columns) != expected_columns:
        raise ValueError(f"Esquema inesperado: {list(frame.columns)}")
    if len(frame) != EXPECTED_ROWS:
        raise ValueError(f"Esperadas {EXPECTED_ROWS} linhas; recebidas {len(frame)}")
    if frame.isna().any().any():
        raise ValueError("A base contém valores ausentes.")
    if set(frame[TARGET].unique()) != EXPECTED_CLASSES:
        raise ValueError(f"Classes inesperadas: {set(frame[TARGET].unique())}")

    numeric = frame[FEATURES].to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all():
        raise ValueError("A base contém valores numéricos não finitos.")
    if (frame[["Area", "Perimeter", "MajorAxisLength", "MinorAxisLength", "ConvexArea"]] <= 0).any().any():
        raise ValueError("Foram encontradas medidas geométricas não positivas.")
    for bounded in ("Eccentricity", "Extent", "Solidity", "roundness", "Compactness", "ShapeFactor3", "ShapeFactor4"):
        if not frame[bounded].between(0, 1, inclusive="both").all():
            raise ValueError(f"{bounded} está fora do domínio [0, 1].")
    if not (frame["ConvexArea"] >= frame["Area"]).all():
        raise ValueError("ConvexArea deve ser maior ou igual a Area.")

    print(
        f"[validate] {len(frame):,} linhas, {len(FEATURES)} atributos, "
        f"{frame[TARGET].nunique()} classes, {int(frame.duplicated().sum())} duplicatas exatas."
    )
    return frame

# %% [markdown]
# ## 3. Análise exploratória
#
# A EDA documenta distribuições, desbalanceamento, valores atípicos univariados,
# correlações e associação entre cada atributo e a classe. O tamanho de efeito
# $\eta^2$ da ANOVA é usado descritivamente, sem pressupor causalidade.

# %%
def eta_squared(groups: list[np.ndarray]) -> float:
    all_values = np.concatenate(groups)
    grand_mean = all_values.mean()
    between = sum(len(group) * (group.mean() - grand_mean) ** 2 for group in groups)
    total = ((all_values - grand_mean) ** 2).sum()
    return float(between / total) if total else 0.0


def run_eda(frame: pd.DataFrame) -> dict[str, Any]:
    ensure_directories()
    print("[eda] Gerando tabelas e figuras descritivas...")

    description = frame[FEATURES].describe(percentiles=[0.01, 0.25, 0.5, 0.75, 0.99]).T
    description["skewness"] = frame[FEATURES].skew()
    description["kurtosis"] = frame[FEATURES].kurtosis()
    description["coefficient_of_variation"] = frame[FEATURES].std() / frame[FEATURES].mean()
    description.to_csv(TABLES_DIR / "descriptive_statistics.csv", encoding="utf-8")

    class_counts = frame[TARGET].value_counts().sort_index().rename("count").to_frame()
    class_counts["percentage"] = 100 * class_counts["count"] / len(frame)
    class_counts.to_csv(TABLES_DIR / "class_distribution.csv", encoding="utf-8")

    outlier_rows: list[dict[str, Any]] = []
    for feature in FEATURES:
        q1, q3 = frame[feature].quantile([0.25, 0.75])
        iqr = q3 - q1
        mask = (frame[feature] < q1 - 1.5 * iqr) | (frame[feature] > q3 + 1.5 * iqr)
        outlier_rows.append({"feature": feature, "iqr_outliers": int(mask.sum()), "percentage": 100 * mask.mean()})
    pd.DataFrame(outlier_rows).to_csv(TABLES_DIR / "iqr_outliers.csv", index=False, encoding="utf-8")

    association_rows: list[dict[str, Any]] = []
    for feature in FEATURES:
        groups = [group[feature].to_numpy() for _, group in frame.groupby(TARGET, sort=True)]
        statistic, p_value = f_oneway(*groups)
        association_rows.append(
            {"feature": feature, "anova_f": statistic, "p_value": p_value, "eta_squared": eta_squared(groups)}
        )
    associations = pd.DataFrame(association_rows).sort_values("eta_squared", ascending=False)
    associations.to_csv(TABLES_DIR / "feature_class_association.csv", index=False, encoding="utf-8")

    correlations = frame[FEATURES].corr(method="spearman")
    correlations.to_csv(TABLES_DIR / "spearman_correlations.csv", encoding="utf-8")
    upper = correlations.where(np.triu(np.ones(correlations.shape), k=1).astype(bool)).stack()
    strongest = upper.abs().sort_values(ascending=False).head(10)

    fig, axis = plt.subplots(figsize=(10, 5))
    sns.countplot(data=frame, x=TARGET, order=class_counts.index, hue=TARGET, legend=False, ax=axis)
    axis.set(title="Distribuição das classes", xlabel="Variedade", ylabel="Observações")
    axis.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "class_distribution.png", dpi=160)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(13, 10))
    sns.heatmap(correlations, cmap="vlag", center=0, square=True, ax=axis)
    axis.set_title("Correlação de Spearman entre atributos")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "correlation_heatmap.png", dpi=160)
    plt.close(fig)

    top_features = associations.head(6)["feature"].tolist()
    standardized = frame[top_features].apply(lambda column: (column - column.mean()) / column.std())
    plot_frame = standardized.assign(**{TARGET: frame[TARGET]}).melt(id_vars=TARGET, var_name="feature", value_name="z_score")
    fig, axis = plt.subplots(figsize=(14, 7))
    sns.boxplot(data=plot_frame, x="feature", y="z_score", hue=TARGET, showfliers=False, ax=axis)
    axis.set(title="Atributos com maior associação à classe", xlabel="Atributo", ylabel="Escore z global")
    axis.legend(title="Classe", bbox_to_anchor=(1.01, 1), loc="upper left")
    axis.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "top_feature_boxplots.png", dpi=160)
    plt.close(fig)

    diagnostics = {
        "raw_rows": len(frame),
        "columns": len(frame.columns),
        "numeric_features": len(FEATURES),
        "classes": frame[TARGET].nunique(),
        "missing_values": int(frame.isna().sum().sum()),
        "exact_duplicate_rows": int(frame.duplicated().sum()),
        "minority_class": class_counts["count"].idxmin(),
        "minority_count": int(class_counts["count"].min()),
        "majority_class": class_counts["count"].idxmax(),
        "majority_count": int(class_counts["count"].max()),
        "imbalance_ratio": float(class_counts["count"].max() / class_counts["count"].min()),
        "strongest_feature_class_associations": associations.head(5).to_dict(orient="records"),
        "strongest_absolute_correlations": [
            {"feature_pair": f"{left} | {right}", "spearman_abs": float(strongest.loc[(left, right)])}
            for left, right in strongest.index
        ],
    }
    save_json(diagnostics, ARTIFACTS_DIR / "data_diagnostics.json")
    return diagnostics

# %% [markdown]
# ## 4. Preparação sem vazamento
#
# Duplicatas exatas são removidas antes da divisão. A separação estratificada
# produz 70% para treino, 15% para validação e 15% para teste. O
# `StandardScaler` aprende média e desvio somente do treino. Pesos de classe são
# calculados somente desse mesmo subconjunto.

# %%
@dataclass
class PreparedData:
    x_train: np.ndarray
    x_validation: np.ndarray
    x_test: np.ndarray
    y_train: np.ndarray
    y_validation: np.ndarray
    y_test: np.ndarray
    test_source_rows: np.ndarray
    scaler: StandardScaler
    label_encoder: LabelEncoder
    class_weights: dict[int, float]


def prepare_data(frame: pd.DataFrame, config: ExperimentConfig) -> PreparedData:
    clean = frame.drop_duplicates().copy()
    clean["source_row"] = clean.index.astype(int)
    x = clean[FEATURES]
    y = clean[TARGET]
    rows = clean["source_row"]

    x_train, x_remainder, y_train_text, y_remainder, rows_train, rows_remainder = train_test_split(
        x,
        y,
        rows,
        test_size=config.test_size + config.validation_size,
        stratify=y,
        random_state=config.seed,
    )
    relative_test_size = config.test_size / (config.test_size + config.validation_size)
    x_validation, x_test, y_validation_text, y_test_text, rows_validation, rows_test = train_test_split(
        x_remainder,
        y_remainder,
        rows_remainder,
        test_size=relative_test_size,
        stratify=y_remainder,
        random_state=config.seed,
    )

    scaler = StandardScaler().fit(x_train)
    label_encoder = LabelEncoder().fit(y_train_text)
    x_train_scaled = scaler.transform(x_train).astype(np.float32)
    x_validation_scaled = scaler.transform(x_validation).astype(np.float32)
    x_test_scaled = scaler.transform(x_test).astype(np.float32)
    y_train = label_encoder.transform(y_train_text).astype(np.int32)
    y_validation = label_encoder.transform(y_validation_text).astype(np.int32)
    y_test = label_encoder.transform(y_test_text).astype(np.int32)

    weights = compute_class_weight(class_weight="balanced", classes=np.unique(y_train), y=y_train)
    class_weights = {int(label): float(weight) for label, weight in zip(np.unique(y_train), weights)}

    split_manifest = pd.concat(
        [
            pd.DataFrame({"source_row": rows_train, "split": "train", TARGET: y_train_text}),
            pd.DataFrame({"source_row": rows_validation, "split": "validation", TARGET: y_validation_text}),
            pd.DataFrame({"source_row": rows_test, "split": "test", TARGET: y_test_text}),
        ],
        ignore_index=True,
    ).sort_values("source_row")
    split_manifest.to_csv(TABLES_DIR / "data_split.csv", index=False, encoding="utf-8")

    split_sets = [set(rows_train), set(rows_validation), set(rows_test)]
    assert not (split_sets[0] & split_sets[1] or split_sets[0] & split_sets[2] or split_sets[1] & split_sets[2])
    assert sum(map(len, split_sets)) == len(clean)
    assert np.allclose(x_train_scaled.mean(axis=0), 0, atol=1e-5)
    assert np.allclose(x_train_scaled.std(axis=0), 1, atol=1e-5)
    assert set(label_encoder.classes_) == EXPECTED_CLASSES

    joblib.dump(scaler, MODEL_DIR / "standard_scaler.joblib")
    joblib.dump(label_encoder, MODEL_DIR / "label_encoder.joblib")
    print(
        f"[prepare] treino={len(y_train):,}, validação={len(y_validation):,}, "
        f"teste={len(y_test):,}; {len(frame) - len(clean)} duplicatas removidas."
    )
    return PreparedData(
        x_train=x_train_scaled,
        x_validation=x_validation_scaled,
        x_test=x_test_scaled,
        y_train=y_train,
        y_validation=y_validation,
        y_test=y_test,
        test_source_rows=rows_test.to_numpy(),
        scaler=scaler,
        label_encoder=label_encoder,
        class_weights=class_weights,
    )

# %% [markdown]
# ## 5. Baselines e arquitetura neural
#
# A classe majoritária e a regressão logística fornecem referências simples. A
# rede usa camadas densas, normalização em lote opcional, regularização L2 e
# dropout. `EarlyStopping` restaura os melhores pesos e `ReduceLROnPlateau`
# reduz a taxa quando a validação estagna.

# %%
def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, probabilities: np.ndarray | None = None) -> dict[str, float]:
    result = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
    }
    if probabilities is not None:
        result["log_loss"] = float(log_loss(y_true, probabilities, labels=np.arange(probabilities.shape[1])))
    return result


def train_baselines(data: PreparedData, seed: int) -> tuple[dict[str, dict[str, float]], np.ndarray]:
    dummy = DummyClassifier(strategy="prior", random_state=seed)
    dummy.fit(data.x_train, data.y_train)
    dummy_prediction = dummy.predict(data.x_test)

    logistic = LogisticRegression(max_iter=3_000, class_weight="balanced", random_state=seed)
    logistic.fit(data.x_train, data.y_train)
    logistic_prediction = logistic.predict(data.x_test)
    logistic_probability = logistic.predict_proba(data.x_test)

    results = {
        "majority_dummy": evaluate_predictions(data.y_test, dummy_prediction),
        "logistic_regression": evaluate_predictions(data.y_test, logistic_prediction, logistic_probability),
    }
    save_json(results, ARTIFACTS_DIR / "baseline_metrics.json")
    return results, logistic_prediction


def build_hypermodel(input_dimension: int, class_count: int) -> Callable[[kt.HyperParameters], tf.keras.Model]:
    def builder(hp: kt.HyperParameters) -> tf.keras.Model:
        l2_strength = hp.Choice("l2_strength", [1e-5, 1e-4, 1e-3])
        dropout_rate = hp.Choice("dropout_rate", [0.10, 0.25, 0.40])
        number_of_layers = hp.Int("hidden_layers", min_value=1, max_value=3)
        use_batch_normalization = hp.Boolean("batch_normalization", default=True)

        inputs = tf.keras.Input(shape=(input_dimension,), name="bean_measurements")
        values = inputs
        for layer_index in range(number_of_layers):
            units = hp.Choice(f"units_{layer_index + 1}", [32, 64, 128, 256])
            values = tf.keras.layers.Dense(
                units,
                activation="relu",
                kernel_initializer="he_normal",
                kernel_regularizer=tf.keras.regularizers.l2(l2_strength),
                name=f"dense_{layer_index + 1}",
            )(values)
            if use_batch_normalization:
                values = tf.keras.layers.BatchNormalization(name=f"batch_norm_{layer_index + 1}")(values)
            values = tf.keras.layers.Dropout(dropout_rate, name=f"dropout_{layer_index + 1}")(values)
        outputs = tf.keras.layers.Dense(class_count, activation="softmax", name="class_probability")(values)
        model = tf.keras.Model(inputs, outputs, name="dry_bean_classifier")
        learning_rate = hp.Choice("learning_rate", [1e-4, 3e-4, 1e-3, 3e-3])
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss="sparse_categorical_crossentropy",
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
        )
        return model

    return builder


def make_callbacks(config: ExperimentConfig, include_lr_reduction: bool = True) -> list[tf.keras.callbacks.Callback]:
    callbacks: list[tf.keras.callbacks.Callback] = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=config.patience,
            min_delta=1e-4,
            restore_best_weights=True,
            verbose=0,
        ),
        tf.keras.callbacks.TerminateOnNaN(),
    ]
    if include_lr_reduction:
        callbacks.append(
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=max(2, config.patience // 2), min_lr=1e-6, verbose=0
            )
        )
    return callbacks

# %% [markdown]
# ## 6. Busca de hiperparâmetros e treinamento final
#
# `RandomSearch` explora profundidade, largura, dropout, L2, normalização e taxa
# de aprendizagem. A seleção usa somente acurácia de validação; o teste não é
# consultado pela busca. Depois, um modelo novo é treinado com a configuração
# vencedora.

# %%
def tune_and_train(data: PreparedData, config: ExperimentConfig) -> tuple[tf.keras.Model, dict[str, Any], dict[str, list[float]]]:
    if TUNER_DIR.exists():
        shutil.rmtree(TUNER_DIR)
    builder = build_hypermodel(data.x_train.shape[1], len(data.label_encoder.classes_))
    tuner = kt.RandomSearch(
        builder,
        objective=kt.Objective("val_accuracy", direction="max"),
        max_trials=config.max_trials,
        executions_per_trial=1,
        overwrite=True,
        directory=TUNER_DIR,
        project_name="dry_bean_random_search",
        seed=config.seed,
    )
    print(f"[tuning] Executando {config.max_trials} tentativas de busca aleatória...")
    tuner.search(
        data.x_train,
        data.y_train,
        validation_data=(data.x_validation, data.y_validation),
        epochs=config.search_epochs,
        batch_size=64,
        class_weight=data.class_weights,
        callbacks=make_callbacks(config, include_lr_reduction=False),
        verbose=0,
    )

    best_hyperparameters = tuner.get_best_hyperparameters(1)[0]
    trials = []
    for trial_id, trial in tuner.oracle.trials.items():
        trials.append(
            {
                "trial_id": trial_id,
                "status": trial.status,
                "validation_accuracy": trial.score,
                **trial.hyperparameters.values,
            }
        )
    pd.DataFrame(trials).sort_values("validation_accuracy", ascending=False).to_csv(
        TABLES_DIR / "hyperparameter_trials.csv", index=False, encoding="utf-8"
    )

    set_reproducibility(config.seed)
    model = tuner.hypermodel.build(best_hyperparameters)
    history = model.fit(
        data.x_train,
        data.y_train,
        validation_data=(data.x_validation, data.y_validation),
        epochs=config.final_epochs,
        batch_size=64,
        class_weight=data.class_weights,
        callbacks=make_callbacks(config),
        verbose=0,
    )
    model.save(MODEL_DIR / "dry_bean_classifier.keras")
    best_values = dict(best_hyperparameters.values)
    best_values["trained_epochs"] = len(history.history["loss"])
    best_values["best_epoch_by_validation_loss"] = int(np.argmin(history.history["val_loss"]) + 1)
    best_values["best_validation_accuracy"] = float(max(history.history["val_accuracy"]))
    save_json(best_values, ARTIFACTS_DIR / "best_hyperparameters.json")
    print(f"[train] Melhor configuração: {best_values}")
    return model, best_values, history.history

# %% [markdown]
# ## 7. Avaliação final, incerteza e interpretação
#
# A avaliação inclui métricas globais e por classe, matriz de confusão,
# intervalos bootstrap e comparação pareada com a regressão logística. A
# importância por permutação mede a queda de acurácia ao embaralhar um atributo;
# é associativa, não causal.

# %%
def bootstrap_intervals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    iterations: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    generator = np.random.default_rng(seed)
    accuracy_values = np.empty(iterations)
    macro_f1_values = np.empty(iterations)
    for index in range(iterations):
        sample = generator.integers(0, len(y_true), size=len(y_true))
        accuracy_values[index] = accuracy_score(y_true[sample], y_pred[sample])
        macro_f1_values[index] = f1_score(y_true[sample], y_pred[sample], average="macro", zero_division=0)
    return {
        "accuracy": {
            "lower_95": float(np.quantile(accuracy_values, 0.025)),
            "upper_95": float(np.quantile(accuracy_values, 0.975)),
        },
        "macro_f1": {
            "lower_95": float(np.quantile(macro_f1_values, 0.025)),
            "upper_95": float(np.quantile(macro_f1_values, 0.975)),
        },
    }


def permutation_importance(
    model: tf.keras.Model,
    x_test: np.ndarray,
    y_test: np.ndarray,
    repeats: int,
    seed: int,
) -> pd.DataFrame:
    generator = np.random.default_rng(seed)
    baseline_prediction = model.predict(x_test, verbose=0).argmax(axis=1)
    baseline_accuracy = accuracy_score(y_test, baseline_prediction)
    rows: list[dict[str, Any]] = []
    for feature_index, feature in enumerate(FEATURES):
        drops = []
        for _ in range(repeats):
            permuted = x_test.copy()
            permuted[:, feature_index] = generator.permutation(permuted[:, feature_index])
            prediction = model.predict(permuted, verbose=0).argmax(axis=1)
            drops.append(baseline_accuracy - accuracy_score(y_test, prediction))
        rows.append(
            {
                "feature": feature,
                "mean_accuracy_drop": float(np.mean(drops)),
                "std_accuracy_drop": float(np.std(drops, ddof=1)) if repeats > 1 else 0.0,
            }
        )
    return pd.DataFrame(rows).sort_values("mean_accuracy_drop", ascending=False)


def plot_learning_curves(history: dict[str, list[float]]) -> None:
    epochs = np.arange(1, len(history["loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(epochs, history["loss"], label="Treino")
    axes[0].plot(epochs, history["val_loss"], label="Validação")
    axes[0].set(title="Perda por época", xlabel="Época", ylabel="Entropia cruzada")
    axes[0].legend()
    axes[1].plot(epochs, history["accuracy"], label="Treino")
    axes[1].plot(epochs, history["val_accuracy"], label="Validação")
    axes[1].set(title="Acurácia por época", xlabel="Época", ylabel="Acurácia")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "learning_curves.png", dpi=160)
    plt.close(fig)


def evaluate_final_model(
    model: tf.keras.Model,
    data: PreparedData,
    logistic_prediction: np.ndarray,
    baseline_metrics: dict[str, dict[str, float]],
    history: dict[str, list[float]],
    config: ExperimentConfig,
) -> dict[str, Any]:
    probabilities = model.predict(data.x_test, batch_size=256, verbose=0)
    predictions = probabilities.argmax(axis=1)
    metrics = evaluate_predictions(data.y_test, predictions, probabilities)
    intervals = bootstrap_intervals(data.y_test, predictions, config.bootstrap_iterations, config.seed)

    neural_correct = predictions == data.y_test
    logistic_correct = logistic_prediction == data.y_test
    neural_only = int(np.sum(neural_correct & ~logistic_correct))
    logistic_only = int(np.sum(~neural_correct & logistic_correct))
    discordant = neural_only + logistic_only
    mcnemar_p = float(binomtest(neural_only, discordant, 0.5).pvalue) if discordant else 1.0

    report = classification_report(
        data.y_test,
        predictions,
        labels=np.arange(len(data.label_encoder.classes_)),
        target_names=data.label_encoder.classes_,
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report).T.to_csv(TABLES_DIR / "classification_report.csv", encoding="utf-8")

    matrix = confusion_matrix(data.y_test, predictions)
    normalized_matrix = confusion_matrix(data.y_test, predictions, normalize="true")
    pd.DataFrame(matrix, index=data.label_encoder.classes_, columns=data.label_encoder.classes_).to_csv(
        TABLES_DIR / "confusion_matrix.csv", encoding="utf-8"
    )
    fig, axis = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        normalized_matrix,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=data.label_encoder.classes_,
        yticklabels=data.label_encoder.classes_,
        ax=axis,
    )
    axis.set(title="Matriz de confusão normalizada", xlabel="Classe predita", ylabel="Classe real")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "confusion_matrix_normalized.png", dpi=160)
    plt.close(fig)

    importance = permutation_importance(
        model, data.x_test, data.y_test, config.permutation_repeats, config.seed
    )
    importance.to_csv(TABLES_DIR / "permutation_importance.csv", index=False, encoding="utf-8")
    fig, axis = plt.subplots(figsize=(9, 6))
    sns.barplot(data=importance.head(10), x="mean_accuracy_drop", y="feature", color="#4472C4", ax=axis)
    axis.set(title="Importância por permutação (10 maiores)", xlabel="Queda média de acurácia", ylabel="Atributo")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "permutation_importance.png", dpi=160)
    plt.close(fig)

    prediction_table = pd.DataFrame(
        {
            "source_row": data.test_source_rows,
            "actual": data.label_encoder.inverse_transform(data.y_test),
            "predicted": data.label_encoder.inverse_transform(predictions),
            "confidence": probabilities.max(axis=1),
            "correct": predictions == data.y_test,
        }
    )
    prediction_table.to_csv(TABLES_DIR / "test_predictions.csv", index=False, encoding="utf-8")
    plot_learning_curves(history)

    final_results = {
        "test_metrics": metrics,
        "bootstrap_95_percent_intervals": intervals,
        "baselines": baseline_metrics,
        "paired_comparison_with_logistic_regression": {
            "neural_only_correct": neural_only,
            "logistic_only_correct": logistic_only,
            "exact_mcnemar_p_value": mcnemar_p,
        },
        "training_diagnostics": {
            "epochs": len(history["loss"]),
            "minimum_validation_loss": float(min(history["val_loss"])),
            "final_training_accuracy": float(history["accuracy"][-1]),
            "final_validation_accuracy": float(history["val_accuracy"][-1]),
            "final_accuracy_gap": float(history["accuracy"][-1] - history["val_accuracy"][-1]),
        },
        "most_important_features": importance.head(5).to_dict(orient="records"),
        "classification_report": report,
    }
    save_json(final_results, ARTIFACTS_DIR / "final_results.json")
    print(f"[evaluate] Métricas finais no teste: {metrics}")
    return final_results

# %% [markdown]
# ## 8. Orquestração e artefatos de auditoria
#
# O manifesto registra versões, hash do dado, configuração e hardware. Isso
# permite auditar exatamente o ambiente que produziu os resultados.

# %%
def write_reproducibility_manifest(config: ExperimentConfig) -> None:
    manifest = {
        "config": asdict(config),
        "dataset": {
            "source": DATA_URL,
            "doi": "10.24432/C50S4B",
            "archive_sha256": sha256_file(ARCHIVE_FILE),
            "license": "CC BY 4.0",
        },
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "tensorflow": tf.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "keras_tuner": kt.__version__,
            "physical_devices": [device.name for device in tf.config.list_physical_devices()],
        },
    }
    save_json(manifest, ARTIFACTS_DIR / "reproducibility_manifest.json")


def run_pipeline(stage: str, quick: bool = False, force_download: bool = False) -> dict[str, Any] | None:
    config = ExperimentConfig.from_quick_flag(quick)
    set_reproducibility(config.seed)
    ensure_directories()
    if stage == "download":
        download_dataset(force=force_download)
        return None

    frame = load_and_validate_data()
    if stage == "validate":
        return {"rows": len(frame), "duplicates": int(frame.duplicated().sum())}

    diagnostics = run_eda(frame)
    if stage == "eda":
        return diagnostics

    prepared = prepare_data(frame, config)
    baselines, logistic_prediction = train_baselines(prepared, config.seed)
    model, best_hyperparameters, history = tune_and_train(prepared, config)
    results = evaluate_final_model(
        model, prepared, logistic_prediction, baselines, history, config
    )
    write_reproducibility_manifest(config)
    results["best_hyperparameters"] = best_hyperparameters
    results["data_diagnostics"] = diagnostics
    print(f"[done] Artefatos gravados em {ARTIFACTS_DIR}")
    return results


def parse_args(arguments: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estudo reproduzível de classificação da base Dry Bean.")
    parser.add_argument(
        "--stage",
        choices=("download", "validate", "eda", "all"),
        default="all",
        help="Etapa final a executar; etapas anteriores são sempre validadas.",
    )
    parser.add_argument("--quick", action="store_true", help="Executa busca reduzida apenas para teste de integração.")
    parser.add_argument("--force-download", action="store_true", help="Baixa novamente o arquivo original da UCI.")
    return parser.parse_args(arguments)


def main(arguments: list[str] | None = None) -> None:
    args = parse_args(arguments)
    run_pipeline(args.stage, quick=args.quick, force_download=args.force_download)


# %%
if __name__ == "__main__":
    # Kernels inject their own command-line arguments. Running the canonical
    # pipeline directly keeps the notebook equivalent to ``--stage all``.
    if "ipykernel" in sys.modules:
        run_pipeline("all")
    else:
        main()

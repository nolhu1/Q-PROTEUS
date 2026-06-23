"""Model training, loading, and sequence predictor adapters."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
from pathlib import Path
from typing import Any, Iterable, Protocol

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .constants import DEFAULT_FEATURE_COLUMNS
from .features import calculate_features, feature_table, feature_vector


class SequenceProbabilityPredictor(Protocol):
    def predict_proba(self, sequences: list[str]) -> np.ndarray:
        ...


class SequenceValuePredictor(Protocol):
    def predict(self, sequences: list[str]) -> list[float] | np.ndarray:
        ...


@dataclass
class ModelReport:
    metrics: dict[str, Any]
    classification_report_text: str


class SklearnSequenceClassifier:
    """Adapter from descriptor features to a scikit-learn probability model."""

    def __init__(
        self,
        pipeline: Pipeline,
        feature_columns: Iterable[str] = DEFAULT_FEATURE_COLUMNS,
        positive_class_index: int = 1,
    ) -> None:
        self.pipeline = pipeline
        self.feature_columns = tuple(feature_columns)
        self.positive_class_index = positive_class_index

    def predict_proba(self, sequences: list[str]) -> np.ndarray:
        table = feature_table(sequences, columns=self.feature_columns)
        probabilities = self.pipeline.predict_proba(table[list(self.feature_columns)])
        return probabilities[:, self.positive_class_index]

    def embed(self, sequences: list[str]) -> np.ndarray:
        """Descriptor-space embedding fallback for RRI structural robustness."""
        table = feature_table(sequences, columns=self.feature_columns)
        scaler = self.pipeline.named_steps.get("scaler")
        values = table[list(self.feature_columns)].to_numpy(dtype=float)
        if scaler is None:
            return values
        return scaler.transform(values)


class DescriptorEmbeddingProvider:
    """Use standardized physicochemical descriptors as an embedding fallback."""

    def __init__(self, feature_columns: Iterable[str] = DEFAULT_FEATURE_COLUMNS) -> None:
        self.feature_columns = tuple(feature_columns)

    def embed(self, sequences: list[str]) -> np.ndarray:
        return np.vstack([feature_vector(sequence, self.feature_columns) for sequence in sequences])


def mic_to_efficacy(log10_mic: float, *, threshold: float = 1.0, slope: float = 3.0) -> float:
    """Map lower log10(MIC uM) values to higher [0, 1] efficacy scores."""
    return float(1.0 / (1.0 + np.exp(slope * (log10_mic - threshold))))


COLUMN_ALIASES = {
    "Sequence": "sequence",
    "Label": "label",
    "Hemolytic_Label": "label",
    "Length": "length",
    "Net_Charge": "net_charge",
    "Hydrophobicity_GRAVY": "gravy",
    "Fraction_Positive_AA": "fraction_positive",
    "Fraction_Negative_AA": "fraction_negative",
    "Fraction_Hydrophobic_AA": "fraction_hydrophobic",
    "Instability_Index": "instability_index",
    "Molecular_Weight": "molecular_weight",
}


def coerce_training_frame(
    frame: pd.DataFrame,
    *,
    label_column: str,
    feature_columns: Iterable[str],
) -> tuple[pd.DataFrame, str, list[str]]:
    """Normalize legacy dataset columns and fill missing descriptors from sequences."""
    # Older project CSVs used display-style column names; normalize them at the
    # boundary so the modeling code can stay consistent.
    working = frame.rename(columns={k: v for k, v in COLUMN_ALIASES.items() if k in frame.columns}).copy()
    resolved_label = COLUMN_ALIASES.get(label_column, label_column)
    if resolved_label not in working.columns:
        raise ValueError(f"Missing label column: {label_column}")

    requested_features = list(feature_columns)
    missing_features = [column for column in requested_features if column not in working.columns]
    if missing_features and "sequence" in working.columns:
        # Recompute only descriptors absent from the input, preserving curated
        # values when they were already stored with the dataset.
        computed = pd.DataFrame([calculate_features(sequence) for sequence in working["sequence"]])
        for column in missing_features:
            if column in computed.columns:
                working[column] = computed[column]

    available_features = [column for column in requested_features if column in working.columns]
    if not available_features:
        raise ValueError("No usable feature columns were found")
    return working, resolved_label, available_features


def train_random_forest_classifier(
    frame: pd.DataFrame,
    *,
    label_column: str = "label",
    feature_columns: Iterable[str] = DEFAULT_FEATURE_COLUMNS,
    random_state: int = 42,
) -> tuple[SklearnSequenceClassifier, ModelReport]:
    working, resolved_label, columns = coerce_training_frame(
        frame,
        label_column=label_column,
        feature_columns=feature_columns,
    )

    x = working[columns]
    y = working[resolved_label].astype(int)
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=random_state,
        stratify=y,
    )
    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=12,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    class_weight="balanced",
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    pipeline.fit(x_train, y_train)

    y_pred = pipeline.predict(x_test)
    y_prob = pipeline.predict_proba(x_test)[:, 1]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    cv_scores = cross_val_score(pipeline, x, y, cv=cv, scoring="accuracy")
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "cv_accuracy_mean": float(np.mean(cv_scores)),
        "cv_accuracy_sd": float(np.std(cv_scores, ddof=1)),
        "n_samples": int(len(working)),
        "n_features": int(len(columns)),
    }
    report = ModelReport(
        metrics=metrics,
        classification_report_text=classification_report(y_test, y_pred),
    )
    return SklearnSequenceClassifier(pipeline, columns), report


def train_xgboost_classifier(
    frame: pd.DataFrame,
    *,
    label_column: str = "label",
    feature_columns: Iterable[str] = DEFAULT_FEATURE_COLUMNS,
    random_state: int = 42,
) -> tuple[SklearnSequenceClassifier, ModelReport]:
    try:
        from xgboost import XGBClassifier
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Install xgboost to train the toxicity model") from exc

    working, resolved_label, columns = coerce_training_frame(
        frame,
        label_column=label_column,
        feature_columns=feature_columns,
    )
    x = working[columns]
    y = working[resolved_label].astype(int)
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=random_state,
        stratify=y,
    )
    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                XGBClassifier(
                    n_estimators=400,
                    max_depth=4,
                    learning_rate=0.05,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    eval_metric="logloss",
                    random_state=random_state,
                ),
            ),
        ]
    )
    pipeline.fit(x_train, y_train)
    y_pred = pipeline.predict(x_test)
    y_prob = pipeline.predict_proba(x_test)[:, 1]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    cv_scores = cross_val_score(pipeline, x, y, cv=cv, scoring="accuracy")
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "cv_accuracy_mean": float(np.mean(cv_scores)),
        "cv_accuracy_sd": float(np.std(cv_scores, ddof=1)),
        "n_samples": int(len(working)),
        "n_features": int(len(columns)),
    }
    report = ModelReport(metrics, classification_report(y_test, y_pred))
    return SklearnSequenceClassifier(pipeline, columns), report


def save_predictor(predictor: Any, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(predictor, path)


def load_predictor(spec: str | Path) -> Any:
    """Load a predictor from a joblib file or a module:attribute spec."""
    spec_str = str(spec)
    if Path(spec_str).exists():
        return joblib.load(spec_str)
    if ":" not in spec_str:
        raise FileNotFoundError(f"Predictor not found: {spec_str}")
    module_name, attribute = spec_str.split(":", 1)
    module = importlib.import_module(module_name)
    target = getattr(module, attribute)
    return target() if isinstance(target, type) else target

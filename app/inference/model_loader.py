from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"

EXPECTED_ARTIFACTS = [
    "hybrid_rf_model.pkl",
    "hybrid_xgb_model.json",
    "hybrid_meta_model.pkl",
    "gru_model.pt",
    "sequence_scaler.pkl",
    "feature_list.json",
    "drift_detector.pkl",
    "sequence_imputer.pkl",
    "hybrid_static_imputer.pkl",
    "crpi_scalers.pkl",
    "linucb_policy.pkl",
]


class BehaviourGRU(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, sequence: Tensor, lengths: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        packed = pack_padded_sequence(
            sequence,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_output, hidden = self.gru(packed)
        hidden_states, _ = pad_packed_sequence(packed_output, batch_first=True)
        final_embedding = hidden[-1]
        logit = self.classifier(final_embedding).squeeze(1)
        return hidden_states, final_embedding, logit


def discover_model_artifacts() -> dict[str, list[str]]:
    return {
        artifact: [str(MODELS_DIR / artifact)]
        if (MODELS_DIR / artifact).is_file()
        else []
        for artifact in EXPECTED_ARTIFACTS
    }


def get_missing_artifacts() -> list[str]:
    return [
        artifact
        for artifact, paths in discover_model_artifacts().items()
        if not paths
    ]


def artifact_status_message() -> str:
    missing = get_missing_artifacts()
    if not missing:
        return "All trained inference artifacts were found."
    details = "\n".join(f"- Missing artifact: {name}" for name in missing)
    return (
        "Trained inference artifacts are unavailable; trained-model inference is disabled.\n"
        + details
    )


def load_trained_artifacts() -> dict[str, Any]:
    missing = get_missing_artifacts()
    if missing:
        raise FileNotFoundError(artifact_status_message())

    with (MODELS_DIR / "feature_list.json").open("r", encoding="utf-8") as handle:
        feature_list = json.load(handle)

    rf_model = joblib.load(MODELS_DIR / "hybrid_rf_model.pkl")
    xgb_model = XGBClassifier()
    xgb_model.load_model(str(MODELS_DIR / "hybrid_xgb_model.json"))
    meta_model = joblib.load(MODELS_DIR / "hybrid_meta_model.pkl")
    gru_checkpoint = torch.load(
        MODELS_DIR / "gru_model.pt",
        map_location="cpu",
        weights_only=True,
    )
    gru_model = BehaviourGRU(
        input_size=int(gru_checkpoint["input_size"]),
        hidden_size=int(gru_checkpoint["hidden_size"]),
        num_layers=int(gru_checkpoint["num_layers"]),
    )
    gru_model.load_state_dict(gru_checkpoint["state_dict"], strict=True)
    gru_model.eval()

    sequence_scaler = joblib.load(MODELS_DIR / "sequence_scaler.pkl")
    sequence_imputer = joblib.load(MODELS_DIR / "sequence_imputer.pkl")
    static_imputer = joblib.load(MODELS_DIR / "hybrid_static_imputer.pkl")
    drift_config = joblib.load(MODELS_DIR / "drift_detector.pkl")
    crpi_scalers = joblib.load(MODELS_DIR / "crpi_scalers.pkl")
    linucb_policy = joblib.load(MODELS_DIR / "linucb_policy.pkl")

    static_features = feature_list["static_features"]
    sequence_features = feature_list["sequence_features"]
    fusion_features = feature_list["fusion_features"]
    context_features = feature_list["bandit_context_features"]

    if len(static_features) != 10:
        raise ValueError(f"Expected 10 trained static features, found {len(static_features)}.")
    if len(sequence_features) != 8:
        raise ValueError(f"Expected 8 trained sequence features, found {len(sequence_features)}.")
    if len(fusion_features) != 66:
        raise ValueError(f"Expected 66 trained fusion features, found {len(fusion_features)}.")
    if rf_model.n_features_in_ != len(static_features):
        raise ValueError("Random Forest input dimension does not match feature_list.json.")
    if xgb_model.n_features_in_ != len(static_features):
        raise ValueError("XGBoost input dimension does not match feature_list.json.")
    if meta_model.n_features_in_ != len(fusion_features):
        raise ValueError("Meta-learner input dimension does not match feature_list.json.")
    if gru_model.gru.input_size != len(sequence_features):
        raise ValueError("GRU input dimension does not match feature_list.json.")
    if sequence_scaler.n_features_in_ != len(sequence_features):
        raise ValueError("Sequence scaler dimension does not match feature_list.json.")
    if static_imputer.statistics_.shape[0] != len(static_features):
        raise ValueError("Static imputer dimension does not match feature_list.json.")
    if sequence_imputer.statistics_.shape[0] != len(sequence_features):
        raise ValueError("Sequence imputer dimension does not match feature_list.json.")
    if len(context_features) != linucb_policy["n_features"]:
        raise ValueError("LinUCB context dimension does not match feature_list.json.")
    if linucb_policy["context_scaler"].n_features_in_ != len(context_features):
        raise ValueError("LinUCB context scaler dimension does not match feature_list.json.")

    return {
        "rf_model": rf_model,
        "xgb_model": xgb_model,
        "meta_model": meta_model,
        "gru_model": gru_model,
        "sequence_scaler": sequence_scaler,
        "sequence_imputer": sequence_imputer,
        "static_imputer": static_imputer,
        "feature_list": feature_list,
        "drift_config": drift_config,
        "crpi_scalers": crpi_scalers,
        "linucb_policy": linucb_policy,
    }

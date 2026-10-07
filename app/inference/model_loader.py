from __future__ import annotations

from pathlib import Path

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
]


def discover_model_artifacts() -> dict:
    findings = {}
    for artifact in EXPECTED_ARTIFACTS:
        matches = list(PROJECT_ROOT.rglob(artifact)) + list(PROJECT_ROOT.rglob(artifact.replace(".pkl", ".joblib")))
        matches = [str(path) for path in matches if path.exists()]
        findings[artifact] = matches
    return findings


def get_missing_artifacts() -> list[str]:
    artifacts = discover_model_artifacts()
    missing = []
    for artifact, paths in artifacts.items():
        if not paths:
            missing.append(artifact)
    return missing


def artifact_status_message() -> str:
    missing = get_missing_artifacts()
    if not missing:
        return "All expected trained artifacts were found in the workspace."

    details = "\n".join(
        f"- Missing artifact: {name} (expected under the project models directory or a matching saved model path)"
        for name in missing
    )
    return (
        "The trained model artifacts are not present in this workspace. "
        "The demo UI will run in demo-fallback mode instead of loading a replacement model.\n"
        + details
    )

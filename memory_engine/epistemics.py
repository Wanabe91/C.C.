from __future__ import annotations

from typing import Any

VERIFICATION_STATUSES = (
    "unverified",
    "self_reported",
    "logically_consistent",
    "user_confirmed",
    "externally_confirmed",
    "contradicted",
)

_STATUS_STRENGTH = {
    "contradicted": -1,
    "unverified": 0,
    "self_reported": 1,
    "logically_consistent": 2,
    "user_confirmed": 3,
    "externally_confirmed": 4,
}

_STATUS_CONFIDENCE_DEFAULTS = {
    "unverified": 0.20,
    "self_reported": 0.55,
    "logically_consistent": 0.65,
    "user_confirmed": 0.80,
    "externally_confirmed": 0.95,
    "contradicted": 0.05,
}

VERIFICATION_METHOD_TO_STATUS = {
    "user_confirmed": "user_confirmed",
    "external_match": "externally_confirmed",
    "logical_consistency": "logically_consistent",
    "contradicted": "contradicted",
}


def normalize_verification_status(value: Any) -> str:
    normalized = str(value or "").strip().lower()
    if normalized in VERIFICATION_STATUSES:
        return normalized
    return "unverified"


def normalize_confidence_score(value: Any, *, fallback: float | None = None) -> float:
    if value is None and fallback is not None:
        value = fallback
    try:
        normalized = float(value)
    except (TypeError, ValueError):
        normalized = 0.0 if fallback is None else float(fallback)
    if normalized < 0.0:
        return 0.0
    if normalized > 1.0:
        return 1.0
    return normalized


def default_confidence_for_status(status: str) -> float:
    normalized = normalize_verification_status(status)
    return _STATUS_CONFIDENCE_DEFAULTS[normalized]


def stronger_verification_status(left: str, right: str) -> str:
    normalized_left = normalize_verification_status(left)
    normalized_right = normalize_verification_status(right)
    if normalized_left == "contradicted":
        return normalized_right
    if normalized_right == "contradicted":
        return normalized_left
    if _STATUS_STRENGTH[normalized_right] > _STATUS_STRENGTH[normalized_left]:
        return normalized_right
    return normalized_left


def weaker_verification_status(left: str, right: str) -> str:
    normalized_left = normalize_verification_status(left)
    normalized_right = normalize_verification_status(right)
    if "contradicted" in {normalized_left, normalized_right}:
        return "contradicted"
    if _STATUS_STRENGTH[normalized_right] < _STATUS_STRENGTH[normalized_left]:
        return normalized_right
    return normalized_left


def remembered_fact_epistemics(meta: dict[str, Any] | None = None) -> tuple[str, float, list[dict[str, Any]]]:
    meta = meta if isinstance(meta, dict) else {}
    status = normalize_verification_status(meta.get("verification_status") or "self_reported")
    confidence = normalize_confidence_score(
        meta.get("confidence_score"),
        fallback=default_confidence_for_status(status),
    )
    evidence = [
        {
            "kind": "capture",
            "method": "user_statement",
            "source": "user",
            "note": "Captured from an explicit remember_fact request.",
        }
    ]
    return status, confidence, evidence


def epistemic_label(status: str, confidence_score: float) -> str:
    normalized_status = normalize_verification_status(status).replace("_", " ")
    return f"{normalized_status} {normalize_confidence_score(confidence_score):.2f}"

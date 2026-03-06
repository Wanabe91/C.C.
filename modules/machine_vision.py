from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

try:
    from ultralytics import YOLO
except ImportError:  # pragma: no cover - dependency availability depends on the local environment.
    YOLO = None


def _coerce_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return default


def _coerce_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _coerce_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


@dataclass(slots=True)
class Detection:
    label: str
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]


@dataclass(slots=True)
class MachineVisionResult:
    enabled: bool
    model_name: str
    summary: str
    detections: tuple[Detection, ...] = ()


class MachineVisionModule:
    def __init__(self, config: dict[str, Any] | None = None):
        cfg = config or {}
        self.model_name = str(cfg.get("model") or "").strip()
        self.requested = _coerce_bool(
            cfg.get("machine_vision_enabled"),
            default=bool(self.model_name),
        )
        self.enabled = self.requested and YOLO is not None
        self.confidence = min(
            1.0,
            max(0.01, _coerce_float(cfg.get("machine_vision_confidence"), 0.25)),
        )
        self.max_detections = max(1, _coerce_int(cfg.get("machine_vision_max_detections"), 10))
        self._model: Any | None = None

    def _ensure_model(self) -> Any:
        if not self.enabled:
            return None
        if self._model is None:
            self._model = YOLO(self.model_name)
        return self._model

    def analyze_frame(self, frame: Any) -> MachineVisionResult:
        if not self.enabled:
            return MachineVisionResult(
                enabled=False,
                model_name=self.model_name,
                summary="",
                detections=(),
            )

        try:
            model = self._ensure_model()
            prediction = model.predict(
                source=frame,
                conf=self.confidence,
                max_det=self.max_detections,
                verbose=False,
            )[0]
        except Exception:
            return MachineVisionResult(
                enabled=False,
                model_name=self.model_name,
                summary="",
                detections=(),
            )

        names = prediction.names
        raw_boxes = getattr(prediction, "boxes", None)
        detections: list[Detection] = []
        if raw_boxes is not None:
            total_boxes = len(raw_boxes)
            for index in range(min(total_boxes, self.max_detections)):
                box = raw_boxes[index]
                cls_index = int(box.cls[0].item())
                if isinstance(names, dict):
                    label_value = names.get(cls_index, cls_index)
                elif isinstance(names, list) and 0 <= cls_index < len(names):
                    label_value = names[cls_index]
                else:
                    label_value = cls_index
                label = str(label_value)
                confidence = float(box.conf[0].item())
                xyxy_raw = box.xyxy[0].tolist()
                xyxy = tuple(float(value) for value in xyxy_raw[:4])
                detections.append(
                    Detection(
                        label=label,
                        confidence=confidence,
                        bbox_xyxy=xyxy,
                    )
                )

        return MachineVisionResult(
            enabled=True,
            model_name=self.model_name,
            summary=self._format_summary(detections),
            detections=tuple(detections),
        )

    def _format_summary(self, detections: list[Detection]) -> str:
        if not detections:
            return (
                "Machine vision observations:\n"
                f"- Model: {self.model_name}\n"
                "- No objects detected above the configured confidence threshold."
            )

        counts = Counter(detection.label for detection in detections)
        count_summary = ", ".join(
            f"{label} x{count}" for label, count in counts.most_common()
        )
        detailed_lines = [
            (
                f"- {detection.label} "
                f"(confidence={detection.confidence:.2f}, "
                f"bbox={tuple(round(value, 1) for value in detection.bbox_xyxy)})"
            )
            for detection in detections
        ]
        return "\n".join(
            [
                "Machine vision observations:",
                f"- Model: {self.model_name}",
                f"- Detected objects: {count_summary}",
                "- Detection details:",
                *detailed_lines,
            ]
        )

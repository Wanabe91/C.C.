from __future__ import annotations

import asyncio
import base64
import sys
import threading
from dataclasses import dataclass
from typing import Any

from memory_engine.llm import LLMRequest, llm_call
from memory_engine.router import TaskType
from modules.machine_vision import MachineVisionModule, MachineVisionResult

try:
    import cv2
except ImportError:  # pragma: no cover - dependency availability depends on the local environment.
    cv2 = None


def _coerce_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _coerce_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


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


@dataclass(slots=True)
class VisionResult:
    prompt: str
    response: str
    machine_vision_summary: str = ""
    mime_type: str = "image/jpeg"


@dataclass(slots=True)
class CapturedFrame:
    image_data_url: str
    machine_vision_result: MachineVisionResult
    mime_type: str = "image/jpeg"


@dataclass(slots=True)
class LiveFrame:
    frame: Any
    machine_vision_result: MachineVisionResult
    mime_type: str = "image/jpeg"


@dataclass(slots=True)
class VisionStreamUpdate:
    machine_vision_summary: str
    machine_vision_compact: str
    should_stop: bool = False


class VisionModule:
    def __init__(self, config: dict[str, Any] | None = None):
        if cv2 is None:
            raise RuntimeError("opencv-python is not installed. Add it to the environment to use vision mode.")

        cfg = config or {}
        self.camera_index = _coerce_int(cfg.get("camera_index"), 0)
        self.jpeg_quality = min(100, max(50, _coerce_int(cfg.get("jpeg_quality"), 90)))
        self.max_tokens = max(256, _coerce_int(cfg.get("max_tokens"), 700))
        self.warmup_frames = max(1, _coerce_int(cfg.get("warmup_frames"), 2))
        self.frame_interval_ms = max(1, _coerce_int(cfg.get("frame_interval"), 30))
        self.stream_llm_interval_sec = max(0.5, _coerce_float(cfg.get("stream_llm_interval_sec"), 3.0))
        self.stream_status_interval_sec = max(0.1, _coerce_float(cfg.get("stream_status_interval_sec"), 0.5))
        self.stream_show_window = _coerce_bool(cfg.get("stream_show_window"), True)
        self.stream_window_name = (
            str(cfg.get("stream_window_name") or "Vision Stream").strip()
            or "Vision Stream"
        )
        self.analysis_prompt = (
            str(cfg.get("analysis_prompt") or "Describe what is visible in the camera frame.")
            .strip()
            or "Describe what is visible in the camera frame."
        )
        self.camera_backend = str(cfg.get("camera_backend") or "").strip().lower()
        self.machine_vision = MachineVisionModule(cfg)
        self._capture: Any | None = None
        self._capture_lock = threading.Lock()
        self._window_available = self.stream_show_window

    def _resolve_backend(self) -> int | None:
        if cv2 is None:
            return None
        backend_map = {
            "any": getattr(cv2, "CAP_ANY", 0),
            "dshow": getattr(cv2, "CAP_DSHOW", None),
            "msmf": getattr(cv2, "CAP_MSMF", None),
        }
        if self.camera_backend:
            return backend_map.get(self.camera_backend)
        if sys.platform == "win32":
            return getattr(cv2, "CAP_DSHOW", None)
        return None

    def _open_camera(self) -> Any:
        backend = self._resolve_backend()
        capture = cv2.VideoCapture(self.camera_index, backend) if backend is not None else cv2.VideoCapture(self.camera_index)
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(
                f"Camera {self.camera_index} is unavailable. Check vision.camera_index and camera permissions."
            )
        return capture

    def _ensure_capture(self) -> Any:
        if self._capture is not None and self._capture.isOpened():
            return self._capture
        self._capture = self._open_camera()
        return self._capture

    def _release_capture(self) -> None:
        if self._capture is None:
            return
        self._capture.release()
        self._capture = None

    def _read_frame(self) -> Any:
        capture = self._ensure_capture()
        frame = None
        ok = False

        for _ in range(self.warmup_frames):
            ok, frame = capture.read()

        if not ok or frame is None:
            self._release_capture()
            capture = self._ensure_capture()
            ok, frame = capture.read()
        if not ok or frame is None:
            raise RuntimeError("Failed to capture a frame from the camera.")
        return frame

    def _encode_frame_data_url(self, frame: Any) -> str:
        encoded_ok, encoded = cv2.imencode(
            ".jpg",
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
        )
        if not encoded_ok:
            raise RuntimeError("Failed to encode the captured frame.")
        image_bytes = encoded.tobytes()
        image_b64 = base64.b64encode(image_bytes).decode("ascii")
        return f"data:image/jpeg;base64,{image_b64}"

    def _capture_frame(self) -> CapturedFrame:
        with self._capture_lock:
            frame = self._read_frame()
        machine_vision_result = self.machine_vision.analyze_frame(frame)
        return CapturedFrame(
            image_data_url=self._encode_frame_data_url(frame),
            machine_vision_result=machine_vision_result,
        )

    def _capture_live_frame(self) -> LiveFrame:
        with self._capture_lock:
            frame = self._read_frame()
        return LiveFrame(
            frame=frame,
            machine_vision_result=self.machine_vision.analyze_frame(frame),
        )

    def _compact_machine_vision(self, result: MachineVisionResult) -> str:
        if result.detections:
            counts: dict[str, int] = {}
            for detection in result.detections:
                counts[detection.label] = counts.get(detection.label, 0) + 1
            return ", ".join(f"{label} x{count}" for label, count in counts.items())
        if self.machine_vision.requested and not result.enabled:
            return "machine vision unavailable; GPT-4o sampling only"
        if result.enabled:
            return "no objects detected"
        return ""

    def _annotate_preview_frame(self, live_frame: LiveFrame) -> Any:
        frame = live_frame.frame.copy()
        compact_summary = self._compact_machine_vision(live_frame.machine_vision_result) or "camera active"
        for detection in live_frame.machine_vision_result.detections:
            x1, y1, x2, y2 = (int(value) for value in detection.bbox_xyxy)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (40, 200, 40), 2)
            cv2.putText(
                frame,
                f"{detection.label} {detection.confidence:.2f}",
                (x1, max(24, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (40, 200, 40),
                2,
                cv2.LINE_AA,
            )
        cv2.putText(
            frame,
            compact_summary[:90],
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            "Press q to stop stream",
            (12, max(40, frame.shape[0] - 14)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return frame

    def _render_preview(self, live_frame: LiveFrame) -> bool:
        if not self._window_available:
            return False
        try:
            cv2.imshow(self.stream_window_name, self._annotate_preview_frame(live_frame))
            key = cv2.waitKey(1) & 0xFF
            return key in {ord("q"), 27}
        except Exception:
            self._window_available = False
            return False

    def _analyze_frame_sync(self, prompt: str, captured_frame: CapturedFrame) -> VisionResult:
        machine_vision_summary = captured_frame.machine_vision_result.summary.strip()
        request = LLMRequest(
            goal=prompt,
            context_snapshot=machine_vision_summary,
            tool_registry_block=(
                "You are a vision assistant. Describe only what is supported by the image. "
                "If machine vision observations are provided, treat them as supporting hints and "
                "prefer the image when there is any disagreement."
            ),
            image_data_urls=(captured_frame.image_data_url,),
        )
        response = llm_call(
            request,
            task=TaskType.VISION,
            has_image=True,
            max_tokens=self.max_tokens,
        )
        return VisionResult(
            prompt=prompt,
            response=response,
            machine_vision_summary=machine_vision_summary,
            mime_type=captured_frame.mime_type,
        )

    async def capture_and_describe(self, prompt: str | None = None) -> VisionResult:
        final_prompt = str(prompt or "").strip() or self.analysis_prompt
        captured_frame = await asyncio.to_thread(self._capture_frame)
        return await asyncio.to_thread(self._analyze_frame_sync, final_prompt, captured_frame)

    async def stream_step(self) -> VisionStreamUpdate:
        live_frame = await asyncio.to_thread(self._capture_live_frame)
        return VisionStreamUpdate(
            machine_vision_summary=live_frame.machine_vision_result.summary.strip(),
            machine_vision_compact=self._compact_machine_vision(live_frame.machine_vision_result),
            should_stop=self._render_preview(live_frame),
        )

    async def close(self) -> None:
        if cv2 is not None and self._window_available:
            try:
                cv2.destroyWindow(self.stream_window_name)
            except Exception:
                pass
        await asyncio.to_thread(self._release_capture)

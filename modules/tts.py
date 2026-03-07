from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from elevenlabs.client import ElevenLabs
except ImportError:  # pragma: no cover - dependency availability depends on the local environment.
    ElevenLabs = None


@dataclass(slots=True)
class SpeechSynthesisResult:
    audio_data: bytes
    headers: dict[str, str]
    voice_id: str
    output_format: str
    request_id: str | None = None
    character_count: int | None = None
    file_path: Path | None = None


def _parse_optional_int(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


class TextToSpeech:
    def __init__(self, config: dict[str, Any] | None = None):
        cfg = config or {}
        if ElevenLabs is None:
            raise RuntimeError(
                "elevenlabs SDK is not installed. Install it or disable tts.enabled."
            )
        api_key = str(cfg.get("api_key") or os.getenv("ELEVENLABS_API_KEY") or "").strip()
        if not api_key:
            raise RuntimeError("ELEVENLABS_API_KEY is not set.")

        self.voice_id = str(cfg.get("voice_id") or os.getenv("ELEVENLABS_VOICE_ID") or "").strip()
        if not self.voice_id:
            raise RuntimeError("ELEVENLABS_VOICE_ID is not set.")

        self.model_id = str(cfg.get("model_id") or "eleven_multilingual_v2").strip()
        self.output_format = str(cfg.get("output_format") or "wav_22050").strip()
        self.language_code = str(cfg.get("language_code") or "").strip() or None
        self.playback = bool(cfg.get("playback", True))
        self.save_audio = bool(cfg.get("save_audio", True))
        self.save_dir = Path(str(cfg.get("save_dir") or "data/tts")).expanduser().resolve()
        self.client = ElevenLabs(api_key=api_key)
        self.last_result: SpeechSynthesisResult | None = None

    def _build_request_kwargs(self, text: str) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "voice_id": self.voice_id,
            "text": text,
            "output_format": self.output_format,
        }
        if self.model_id:
            kwargs["model_id"] = self.model_id
        if self.language_code:
            kwargs["language_code"] = self.language_code
        return kwargs

    def _file_suffix(self) -> str:
        prefix = self.output_format.split("_", 1)[0].lower()
        if prefix in {"wav", "mp3", "opus", "pcm"}:
            return f".{prefix}"
        return ".bin"

    def _save_audio_file(self, audio_data: bytes, request_id: str | None) -> Path | None:
        if not self.save_audio:
            return None
        self.save_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        stem = request_id or f"tts_{stamp}"
        path = self.save_dir / f"{stem}{self._file_suffix()}"
        path.write_bytes(audio_data)
        return path

    def _synthesize_sync(self, text: str) -> SpeechSynthesisResult:
        with self.client.text_to_speech.with_raw_response.convert(
            **self._build_request_kwargs(text)
        ) as response:
            headers = dict(response.headers)
            audio_data = b"".join(response.data)

        request_id = headers.get("request-id") or headers.get("x-request-id")
        result = SpeechSynthesisResult(
            audio_data=audio_data,
            headers=headers,
            voice_id=self.voice_id,
            output_format=self.output_format,
            request_id=request_id,
            character_count=_parse_optional_int(headers.get("x-character-count")),
        )
        result.file_path = self._save_audio_file(audio_data, request_id)
        return result

    def _materialize_playback_file(self, result: SpeechSynthesisResult) -> tuple[Path, bool]:
        if result.file_path is not None and result.file_path.exists():
            return result.file_path, False

        suffix = self._file_suffix()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(result.audio_data)
            return Path(tmp.name), True

    @staticmethod
    def _run_player(commands: list[list[str]], error_message: str) -> None:
        for command in commands:
            if shutil.which(command[0]) is None:
                continue
            completed = subprocess.run(
                command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            if completed.returncode == 0:
                return

        raise RuntimeError(error_message)

    def _play_on_linux(self, path: Path, output_format: str) -> None:
        suffix = path.suffix.lower()
        commands: list[list[str]] = []

        if suffix == ".wav":
            commands.extend(
                [
                    ["aplay", "-q", str(path)],
                    ["paplay", str(path)],
                ]
            )
        elif suffix == ".pcm":
            sample_rate = output_format.split("_", 1)[1] if "_" in output_format else "22050"
            commands.extend(
                [
                    ["aplay", "-q", "-t", "raw", "-f", "S16_LE", "-r", sample_rate, "-c", "1", str(path)],
                    [
                        "ffplay",
                        "-f",
                        "s16le",
                        "-ar",
                        sample_rate,
                        "-ac",
                        "1",
                        "-nodisp",
                        "-autoexit",
                        "-loglevel",
                        "error",
                        str(path),
                    ],
                ]
            )
        elif suffix == ".mp3":
            commands.append(["mpg123", "-q", str(path)])

        commands.extend(
            [
                ["ffplay", "-nodisp", "-autoexit", "-loglevel", "error", str(path)],
                ["mpv", "--no-video", "--really-quiet", str(path)],
                ["cvlc", "--play-and-exit", "--intf", "dummy", str(path)],
            ]
        )

        self._run_player(
            commands,
            "Linux audio playback requires one of: aplay, paplay, ffplay, mpv, cvlc, or mpg123.",
        )

    def _play_on_macos(self, path: Path) -> None:
        self._run_player(
            [
                ["afplay", str(path)],
                ["ffplay", "-nodisp", "-autoexit", "-loglevel", "error", str(path)],
                ["mpv", "--no-video", "--really-quiet", str(path)],
            ],
            "macOS audio playback requires one of: afplay, ffplay, or mpv.",
        )

    def _play_audio_sync(self, result: SpeechSynthesisResult) -> None:
        if not self.playback:
            return

        if sys.platform == "win32" and result.output_format.startswith("wav_"):
            import winsound

            winsound.PlaySound(result.audio_data, winsound.SND_MEMORY)
            return

        path, is_temporary = self._materialize_playback_file(result)
        try:
            if sys.platform.startswith("linux"):
                self._play_on_linux(path, result.output_format)
                return
            if sys.platform == "darwin":
                self._play_on_macos(path)
                return
            raise RuntimeError(f"Audio playback is not supported on platform: {sys.platform}")
        finally:
            if is_temporary:
                path.unlink(missing_ok=True)

    async def synthesize(self, text: str) -> SpeechSynthesisResult:
        result = await asyncio.to_thread(self._synthesize_sync, text)
        self.last_result = result
        return result

    async def speak(self, text: str) -> SpeechSynthesisResult:
        result = await self.synthesize(text)
        if self.playback:
            await asyncio.to_thread(self._play_audio_sync, result)
        return result

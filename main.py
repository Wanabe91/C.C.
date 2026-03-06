import argparse
import asyncio
import time

import yaml
from memory_engine.hf_auth import configure_huggingface_auth

configure_huggingface_auth()

# from utils.event_bus import EventBus
# from modules.stt import SpeechRecognizer
from modules.tts import TextToSpeech
from modules.planner import Planner
from modules.vision import VisionModule


class PersonalAI:
    def __init__(self, config):
        self.config = config if isinstance(config, dict) else {}
        self.planner = Planner(config)

        # Text mode works without optional voice/vision modules.
        self.tts = self._build_tts(config)
        self.stt = None
        self.vision = None

    @staticmethod
    def _build_tts(config):
        tts_cfg = config.get("tts", {}) if isinstance(config, dict) else {}
        if not isinstance(tts_cfg, dict):
            return None
        if not bool(tts_cfg.get("enabled", False)):
            return None
        if str(tts_cfg.get("engine") or "").strip().lower() != "elevenlabs":
            return None
        try:
            return TextToSpeech(tts_cfg)
        except Exception as exc:
            print(f"TTS disabled: {exc}")
            return None

    async def _ensure_vision(self):
        if self.vision is not None:
            return self.vision

        vision_cfg = self.config.get("vision", {})
        if not isinstance(vision_cfg, dict) or not bool(vision_cfg.get("enabled", False)):
            raise RuntimeError("Vision mode is disabled in config.yaml. Set vision.enabled to true.")
        self.vision = VisionModule(vision_cfg)
        return self.vision

    async def _on_text(self, text):
        response = await self.planner.run(text)
        await self._speak_text(response)
        return response

    async def _on_voice(self, _audio):
        raise RuntimeError("Voice mode is unavailable in this repository.")

    async def _on_frame(self, prompt):
        vision = await self._ensure_vision()
        result = await vision.capture_and_describe(prompt)
        await self._speak_text(result.response)
        return result.response

    async def _speak_text(self, text):
        if self.tts is None:
            return
        tts_result = await self.tts.speak(text)
        meta = []
        if tts_result.character_count is not None:
            meta.append(f"chars={tts_result.character_count}")
        if tts_result.request_id:
            meta.append(f"request_id={tts_result.request_id}")
        if meta:
            print(f"[TTS] {' '.join(meta)}")

    @staticmethod
    async def _read_input(prompt: str) -> str | None:
        try:
            return (await asyncio.to_thread(input, prompt)).strip()
        except EOFError:
            return None

    async def close(self):
        if self.vision is not None:
            await self.vision.close()
        await self.planner.close()

    async def text_mode(self):
        while True:
            inp = await self._read_input("You: ")
            if inp is None:
                break
            if not inp:
                continue
            if inp.lower() in ("exit", "\u0432\u044b\u0445\u043e\u0434"):
                break
            try:
                print(f"AI: {await self._on_text(inp)}\n")
            except Exception as exc:
                print(f"Error: {exc}\n")

    async def vision_mode(self):
        print("Vision mode: type a prompt for the current camera frame and press Enter.")
        print("Press Enter on an empty line to use the default prompt. Type 'exit' to quit.\n")
        while True:
            inp = await self._read_input("Vision: ")
            if inp is None:
                break
            if inp.lower() in ("exit", "\u0432\u044b\u0445\u043e\u0434"):
                break
            try:
                print(f"AI: {await self._on_frame(inp)}\n")
            except Exception as exc:
                print(f"Error: {exc}\n")

    async def vision_stream_mode(self):
        vision = await self._ensure_vision()
        prompt = await self._read_input("Stream prompt (blank = default): ")
        if prompt is None:
            return
        print("Vision stream mode: local machine vision runs continuously, GPT-4o samples frames periodically.")
        if vision.stream_show_window:
            print("Close the preview window with 'q' or stop the console process to exit.\n")
        else:
            print("Stop the console process to exit.\n")

        pending_llm_task = None
        last_llm_started_at = 0.0
        last_status_at = 0.0
        last_status = ""
        try:
            while True:
                now = time.monotonic()
                if pending_llm_task is None and now - last_llm_started_at >= vision.stream_llm_interval_sec:
                    pending_llm_task = asyncio.create_task(vision.capture_and_describe(prompt))
                    last_llm_started_at = now

                update = await vision.stream_step()
                if update.should_stop:
                    break

                if (
                    update.machine_vision_compact
                    and update.machine_vision_compact != last_status
                    and now - last_status_at >= vision.stream_status_interval_sec
                ):
                    print(f"[CV] {update.machine_vision_compact}")
                    last_status = update.machine_vision_compact
                    last_status_at = now

                if pending_llm_task is not None and pending_llm_task.done():
                    try:
                        result = pending_llm_task.result()
                    except Exception as exc:
                        print(f"[GPT-4o] Error: {exc}")
                    else:
                        print(f"[GPT-4o] {result.response}\n")
                        await self._speak_text(result.response)
                    pending_llm_task = None

                await asyncio.sleep(vision.frame_interval_ms / 1000)
        finally:
            if pending_llm_task is not None:
                pending_llm_task.cancel()
                await asyncio.gather(pending_llm_task, return_exceptions=True)


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["text", "voice", "vision", "vision-stream"], default="text")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    ai = PersonalAI(cfg)
    try:
        if args.mode == "text":
            await ai.text_mode()
            return
        if args.mode == "vision":
            await ai.vision_mode()
            return
        if args.mode == "vision-stream":
            await ai.vision_stream_mode()
            return

        raise SystemExit(
            f"{args.mode} mode is unavailable in this repository. "
            "Use --mode text, --mode vision, or --mode vision-stream."
        )
    finally:
        await ai.close()


if __name__ == "__main__":
    asyncio.run(main())

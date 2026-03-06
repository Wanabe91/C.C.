import argparse
import asyncio
import yaml
from memory_engine.hf_auth import configure_huggingface_auth

configure_huggingface_auth()

# from utils.event_bus import EventBus
# from modules.stt import SpeechRecognizer
# from modules.tts import TextToSpeech
# from modules.vision import VisionModule
from modules.planner import Planner


class PersonalAI:
    def __init__(self, config):
        self.planner = Planner(config)

        # Text mode works without optional voice/vision modules.
        self.tts = None
        self.stt = None
        self.vision = None

    async def _on_text(self, text):
        response = await self.planner.run(text)
        if self.tts is not None:
            await self.tts.speak(response)
        return response

    async def _on_voice(self, _audio):
        raise RuntimeError("Voice mode is unavailable in this repository.")

    async def _on_frame(self, _frame):
        raise RuntimeError("Vision mode is unavailable in this repository.")

    async def close(self):
        await self.planner.close()

    async def text_mode(self):
        while True:
            inp = input("You: ").strip()
            if not inp:
                continue
            if inp.lower() in ("exit", "\u0432\u044b\u0445\u043e\u0434"):
                break
            try:
                print(f"AI: {await self._on_text(inp)}\n")
            except Exception as exc:
                print(f"Error: {exc}\n")


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["text", "voice", "vision"], default="text")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    ai = PersonalAI(cfg)
    try:
        if args.mode == "text":
            await ai.text_mode()
            return

        raise SystemExit(f"{args.mode} mode is unavailable in this repository. Use --mode text.")
    finally:
        await ai.close()


if __name__ == "__main__":
    asyncio.run(main())

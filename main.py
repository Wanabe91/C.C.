import asyncio, argparse, yaml
from utils.event_bus import EventBus
from modules.llm_client import LLMClient
from modules.stt import SpeechRecognizer
from modules.tts import TextToSpeech
from modules.vision import VisionModule
from modules.planner import Planner
from modules.memory import Memory

class PersonalAI:
    def __init__(self, config):
        self.bus     = EventBus()
        self.llm     = LLMClient(config["llm"])
        self.memory  = Memory(config.get("memory", {}))
        self.planner = Planner(self.llm, self.memory)
        self.tts     = TextToSpeech(config.get("tts", {}))
        self.stt     = SpeechRecognizer(config.get("stt", {}))
        self.vision  = VisionModule(config.get("vision", {}))

        self.bus.subscribe("user_text",  self._on_text)
        self.bus.subscribe("user_voice", self._on_voice)
        self.bus.subscribe("frame",      self._on_frame)

    async def _on_text(self, text):
        response = await self.planner.run(text)
        await self.tts.speak(response)
        return response

    async def _on_voice(self, audio):
        text = await self.stt.transcribe(audio)
        if text: await self.bus.emit("user_text", text)

    async def _on_frame(self, frame):
        desc = await self.vision.describe(frame)
        if desc: await self.bus.emit("user_text", f"[Вижу]: {desc}")

    async def text_mode(self):
        while True:
            inp = input("Вы: ").strip()
            if inp.lower() in ("exit","выход"): break
            print(f"AI: {await self._on_text(inp)}\n")

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["text","voice","vision"], default="text")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    with open(args.config) as f: cfg = yaml.safe_load(f)
    ai = PersonalAI(cfg)
    if args.mode == "text": await ai.text_mode()

if __name__ == "__main__":
    asyncio.run(main())
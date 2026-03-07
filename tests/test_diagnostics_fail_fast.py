from __future__ import annotations

import importlib
import sys
import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch


def _import_main_with_fakes():
    fake_tts = ModuleType("modules.tts")
    fake_tts.TextToSpeech = type("TextToSpeech", (), {})

    fake_planner = ModuleType("modules.planner")
    fake_planner.Planner = type("Planner", (), {})

    fake_vision = ModuleType("modules.vision")
    fake_vision.VisionModule = type("VisionModule", (), {})

    with patch.dict(
        sys.modules,
        {
            "modules.tts": fake_tts,
            "modules.planner": fake_planner,
            "modules.vision": fake_vision,
        },
    ):
        sys.modules.pop("main", None)
        return importlib.import_module("main")


class DiagnosticsFailFastTests(unittest.IsolatedAsyncioTestCase):
    def test_build_tts_raises_instead_of_silently_disabling(self) -> None:
        main_module = _import_main_with_fakes()

        with patch.object(main_module, "TextToSpeech", side_effect=ValueError("bad api key")):
            with self.assertRaisesRegex(RuntimeError, "Failed to initialize ElevenLabs TTS"):
                main_module.PersonalAI._build_tts(
                    {
                        "tts": {
                            "enabled": True,
                            "engine": "elevenlabs",
                        }
                    }
                )

    async def test_planner_is_initialized_lazily(self) -> None:
        main_module = _import_main_with_fakes()

        planner_instance = AsyncMock()
        planner_instance.run = AsyncMock(return_value="ok")
        with patch.object(main_module, "Planner", return_value=planner_instance) as planner_cls:
            ai = main_module.PersonalAI({"tts": {"enabled": False}})
            self.assertIsNone(ai.planner)

            response = await ai._on_text("hello")

        self.assertEqual(response, "ok")
        planner_cls.assert_called_once()
        planner_instance.run.assert_awaited_once_with("hello")

    async def test_text_mode_propagates_handler_failures(self) -> None:
        main_module = _import_main_with_fakes()
        ai = main_module.PersonalAI.__new__(main_module.PersonalAI)

        async def fake_read_input(_prompt: str) -> str | None:
            return "hello"

        async def fake_on_text(_text: str) -> str:
            raise RuntimeError("planner exploded")

        ai._read_input = fake_read_input
        ai._on_text = fake_on_text

        with self.assertRaisesRegex(RuntimeError, "planner exploded"):
            await ai.text_mode()

    async def test_vision_stream_mode_propagates_pending_llm_failures(self) -> None:
        main_module = _import_main_with_fakes()
        ai = main_module.PersonalAI.__new__(main_module.PersonalAI)

        class FakeVision:
            stream_show_window = False
            stream_llm_interval_sec = 0.0
            stream_status_interval_sec = 10.0
            frame_interval_ms = 1

            async def capture_and_describe(self, _prompt):
                raise RuntimeError("gpt4o failed")

            async def stream_step(self):
                await main_module.asyncio.sleep(0)
                return SimpleNamespace(
                    should_stop=False,
                    machine_vision_compact="",
                )

        async def fake_read_input(_prompt: str) -> str | None:
            return ""

        ai._ensure_vision = AsyncMock(return_value=FakeVision())
        ai._read_input = fake_read_input
        ai._speak_text = AsyncMock()

        with self.assertRaisesRegex(RuntimeError, "gpt4o failed"):
            await ai.vision_stream_mode()


class MachineVisionDiagnosticsTests(unittest.TestCase):
    def test_machine_vision_inference_failure_is_not_silenced(self) -> None:
        module = importlib.import_module("modules.machine_vision")
        machine_vision = module.MachineVisionModule(
            {
                "model": "yolov8n.pt",
                "machine_vision_enabled": True,
            }
        )

        failing_model = Mock()
        failing_model.predict.side_effect = ValueError("weights corrupted")
        machine_vision._model = failing_model

        with patch.object(module, "YOLO", object()):
            with self.assertRaisesRegex(RuntimeError, "Machine vision inference failed"):
                machine_vision.analyze_frame(frame=object())


class VisionPreviewDiagnosticsTests(unittest.TestCase):
    def test_preview_render_failure_is_not_silenced(self) -> None:
        module = importlib.import_module("modules.vision")
        vision = module.VisionModule.__new__(module.VisionModule)
        vision._window_available = True
        vision.stream_window_name = "Vision Stream"
        vision._annotate_preview_frame = Mock(return_value="frame")

        live_frame = module.LiveFrame(
            frame="frame",
            machine_vision_result=module.MachineVisionResult(
                enabled=True,
                model_name="yolo",
                summary="",
            ),
        )

        fake_cv2 = SimpleNamespace(
            imshow=Mock(side_effect=RuntimeError("preview backend failed")),
            waitKey=Mock(return_value=-1),
        )
        with patch.object(module, "cv2", fake_cv2):
            with self.assertRaisesRegex(RuntimeError, "Vision preview failed"):
                vision._render_preview(live_frame)


class OptionalDependencyTests(unittest.TestCase):
    def test_tts_module_is_import_safe_without_elevenlabs_sdk(self) -> None:
        module = importlib.import_module("modules.tts")

        with patch.object(module, "ElevenLabs", None):
            with self.assertRaisesRegex(RuntimeError, "elevenlabs SDK is not installed"):
                module.TextToSpeech({"api_key": "x", "voice_id": "y"})


if __name__ == "__main__":
    unittest.main()

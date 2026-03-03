from __future__ import annotations

import os

from dotenv import load_dotenv


def configure_huggingface_auth() -> None:
    load_dotenv(override=False)
    token = (os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN") or "").strip()
    if token and not os.getenv("HUGGINGFACE_HUB_TOKEN"):
        os.environ["HUGGINGFACE_HUB_TOKEN"] = token

from __future__ import annotations

import re
from enum import Enum


class TaskType(str, Enum):
    FAST = "fast"
    REASON = "reason"
    MATH = "math"
    VISION = "vision"
    CONSOLIDATE = "consolidate"
    SUMMARIZE = "summarize"
    PLAN = "plan"


LONG_CONTEXT_THRESHOLD = 8_000
NON_UPGRADABLE_BATCH_TASKS = {
    TaskType.CONSOLIDATE,
    TaskType.SUMMARIZE,
}

MATH_KEYWORDS: tuple[str, ...] = (
    "math",
    "solve",
    "proof",
    "prove",
    "theorem",
    "equation",
    "integral",
    "derivative",
    "matrix",
    "vector",
    "algebra",
    "geometry",
    "calculus",
    "probability",
    "statistics",
    "algorithm",
    "complexity",
    "recursion",
    "optimization",
    "матем",
    "докажи",
    "доказ",
    "теорем",
    "уравнен",
    "интеграл",
    "производн",
    "матриц",
    "вектор",
    "алгебр",
    "геометр",
    "вероятност",
    "статист",
    "алгоритм",
    "рекурс",
    "оптимиза",
    "вычисл",
)

ANALYSIS_KEYWORDS: tuple[str, ...] = (
    "analyze",
    "analyse",
    "analysis",
    "compare",
    "explain",
    "reason",
    "critique",
    "strategy",
    "architecture",
    "tradeoff",
    "trade-off",
    "why",
    "проанализ",
    "анализ",
    "сравни",
    "сравнен",
    "объясни",
    "почему",
    "разбери",
    "критик",
    "стратег",
    "архитект",
    "концепц",
)

IMAGE_KEYWORDS: tuple[str, ...] = (
    "image",
    "picture",
    "photo",
    "screenshot",
    "diagram",
    "scan",
    "картин",
    "изображен",
    "скриншот",
    "фото",
    "диаграм",
)

IMAGE_EXTENSION_RE = re.compile(r"\.(?:png|jpe?g|gif|webp|bmp|tiff?|svg)\b", re.IGNORECASE)
NON_WORD_RE = re.compile(r"[^\w\s]+", re.UNICODE)


def route(
    task: TaskType,
    context_len: int = 0,
    has_image: bool = False,
    user_message: str = "",
) -> TaskType:
    if task == TaskType.PLAN:
        return TaskType.PLAN

    normalized_task = task
    raw_message = user_message or ""
    normalized_message = _normalize(user_message)

    if has_image or _contains_image_reference(raw_message, normalized_message):
        return TaskType.VISION

    if _contains_keyword(normalized_message, MATH_KEYWORDS):
        return TaskType.MATH

    if (
        context_len > LONG_CONTEXT_THRESHOLD
        and normalized_task not in NON_UPGRADABLE_BATCH_TASKS
    ):
        return TaskType.REASON

    if (
        normalized_task not in NON_UPGRADABLE_BATCH_TASKS
        and _contains_keyword(normalized_message, ANALYSIS_KEYWORDS)
    ):
        return TaskType.REASON

    return normalized_task


def _normalize(text: str) -> str:
    cleaned = NON_WORD_RE.sub(" ", text or "")
    return re.sub(r"\s+", " ", cleaned).strip().lower()


def _contains_image_reference(raw_text: str, normalized_text: str) -> bool:
    if not raw_text and not normalized_text:
        return False
    if IMAGE_EXTENSION_RE.search(raw_text):
        return True
    return _contains_keyword(normalized_text, IMAGE_KEYWORDS)


def _contains_keyword(text: str, keywords: tuple[str, ...]) -> bool:
    if not text:
        return False
    return any(keyword in text for keyword in keywords)

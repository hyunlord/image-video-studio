from __future__ import annotations

import logging
import re

from backend.models import PromptAnalysis

logger = logging.getLogger(__name__)

# Keyword → parameter adjustment mappings
KEYWORD_ADJUSTMENTS = {
    # Fast motion keywords
    "fast": {"guidance_delta": 1.0, "motion": 0.8},
    "quick": {"guidance_delta": 0.8, "motion": 0.7},
    "rapid": {"guidance_delta": 1.0, "motion": 0.9},
    "sudden": {"guidance_delta": 1.2, "motion": 1.0},
    "burst": {"guidance_delta": 1.0, "motion": 0.8},
    "explosive": {"guidance_delta": 1.5, "motion": 1.0},
    "dynamic": {"guidance_delta": 0.8, "motion": 0.6},
    # Slow motion keywords
    "slow": {"guidance_delta": -0.5, "motion": -0.5},
    "gradual": {"guidance_delta": -0.3, "motion": -0.4},
    "gentle": {"guidance_delta": -0.5, "motion": -0.5},
    "smooth": {"guidance_delta": -0.3, "motion": -0.3},
    "subtle": {"guidance_delta": -1.0, "motion": -0.7},
    # Dramatic keywords
    "dramatic": {"guidance_delta": 1.5, "steps_delta": 20},
    "intense": {"guidance_delta": 1.2, "steps_delta": 15},
    "powerful": {"guidance_delta": 1.0, "steps_delta": 10},
    "extreme": {"guidance_delta": 1.5, "steps_delta": 20},
    # Face-related keywords
    "aging": {"codeformer": True, "fidelity": 0.7},
    "age": {"codeformer": True, "fidelity": 0.7},
    "face": {"codeformer": True, "fidelity": 0.8},
    "portrait": {"codeformer": True, "fidelity": 0.8},
    "wrinkle": {"codeformer": True, "fidelity": 0.6},
    # Transformation keywords
    "transform": {"steps_delta": 10},
    "morph": {"steps_delta": 15},
    "change": {"steps_delta": 5},
    "evolve": {"steps_delta": 10},
}

# Korean Unicode range check
_KOREAN_RE = re.compile(r"[\uac00-\ud7a3\u3131-\u3163\u1100-\u11ff]")


def _is_korean(text: str) -> bool:
    return bool(_KOREAN_RE.search(text))


def _translate_to_english(text: str) -> str:
    """Translate Korean text to English using deep-translator."""
    if not _is_korean(text):
        return text

    try:
        from deep_translator import GoogleTranslator

        translated = GoogleTranslator(source="ko", target="en").translate(text)
        logger.info("Translated prompt: '%s' → '%s'", text, translated)
        return translated
    except Exception as e:
        logger.warning("Translation failed: %s. Using original text.", e)
        return text


def analyze_prompt(prompt: str) -> PromptAnalysis:
    """Analyze prompt for keywords and translate if needed."""
    is_korean = _is_korean(prompt)
    translated = _translate_to_english(prompt) if is_korean else prompt

    # Keyword analysis on English text
    lower = translated.lower()
    guidance_delta = 0.0
    steps_delta = 0
    motion_intensity = 0.0
    suggest_codeformer = False
    match_count = 0

    for keyword, adjustments in KEYWORD_ADJUSTMENTS.items():
        if keyword in lower:
            match_count += 1
            guidance_delta += adjustments.get("guidance_delta", 0.0)
            steps_delta += adjustments.get("steps_delta", 0)
            motion_intensity += adjustments.get("motion", 0.0)
            if adjustments.get("codeformer"):
                suggest_codeformer = True

    # Average out if multiple keywords matched
    if match_count > 1:
        guidance_delta /= match_count
        motion_intensity /= match_count

    # Clamp values
    guidance_delta = max(-2.0, min(2.0, guidance_delta))
    steps_delta = max(-20, min(30, steps_delta))
    motion_intensity = max(-1.0, min(1.0, motion_intensity))

    return PromptAnalysis(
        translated_prompt=translated,
        is_korean=is_korean,
        motion_intensity=round(motion_intensity, 2),
        suggest_codeformer=suggest_codeformer,
        guidance_delta=round(guidance_delta, 2),
        steps_delta=steps_delta,
    )

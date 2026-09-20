"""Emotion / cognitive-distortion extraction models."""
from typing import Literal

from pydantic import BaseModel, Field

Distortion = Literal[
    "none",
    "all_or_nothing_thinking",
    "catastrophizing",
    "mind_reading",
    "fortune_telling",
    "overgeneralization",
    "emotional_reasoning",
    "should_statements",
    "personalization",
    "labeling",
    "mental_filter",
    "discounting_the_positive",
]


class EmotionalState(BaseModel):
    primary_emotion: str = Field(
        default="neutral", description="One or two words, e.g. anxiety, guilt, shame, resentment"
    )
    cognitive_distortion: Distortion = "none"
    core_conflict: str = Field(
        default="", description="The underlying tension in a few words, e.g. 'autonomy vs. security'"
    )
    intensity: int = Field(default=1, ge=1, le=5, description="Emotional intensity, 1 (low) to 5")
    crisis_risk: bool = Field(
        default=False, description="True if the person expresses intent or wish to harm themselves"
    )

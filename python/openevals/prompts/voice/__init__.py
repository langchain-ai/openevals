import warnings

warnings.warn(
    "Voice prompts are in beta and may change in future releases.",
    UserWarning,
    stacklevel=2,
)

from .audio_quality import AUDIO_QUALITY_PROMPT
from .transcription_accuracy import TRANSCRIPTION_ACCURACY_PROMPT
from .dialogue_flow import DIALOGUE_FLOW_PROMPT
from .vocal_affect import VOCAL_AFFECT_PROMPT

__all__ = [
    "AUDIO_QUALITY_PROMPT",
    "TRANSCRIPTION_ACCURACY_PROMPT",
    "DIALOGUE_FLOW_PROMPT",
    "VOCAL_AFFECT_PROMPT",
]

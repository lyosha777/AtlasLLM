from backend.core.config import CHAT_MODEL
from backend.models.capabilities import detect_reasoning


def default_inference(*args, **kwargs):
    return {"provider": "ollama", "model": CHAT_MODEL, "model_digest": "test-digest",
            "requested_reasoning": None, "reasoning_mode": "provider_default_unverified",
            "capability": detect_reasoning({"capabilities": ["completion"]}).serialize(),
            "settings": {}, "actual_settings": None}

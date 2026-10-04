from collections.abc import Iterator
from datetime import datetime, timezone
import os

import httpx
import ollama

from backend.core.config import CHAT_MODEL
from backend.models.capabilities import ReasoningCapability, detect_reasoning


class ProviderUnavailable(RuntimeError):
    pass


class InferenceError(RuntimeError):
    pass


class AnswerFilter:
    """Discard legacy inline reasoning, including tags split across chunks."""

    def __init__(self):
        self.pending = ""
        self.closing = None

    def feed(self, text, final=False):
        result = []
        for character in text:
            self.pending += character
            tags = (self.closing,) if self.closing else ("<think>", "<analysis>")
            while self.pending:
                if self.pending in tags:
                    self.closing = None if self.closing else self.pending.replace("<", "</", 1)
                    self.pending = ""
                    break
                if any(tag.startswith(self.pending) for tag in tags):
                    break
                if not self.closing:
                    result.append(self.pending[0])
                self.pending = self.pending[1:]
        if final:
            # Incomplete opening markers are suppressed conservatively.
            self.pending = ""
        return "".join(result)


class OllamaProvider:
    name = "ollama"

    @staticmethod
    def canonical_name(name):
        return name if ":" in name.rsplit("/", 1)[-1] else name + ":latest"

    @staticmethod
    def host():
        host = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
        return host if "://" in host else "http://" + host

    def request(self, path, payload=None):
        try:
            with httpx.Client(base_url=self.host(), timeout=5, trust_env=False) as client:
                response = client.get(path) if payload is None else client.post(path, json=payload)
                if response.status_code == 404:
                    raise LookupError("This model is no longer installed in Ollama.")
                response.raise_for_status()
                return response.json()
        except (httpx.HTTPError, ValueError):
            raise ProviderUnavailable("Ollama is unavailable. Start Ollama and refresh the model list.") from None

    def installed(self):
        result = self.request("/api/tags")
        return result.get("models", [])

    def describe(self, tag):
        name = tag.get("name") or tag["model"]
        metadata = self.request("/api/show", {"model": name})
        details = {**tag.get("details", {}), **metadata.get("details", {})}
        info = metadata.get("model_info", {})
        family = details.get("family") or info.get("general.architecture")
        context = info.get(f"{info.get('general.architecture', family)}.context_length") or details.get("context_length")
        capabilities = metadata.get("capabilities", tag.get("capabilities", []))
        return {
            "name": name, "provider": self.name, "family": family,
            "families": details.get("families", []), "context_length": context,
            "parameter_size": details.get("parameter_size"), "quantization": details.get("quantization_level"),
            "format": details.get("format"), "size": tag.get("size"), "digest": tag.get("digest"),
            "modified_at": tag.get("modified_at"), "license": info.get("general.license"),
            "capabilities": capabilities, "available": True, "can_chat": "completion" in capabilities,
            "reasoning": detect_reasoning(metadata).serialize(),
        }

    def discover(self):
        catalog = {"provider": self.name, "online": True, "default_model": self.canonical_name(CHAT_MODEL),
                   "models": [], "error": None}
        try:
            tags = self.installed()
            for tag in tags:
                try:
                    catalog["models"].append(self.describe(tag))
                except (ProviderUnavailable, LookupError):
                    catalog["models"].append({"name": tag.get("name") or tag.get("model"),
                                              "available": False, "can_chat": False,
                                              "error": "Model metadata is unavailable. Refresh to retry."})
        except ProviderUnavailable as error:
            catalog.update(online=False, models=[], error=str(error))
        if catalog["online"]:
            try:
                catalog["server_version"] = self.request("/api/version").get("version")
            except (ProviderUnavailable, LookupError):
                catalog["server_version"] = None
        return catalog

    def inspect(self, name):
        canonical = self.canonical_name(name)
        tag = next((tag for tag in self.installed() if (tag.get("name") or tag.get("model")) == canonical), None)
        if tag is None:
            raise LookupError("Selected model is not installed. Choose an installed model or pull it manually with Ollama.")
        return self.describe(tag)

    def resolve(self, model=None, reasoning=None):
        descriptor = self.inspect(model or CHAT_MODEL)
        if not descriptor["can_chat"]:
            raise ValueError("This model does not advertise text completion support.")
        capability = descriptor["reasoning"]
        detected = ReasoningCapability(**capability)
        mode, settings = detected.resolve(reasoning)
        return {"provider": self.name, "model": descriptor["name"], "model_digest": descriptor["digest"],
                "requested_reasoning": reasoning, "reasoning_mode": mode, "capability": capability,
                "settings": settings, "actual_settings": None,
                "detected_at": datetime.now(timezone.utc).isoformat()}

    @staticmethod
    def inference_settings(inference, research=False):
        return {"model": inference["model"], **inference["settings"],
                "options": {"temperature": 0 if research else 0.3, "num_predict": 600}}

    def stream(self, messages, inference, research=False) -> Iterator[dict]:
        current = self.inspect(inference["model"])
        if current["digest"] != inference["model_digest"] or current["reasoning"] != inference["capability"]:
            raise InferenceError("Model capabilities changed after submission. Submit a new request.")
        settings = self.inference_settings(inference, research)
        yield {"type": "inference", "settings": settings}
        answer_filter = AnswerFilter()
        visible = False
        thinking = False
        stream = None
        try:
            stream = ollama.chat(messages=messages, stream=True, **settings)
            for chunk in stream:
                message = chunk.get("message", {})
                if message.get("thinking"):
                    if not thinking and not visible:
                        thinking = True
                        yield {"type": "status", "stage": "reasoning", "label": "Thinking"}
                content = answer_filter.feed(message.get("content") or "")
                if content:
                    if not visible:
                        visible = True
                        yield {"type": "status", "stage": "generating", "label": "Writing"}
                    yield {"type": "token", "content": content}
                else:
                    yield {"type": "checkpoint"}
            answer_filter.feed("", final=True)
            if not visible:
                raise InferenceError("The model produced no final answer. Its response budget may have been exhausted; try another model.")
        except InferenceError:
            raise
        except Exception as error:
            # SDK validation errors may contain response bodies with private thinking.
            raise InferenceError(f"Ollama inference failed ({type(error).__name__}). Check the selected model and Ollama.") from None
        finally:
            if stream is not None and hasattr(stream, "close"):
                try:
                    stream.close()
                except Exception:
                    raise InferenceError("Could not close the Ollama inference stream.") from None


provider = OllamaProvider()

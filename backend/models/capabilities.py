from dataclasses import asdict, dataclass
import re


@dataclass(frozen=True)
class ReasoningCapability:
    kind: str
    thinking_supported: bool | None
    settings: tuple[str, ...]
    default: str | None
    values: tuple[bool | str, ...]
    verified: bool
    explanation: str
    provider_default: bool | str | None = None

    def serialize(self):
        result = asdict(self)
        result.update(settings=list(self.settings), values=list(self.values))
        return result

    def resolve(self, requested=None):
        if requested is not None and requested not in self.settings:
            raise ValueError(f"Reasoning setting '{requested}' is not supported by this model.")
        mode = requested if requested is not None else self.default
        if self.kind == "none" and not self.verified:
            return "unsupported", {}
        if not self.verified or mode is None:
            return "provider_default_unverified", {}
        value = self.values[0] if not self.settings else self.values[self.settings.index(mode)]
        return mode, {"think": value}


def detect_reasoning(metadata):
    thinking = metadata.get("thinking")
    values = thinking.get("values") if isinstance(thinking, dict) else None
    valid = isinstance(values, list) and 0 < len(values) <= 16 and all(
        type(value) is bool or (isinstance(value, str) and re.fullmatch(r"[a-z][a-z0-9_-]{0,31}", value))
        for value in values
    )
    if valid and (all(type(value) is bool for value in values) or all(type(value) is str for value in values)):
        values = tuple(dict.fromkeys(values))
        settings = tuple("on" if value is True else "off" if value is False else value for value in values)
        default_value = thinking.get("default")
        default = next((setting for value, setting in zip(values, settings)
                        if type(value) is type(default_value) and value == default_value), None)
        if default is None:
            default_value = None
            default = "medium" if "medium" in settings else "on" if "on" in settings else settings[0]
        if "medium" in settings:
            default = "medium"
        if values == (False,):
            return ReasoningCapability("none", False, (), "off", values, True, "This model does not support thinking.", False)
        if len(values) == 1:
            return ReasoningCapability("fixed", True, (), default, values, True, "Reasoning is fixed by this model; effort cannot be adjusted.", default_value)
        kind = "binary" if all(type(value) is bool for value in values) else "effort"
        return ReasoningCapability(kind, True, settings, default, values, True, "Controls verified by Ollama /api/show thinking metadata.", default_value)
    capabilities = metadata.get("capabilities")
    if thinking is None and isinstance(capabilities, list) and "thinking" not in capabilities:
        return ReasoningCapability("none", False, (), None, (), False, "Ollama does not advertise thinking support for this model.")
    advertised = True if isinstance(capabilities, list) and "thinking" in capabilities else None
    return ReasoningCapability("unverified", advertised, (), None, (), False,
                               "Ollama has not supplied verified reasoning controls. Model defaults apply.")

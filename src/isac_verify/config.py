"""Typed access to the pilot configuration.

Every physical, waveform, statistical, and device setting is read from
``configs/pilot.yaml``. Modules do not embed scene constants.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "pilot.yaml"


class ConfigError(ValueError):
    """Raised when a required configuration key is missing or malformed."""


@dataclass(frozen=True)
class PilotConfig:
    path: Path
    data: dict[str, Any]

    @classmethod
    def load(cls, path: str | Path | None = None) -> "PilotConfig":
        config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
        with config_path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
        if not isinstance(data, dict):
            raise ConfigError(f"{config_path} does not contain a YAML mapping")
        return cls(path=config_path, data=data)

    def get(self, *keys: str) -> Any:
        value: Any = self.data
        for key in keys:
            if not isinstance(value, dict) or key not in value:
                dotted = ".".join(keys)
                raise ConfigError(f"missing configuration key {dotted!r}")
            value = value[key]
        return value

    def section(self, *keys: str) -> dict[str, Any]:
        value = self.get(*keys)
        if not isinstance(value, dict):
            raise ConfigError(f"configuration key {'.'.join(keys)!r} is not a mapping")
        return value

    def profile(self, name: str) -> dict[str, Any]:
        profiles = self.get("waveform", "profiles")
        for profile in profiles:
            if profile.get("name") == name:
                return profile
        raise ConfigError(f"unknown waveform profile {name!r}")

    @property
    def speed_of_light(self) -> float:
        return float(self.get("constants", "speed_of_light_m_per_s"))

    @property
    def frequency_hz(self) -> float:
        return float(self.get("rf", "carrier_frequency_hz"))

    @property
    def wavelength_m(self) -> float:
        return self.speed_of_light / self.frequency_hz

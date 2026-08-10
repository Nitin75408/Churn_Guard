"""Load and expose project configuration.

Every script in this project starts by calling :func:`load_config`. Nothing reads
``configs/config.yaml`` directly, and nothing hardcodes a path or a magic number.

Why this module exists at all
-----------------------------
Two problems it solves:

1. **Relative paths break.** ``pd.read_csv("data/raw/x.csv")`` works when you run
   from the project root and fails from ``notebooks/``. Here, every path in the
   YAML is resolved to an *absolute* path anchored at the project root, so code
   works no matter which directory you launch it from.

2. **Typos should fail loudly.** ``cfg["spilt"]`` on a plain dict raises a bare
   ``KeyError`` deep inside your training run. Here it raises a
   :class:`~churn_guard.exception.ConfigError` naming the key and listing the
   valid options.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from churn_guard.exception import ConfigError

# ---------------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------------
# This file lives at  <root>/src/churn_guard/config.py
#                                ^parents[0]  ^parents[1]  ^parents[2]
# so parents[2] is the project root. Computed from __file__ rather than from
# the current working directory, which is what makes paths run-location proof.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

DEFAULT_CONFIG_PATH: Path = PROJECT_ROOT / "configs" / "config.yaml"

# Keys under `paths:` are resolved from strings into absolute Path objects.
_PATHS_SECTION = "paths"


class ConfigSection:
    """A dict wrapper giving attribute access with helpful errors.

    ``cfg.split.test_size`` reads better than ``cfg["split"]["test_size"]`` and,
    more usefully, tells you what you got wrong when you mistype it.
    """

    def __init__(self, data: dict[str, Any], _path: str = "config") -> None:
        self._data = data
        self._path = _path

    def __getattr__(self, name: str) -> Any:
        # __getattr__ only fires for names not found normally, so `_data` and
        # `_path` never reach here.
        try:
            value = self._data[name]
        except KeyError as err:
            valid = ", ".join(sorted(self._data)) or "(section is empty)"
            raise ConfigError(
                f"No key '{name}' in {self._path}. Available keys: {valid}"
            ) from err

        if isinstance(value, dict):
            return ConfigSection(value, _path=f"{self._path}.{name}")
        return value

    def __getitem__(self, name: str) -> Any:
        return getattr(self, name)

    def __contains__(self, name: str) -> bool:
        return name in self._data

    def to_dict(self) -> dict[str, Any]:
        """Return the raw nested dict.

        Useful for logging a whole run's settings, or handing the config to a
        library that expects plain dicts (MLflow's ``log_params``, for example).
        """
        return self._data

    def __repr__(self) -> str:
        return f"ConfigSection({self._path}: {sorted(self._data)})"


def _resolve_paths(raw: dict[str, Any]) -> dict[str, Any]:
    """Turn every entry under ``paths:`` into an absolute Path, creating it."""
    paths = raw.get(_PATHS_SECTION)
    if not isinstance(paths, dict):
        return raw

    resolved: dict[str, Path] = {}
    for key, value in paths.items():
        directory = PROJECT_ROOT / value
        # Create directories eagerly so no downstream code has to remember to.
        # A missing output folder is a boring way to lose a 20-minute training run.
        directory.mkdir(parents=True, exist_ok=True)
        resolved[key] = directory

    raw[_PATHS_SECTION] = resolved
    return raw


def load_config(config_path: str | Path | None = None) -> ConfigSection:
    """Read the YAML config and return it with absolute, existing paths.

    Args:
        config_path: Override for the config file location. Defaults to
            ``<project root>/configs/config.yaml``.

    Returns:
        The configuration, with attribute access (``cfg.data.target_column``).

    Raises:
        ConfigError: If the file is missing, is not valid YAML, or is empty.
    """
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH

    if not path.is_file():
        raise ConfigError(
            f"Config file not found at {path}. Expected it at "
            f"{DEFAULT_CONFIG_PATH.relative_to(PROJECT_ROOT)} relative to the project root."
        )

    try:
        with path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
    except yaml.YAMLError as err:
        # Almost always an indentation mistake; say so rather than dumping a
        # parser trace on someone who has never read one.
        raise ConfigError(
            f"{path} is not valid YAML — check indentation (spaces, never tabs)."
        ) from err

    if not raw:
        raise ConfigError(f"{path} is empty.")

    return ConfigSection(_resolve_paths(raw))

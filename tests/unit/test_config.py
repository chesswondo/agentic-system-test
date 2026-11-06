import pytest
import yaml
from pydantic import ValidationError

from picos.config import DEFAULT_CONFIG_PATH, AppConfig, load_config
from tests.helpers import ROOT


def _raw() -> dict:
    return yaml.safe_load((ROOT / DEFAULT_CONFIG_PATH).read_text(encoding="utf-8"))


def test_default_config_loads() -> None:
    cfg = load_config(ROOT / DEFAULT_CONFIG_PATH)
    assert cfg.llm.provider == "fake"
    assert cfg.runtime.max_ticks > 0


def test_unknown_key_rejected() -> None:
    raw = _raw()
    raw["runtime"]["max_tick"] = 5
    with pytest.raises(ValidationError):
        AppConfig.model_validate(raw)


def test_out_of_range_rejected() -> None:
    raw = _raw()
    raw["faults"]["default"]["p_error"] = 1.5
    with pytest.raises(ValidationError):
        AppConfig.model_validate(raw)


def test_missing_file() -> None:
    with pytest.raises(FileNotFoundError):
        load_config(ROOT / "config" / "nope.yaml")

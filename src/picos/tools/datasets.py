"""Loaders for the static world data in `data/` (validated by pydantic)."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

from pydantic import TypeAdapter

from picos.domain import DailyForecast, GearCatalog, Trail

DEFAULT_DATA_DIR = Path("data")

_TRAILS = TypeAdapter(list[Trail])


def _read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def load_trails(data_dir: Path = DEFAULT_DATA_DIR) -> list[Trail]:
    trails = _TRAILS.validate_python(_read_json(data_dir / "trails.json"))
    ids = [t.id for t in trails]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate trail ids in trails.json")
    return trails


def load_gear_catalog(data_dir: Path = DEFAULT_DATA_DIR) -> GearCatalog:
    return GearCatalog.model_validate(_read_json(data_dir / "gear_catalog.json"))


def parse_open_meteo_daily(daily: dict[str, list[Any]]) -> list[DailyForecast]:
    """Turn Open-Meteo's column-oriented `daily` block into one record per day."""
    times = daily["time"]
    columns = {k: v for k, v in daily.items() if k != "time"}
    bad = sorted(k for k, v in columns.items() if len(v) != len(times))
    if bad:
        raise ValueError(f"daily columns with wrong length: {bad}")
    return [
        DailyForecast(date=date.fromisoformat(t), **{k: v[i] for k, v in columns.items()})
        for i, t in enumerate(times)
    ]


def load_weather_snapshot(data_dir: Path = DEFAULT_DATA_DIR) -> dict[str, list[DailyForecast]]:
    """Return forecasts keyed by area id."""
    raw = _read_json(data_dir / "weather_snapshot.json")
    return {area: parse_open_meteo_daily(body["daily"]) for area, body in raw["areas"].items()}

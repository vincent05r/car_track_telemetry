from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from telemetry_utils import (
    build_lap_summary,
    build_sector_summary,
    discover_telemetry_files,
    fastest_lap_id,
    format_lap_time,
    load_telemetry,
    nearest_sample_index,
    prepare_lap,
    timed_laps,
)


def sample_telemetry() -> pd.DataFrame:
    rows = []
    for lap, elapsed_values, latitude_offset in (
        (0, [0, 0], 0.0),
        (1, [100, 200, 300], 0.001),
        (2, [100, 180, 250], 0.002),
    ):
        for index, elapsed in enumerate(elapsed_values):
            rows.append(
                {
                    "Lap": lap,
                    "Elapsed Time (ms)": elapsed,
                    "Speed (MPH)": 40 + index + lap,
                    "Latitude (decimal)": -33.8 + latitude_offset + index * 0.0001,
                    "Longitude (decimal)": 150.8 + index * 0.0001,
                    "Lateral Acceleration (m/s^2)": 1.0,
                    "Longitudinal Acceleration (m/s^2)": 0.5,
                    "Throttle Position (%)": 30 + index,
                    "Brake Pressure (bar)": index,
                    "Steering Angle (deg)": index * 2,
                    "Power Level (KW)": 20 + index,
                    "State of Charge (%)": 80,
                }
            )
    frame = pd.DataFrame(rows)
    frame["Elapsed (s)"] = frame["Elapsed Time (ms)"] / 1_000
    frame["Speed (km/h)"] = frame["Speed (MPH)"] * 1.609344
    frame["Lateral Acceleration (g)"] = frame["Lateral Acceleration (m/s^2)"] / 9.80665
    frame["Longitudinal Acceleration (g)"] = frame["Longitudinal Acceleration (m/s^2)"] / 9.80665
    return frame


def test_discover_and_load_telemetry(tmp_path: Path) -> None:
    frame = sample_telemetry().drop(
        columns=[
            "Elapsed (s)",
            "Speed (km/h)",
            "Lateral Acceleration (g)",
            "Longitudinal Acceleration (g)",
        ]
    )
    later = tmp_path / "telemetry-02.csv"
    earlier = tmp_path / "telemetry-01.csv"
    frame.to_csv(later, index=False)
    frame.to_csv(earlier, index=False)

    assert discover_telemetry_files(tmp_path) == [earlier, later]
    loaded = load_telemetry(earlier)
    assert loaded["Source"].iat[0] == earlier.name
    assert loaded["Speed (km/h)"].iat[0] == pytest.approx(loaded["Speed (MPH)"].iat[0] * 1.609344)


def test_summary_selects_fastest_lap_and_formats_time() -> None:
    timed = timed_laps(sample_telemetry())
    summary = build_lap_summary(timed)

    assert summary["Lap"].tolist() == [1, 2]
    assert fastest_lap_id(summary) == 2
    assert summary.loc[summary["Lap"].eq(2), "Gap to Fastest (s)"].iat[0] == 0
    assert format_lap_time(62_345) == "1:02.345"


def test_sector_times_sum_to_each_recorded_lap_time() -> None:
    timed = timed_laps(sample_telemetry())
    sector_summary, best = build_sector_summary(timed, n_sectors=2)

    expected = {1: 300.0, 2: 250.0}
    for _, row in sector_summary.iterrows():
        assert row["Sector Sum"] == pytest.approx(expected[int(row["Lap"])])
    assert best["Sector"].tolist() == [1, 2]


def test_nearest_sample_prefers_current_neighborhood_for_duplicate_gps() -> None:
    timed = timed_laps(sample_telemetry())
    lap = prepare_lap(timed, 1)
    duplicate = pd.concat([lap, lap.iloc[[1]]], ignore_index=True)
    duplicate.loc[3, "Elapsed (s)"] = 0.4

    index = nearest_sample_index(
        duplicate,
        longitude=float(lap.loc[1, "Longitude (decimal)"]),
        latitude=float(lap.loc[1, "Latitude (decimal)"]),
        current_index=3,
    )
    assert index == 3


def test_load_rejects_missing_required_columns(tmp_path: Path) -> None:
    path = tmp_path / "bad.csv"
    pd.DataFrame({"Lap": [1]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing required columns"):
        load_telemetry(path)

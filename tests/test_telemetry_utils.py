import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from telemetry_utils import (
    TelemetryDashboard,
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


def test_dashboard_uses_bundled_vegalite_renderer(tmp_path: Path) -> None:
    frame = sample_telemetry().drop(
        columns=[
            "Elapsed (s)",
            "Speed (km/h)",
            "Lateral Acceleration (g)",
            "Longitudinal Acceleration (g)",
        ]
    )
    frame.to_csv(tmp_path / "telemetry.csv", index=False)

    dashboard = TelemetryDashboard(tmp_path)
    try:
        spec = dashboard.plot_spec
        assert spec is not None
        assert spec["$schema"].endswith("/vega-lite/v5.json")
        assert len(spec["data"]["values"]) == len(dashboard.lap_data)

        track = spec["vconcat"][-1]
        selection = track["layer"][2]["params"][0]
        assert selection["name"] == "telemetry_cursor"
        assert selection["select"]["nearest"] is True
        assert selection["select"]["on"] == "pointermove"
        assert selection["value"] == [{"sample": 0}]

        serialized = json.dumps(spec, allow_nan=False)
        assert "jupyter-matplotlib" not in serialized
        assert "car-track-telemetry" not in serialized

        dashboard.speed_unit.value = "MPH"
        assert dashboard.plot_spec is not None
        speed_axis = dashboard.plot_spec["vconcat"][0]["layer"][0]["encoding"]["y"]
        assert speed_axis["title"] == "Speed (MPH)"
    finally:
        dashboard._clear_figure()


def test_dashboard_optional_charts_and_cross_session_comparison(tmp_path: Path) -> None:
    frame = sample_telemetry().drop(
        columns=[
            "Elapsed (s)",
            "Speed (km/h)",
            "Lateral Acceleration (g)",
            "Longitudinal Acceleration (g)",
        ]
    )
    comparison_frame = frame.copy()
    comparison_frame["Speed (MPH)"] += 8
    comparison_path = tmp_path / "comparison.csv"
    primary_path = tmp_path / "primary.csv"
    comparison_frame.to_csv(comparison_path, index=False)
    frame.to_csv(primary_path, index=False)

    dashboard = TelemetryDashboard(tmp_path, default_file=primary_path.name)
    try:
        assert dashboard.selected_optional_charts == ()
        assert set(dashboard.optional_plot_checkboxes) == {
            "driver_inputs",
            "vehicle_dynamics",
            "power_steering",
        }
        assert dashboard.plot_spec is not None
        assert len(dashboard.plot_spec["vconcat"]) == 2
        assert dashboard.plot_spec["vconcat"][-2]["title"].endswith("Speed")
        assert dashboard.plot_spec["vconcat"][-1]["title"].startswith("Track map")

        for checkbox in dashboard.optional_plot_checkboxes.values():
            checkbox.value = True
        assert dashboard.plot_spec is not None
        chart_titles = [chart["title"] for chart in dashboard.plot_spec["vconcat"]]
        assert chart_titles[:3] == [
            "Driver inputs",
            "Vehicle dynamics",
            "Power and steering",
        ]
        assert chart_titles[-2].endswith("Speed")
        assert chart_titles[-1].startswith("Track map")

        dashboard.comparison_enabled.value = True
        dashboard.comparison_file_dropdown.value = str(comparison_path)
        dashboard.comparison_lap_dropdown.value = 1

        spec = dashboard.plot_spec
        assert spec is not None
        assert dashboard.selected_comparison_file == comparison_path
        assert dashboard.selected_comparison_lap == 1
        assert not dashboard.comparison_lap_data.empty
        assert dashboard.comparison_telemetry["Source"].unique().tolist() == [
            comparison_path.name
        ]
        assert len(spec["data"]["values"]) == 2 * len(dashboard.lap_data)
        assert {row["series_role"] for row in spec["data"]["values"]} == {
            "Primary",
            "Comparison",
        }

        speed_chart = spec["vconcat"][-2]
        assert speed_chart["title"] == "Speed comparison"
        colour = speed_chart["layer"][0]["encoding"]["color"]
        assert colour["field"] == "series"
        assert colour["legend"]["title"] == "Lap"
        assert len(colour["scale"]["domain"]) == 2

        track = spec["vconcat"][-1]
        assert track["layer"][0]["transform"] == [
            {"filter": "datum.series_role === 'Primary'"}
        ]
        assert track["layer"][2]["params"][0]["select"]["fields"] == ["sample"]
    finally:
        dashboard._clear_figure()

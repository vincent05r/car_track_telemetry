"""Utilities and interactive widgets for Tesla Track Mode telemetry.

The data-processing functions in this module are deliberately independent from
the notebook so they can be tested and reused from scripts.  ``TelemetryDashboard``
adds Jupyter controls and a synchronized Vega-Lite telemetry view.
"""

from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from IPython.display import display
import ipywidgets as widgets


EARTH_RADIUS_M = 6_371_000.0

REQUIRED_COLUMNS = (
    "Lap",
    "Elapsed Time (ms)",
    "Speed (MPH)",
    "Latitude (decimal)",
    "Longitude (decimal)",
    "Throttle Position (%)",
    "Brake Pressure (bar)",
)

GRAPH_CHANNELS = (
    "Lateral Acceleration (m/s^2)",
    "Longitudinal Acceleration (m/s^2)",
    "Steering Angle (deg)",
    "Power Level (KW)",
    "State of Charge (%)",
)

OPTIONAL_PLOT_GROUPS = (
    "driver_inputs",
    "vehicle_dynamics",
    "power_steering",
)


def discover_telemetry_files(data_dir: str | Path) -> list[Path]:
    """Return telemetry CSVs directly inside *data_dir*, sorted by filename."""

    directory = Path(data_dir).expanduser()
    if not directory.is_dir():
        raise FileNotFoundError(f"Telemetry directory not found: {directory.resolve()}")

    files = sorted(
        (path for path in directory.glob("*.csv") if path.is_file()),
        key=lambda path: path.name.casefold(),
    )
    if not files:
        raise FileNotFoundError(f"No CSV telemetry files found in {directory.resolve()}")
    return files


def load_telemetry(csv_path: str | Path) -> pd.DataFrame:
    """Load and validate one Tesla Track Mode telemetry CSV."""

    path = Path(csv_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Telemetry CSV not found: {path.resolve()}")

    telemetry = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    if telemetry.empty:
        raise ValueError(f"Telemetry CSV is empty: {path.name}")

    missing = sorted(set(REQUIRED_COLUMNS).difference(telemetry.columns))
    if missing:
        raise ValueError(f"{path.name} is missing required columns: {missing}")

    for column in telemetry.columns:
        telemetry[column] = pd.to_numeric(telemetry[column], errors="coerce")

    telemetry["Source"] = path.name
    telemetry["Elapsed (s)"] = telemetry["Elapsed Time (ms)"] / 1_000.0
    telemetry["Speed (km/h)"] = telemetry["Speed (MPH)"] * 1.609344

    if "Lateral Acceleration (m/s^2)" in telemetry:
        telemetry["Lateral Acceleration (g)"] = (
            telemetry["Lateral Acceleration (m/s^2)"] / 9.80665
        )
    if "Longitudinal Acceleration (m/s^2)" in telemetry:
        telemetry["Longitudinal Acceleration (g)"] = (
            telemetry["Longitudinal Acceleration (m/s^2)"] / 9.80665
        )
    return telemetry


def timed_laps(telemetry: pd.DataFrame) -> pd.DataFrame:
    """Return positive-lap, positive-time samples in lap/time order."""

    timed = telemetry.loc[
        telemetry["Lap"].gt(0) & telemetry["Elapsed Time (ms)"].gt(0)
    ].copy()
    if timed.empty:
        return timed

    timed["Lap"] = timed["Lap"].astype(int)
    return timed.sort_values(["Lap", "Elapsed Time (ms)"], kind="stable").reset_index(drop=True)


def format_lap_time(milliseconds: float | int) -> str:
    """Format milliseconds as ``m:ss.mmm``."""

    if not np.isfinite(milliseconds) or milliseconds < 0:
        return "—"
    total_ms = int(round(float(milliseconds)))
    minutes, remainder = divmod(total_ms, 60_000)
    seconds, millis = divmod(remainder, 1_000)
    return f"{minutes}:{seconds:02d}.{millis:03d}"


def build_lap_summary(timed: pd.DataFrame) -> pd.DataFrame:
    """Build one summary row per timed lap."""

    columns = [
        "Lap",
        "Lap Time (ms)",
        "Lap Time",
        "Gap to Fastest (s)",
        "Samples",
        "Average Speed (MPH)",
        "Maximum Speed (MPH)",
        "Average Speed (km/h)",
        "Maximum Speed (km/h)",
        "Average Throttle (%)",
        "Maximum Brake (bar)",
        "Median Sample Interval (ms)",
    ]
    if timed.empty:
        return pd.DataFrame(columns=columns)

    summary = (
        timed.groupby("Lap", as_index=False)
        .agg(
            **{
                "Lap Time (ms)": ("Elapsed Time (ms)", "max"),
                "Samples": ("Elapsed Time (ms)", "size"),
                "Average Speed (MPH)": ("Speed (MPH)", "mean"),
                "Maximum Speed (MPH)": ("Speed (MPH)", "max"),
                "Average Throttle (%)": ("Throttle Position (%)", "mean"),
                "Maximum Brake (bar)": ("Brake Pressure (bar)", "max"),
            }
        )
        .sort_values("Lap")
        .reset_index(drop=True)
    )
    summary["Lap"] = summary["Lap"].astype(int)
    summary["Average Speed (km/h)"] = summary["Average Speed (MPH)"] * 1.609344
    summary["Maximum Speed (km/h)"] = summary["Maximum Speed (MPH)"] * 1.609344
    summary["Lap Time"] = summary["Lap Time (ms)"].map(format_lap_time)
    fastest_ms = float(summary["Lap Time (ms)"].min())
    summary["Gap to Fastest (s)"] = (summary["Lap Time (ms)"] - fastest_ms) / 1_000.0

    median_intervals = timed.groupby("Lap")["Elapsed Time (ms)"].apply(
        lambda values: values.diff().loc[lambda diff: diff.gt(0)].median()
    )
    summary["Median Sample Interval (ms)"] = summary["Lap"].map(median_intervals)
    return summary[columns]


def fastest_lap_id(lap_summary: pd.DataFrame) -> int:
    """Return the lap ID with the shortest recorded duration."""

    if lap_summary.empty:
        raise ValueError("No timed laps are available")
    return int(lap_summary.loc[lap_summary["Lap Time (ms)"].idxmin(), "Lap"])


def _haversine_step_distances(latitude: np.ndarray, longitude: np.ndarray) -> np.ndarray:
    latitude_rad = np.radians(latitude)
    longitude_rad = np.radians(longitude)
    delta_latitude = np.diff(latitude_rad)
    delta_longitude = np.diff(longitude_rad)
    haversine = (
        np.sin(delta_latitude / 2.0) ** 2
        + np.cos(latitude_rad[:-1])
        * np.cos(latitude_rad[1:])
        * np.sin(delta_longitude / 2.0) ** 2
    )
    return 2.0 * EARTH_RADIUS_M * np.arcsin(np.sqrt(np.clip(haversine, 0.0, 1.0)))


def lap_sector_times(lap_data: pd.DataFrame, n_sectors: int = 3) -> tuple[np.ndarray, float]:
    """Return equal-distance sector durations and measured GPS lap distance."""

    if not isinstance(n_sectors, int) or n_sectors < 1:
        raise ValueError("n_sectors must be a positive integer")

    clean = (
        lap_data.dropna(
            subset=["Elapsed Time (ms)", "Latitude (decimal)", "Longitude (decimal)"]
        )
        .sort_values("Elapsed Time (ms)", kind="stable")
        .reset_index(drop=True)
    )
    if len(clean) < 2:
        raise ValueError("The lap does not have enough GPS samples")

    latitude = clean["Latitude (decimal)"].to_numpy(dtype=float)
    longitude = clean["Longitude (decimal)"].to_numpy(dtype=float)
    step_distance = _haversine_step_distances(latitude, longitude)
    cumulative_distance = np.concatenate(([0.0], np.cumsum(step_distance)))
    total_distance = float(cumulative_distance[-1])
    if not np.isfinite(total_distance) or total_distance <= 0:
        raise ValueError("The lap does not have a usable GPS trace")

    moving = np.r_[True, np.diff(cumulative_distance) > 0]
    elapsed_ms = clean["Elapsed Time (ms)"].to_numpy(dtype=float)
    boundaries = np.linspace(0.0, total_distance, n_sectors + 1)
    boundary_times = np.interp(
        boundaries,
        cumulative_distance[moving],
        elapsed_ms[moving],
    )
    boundary_times[0] = 0.0
    boundary_times[-1] = elapsed_ms[-1]
    return np.diff(boundary_times), total_distance


def build_sector_summary(
    timed: pd.DataFrame, n_sectors: int = 3
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return per-lap sector times and the best observed sector rows."""

    records: list[dict[str, float | int]] = []
    for lap, lap_data in timed.groupby("Lap", sort=True):
        sector_times_ms, lap_distance_m = lap_sector_times(lap_data, n_sectors)
        for sector, sector_time_ms in enumerate(sector_times_ms, start=1):
            records.append(
                {
                    "Lap": int(lap),
                    "Sector": sector,
                    "Sector Time (ms)": float(sector_time_ms),
                    "GPS Lap Distance (m)": lap_distance_m,
                }
            )

    if not records:
        return pd.DataFrame(), pd.DataFrame()

    long = pd.DataFrame.from_records(records)
    wide = long.pivot(index="Lap", columns="Sector", values="Sector Time (ms)")
    wide.columns = [f"S{sector}" for sector in wide.columns]
    wide["Sector Sum"] = wide.sum(axis=1)
    wide["GPS Distance (m)"] = long.groupby("Lap")["GPS Lap Distance (m)"].first()
    wide = wide.reset_index()

    best_indices = long.groupby("Sector")["Sector Time (ms)"].idxmin()
    best = long.loc[best_indices, ["Sector", "Lap", "Sector Time (ms)"]].reset_index(drop=True)
    best["Sector"] = best["Sector"].astype(int)
    best["Lap"] = best["Lap"].astype(int)
    return wide, best


def prepare_lap(timed: pd.DataFrame, lap: int) -> pd.DataFrame:
    """Return one lap with valid GPS points and cumulative GPS distance."""

    lap_data = (
        timed.loc[timed["Lap"].eq(int(lap))]
        .dropna(subset=["Elapsed (s)", "Latitude (decimal)", "Longitude (decimal)"])
        .sort_values("Elapsed Time (ms)", kind="stable")
        .reset_index(drop=True)
    )
    if len(lap_data) < 2:
        raise ValueError(f"Lap {lap} does not have enough valid GPS samples")

    steps = _haversine_step_distances(
        lap_data["Latitude (decimal)"].to_numpy(dtype=float),
        lap_data["Longitude (decimal)"].to_numpy(dtype=float),
    )
    lap_data["GPS Distance (m)"] = np.concatenate(([0.0], np.cumsum(steps)))
    return lap_data


def nearest_sample_index(
    lap_data: pd.DataFrame,
    longitude: float,
    latitude: float,
    current_index: int | None = None,
) -> int:
    """Find the closest GPS sample, preferring the current-time neighborhood on ties."""

    if not np.isfinite(longitude) or not np.isfinite(latitude):
        raise ValueError("A finite longitude and latitude are required")

    longitude_values = lap_data["Longitude (decimal)"].to_numpy(dtype=float)
    latitude_values = lap_data["Latitude (decimal)"].to_numpy(dtype=float)
    longitude_scale = np.cos(np.radians(np.nanmean(latitude_values)))
    distance_squared = (
        (longitude_values - longitude) * longitude_scale
    ) ** 2 + (latitude_values - latitude) ** 2
    minimum = float(np.nanmin(distance_squared))
    candidates = np.flatnonzero(np.isclose(distance_squared, minimum, rtol=1e-9, atol=1e-18))
    if not len(candidates):
        raise ValueError("The lap has no finite GPS samples")
    if current_index is None or len(candidates) == 1:
        return int(candidates[0])
    return int(candidates[np.argmin(np.abs(candidates - int(current_index)))])


def _channel(lap_data: pd.DataFrame, name: str) -> np.ndarray:
    if name in lap_data:
        return lap_data[name].to_numpy(dtype=float)
    return np.full(len(lap_data), np.nan, dtype=float)


def _json_values(values: np.ndarray, decimals: int = 6) -> list[float | None]:
    rounded = np.round(np.asarray(values, dtype=float), decimals)
    return [float(value) if np.isfinite(value) else None for value in rounded]


def build_telemetry_plot_spec(
    lap_data: pd.DataFrame,
    *,
    title: str,
    speed_unit: str,
    optional_charts: tuple[str, ...] = OPTIONAL_PLOT_GROUPS,
    comparison_lap_data: pd.DataFrame | None = None,
    comparison_title: str | None = None,
) -> dict[str, Any]:
    """Build a self-contained Vega-Lite v5 specification for one or two laps.

    VS Code bundles the Vega-Lite renderer with its Jupyter renderer extension,
    so this view does not require ipympl, a widget CDN, or custom JavaScript.
    The comparison lap is rotated to the GPS point nearest the primary start,
    then interpolated onto the primary lap's distance-progress points. This
    aligns corners even when two sessions recorded different lap boundaries.
    """

    if lap_data.empty:
        raise ValueError("Cannot plot an empty lap")
    if speed_unit not in {"km/h", "MPH"}:
        raise ValueError("speed_unit must be 'km/h' or 'MPH'")
    if comparison_lap_data is not None and comparison_lap_data.empty:
        raise ValueError("Cannot compare an empty lap")

    selected_optional = tuple(dict.fromkeys(optional_charts))
    unknown_optional = sorted(set(selected_optional).difference(OPTIONAL_PLOT_GROUPS))
    if unknown_optional:
        raise ValueError(f"Unknown optional chart groups: {unknown_optional}")
    for frame, frame_name in (
        (lap_data, "primary lap"),
        (comparison_lap_data, "comparison lap"),
    ):
        if frame is not None and "GPS Distance (m)" not in frame:
            raise ValueError(f"The {frame_name} has no GPS Distance (m) column")

    speed_column = "Speed (km/h)" if speed_unit == "km/h" else "Speed (MPH)"
    latitude = lap_data["Latitude (decimal)"].to_numpy(dtype=float)
    longitude = lap_data["Longitude (decimal)"].to_numpy(dtype=float)
    mean_latitude = float(np.nanmean(latitude))
    mean_longitude = float(np.nanmean(longitude))
    latitude_scale = np.pi * EARTH_RADIUS_M / 180.0
    longitude_scale = latitude_scale * np.cos(np.radians(mean_latitude))
    east_m = (longitude - mean_longitude) * longitude_scale
    north_m = (latitude - mean_latitude) * latitude_scale

    def lap_progress(frame: pd.DataFrame) -> np.ndarray:
        distance = frame["GPS Distance (m)"].to_numpy(dtype=float)
        finite = np.isfinite(distance)
        if finite.sum() >= 2:
            row_index = np.arange(len(distance), dtype=float)
            distance = np.interp(row_index, row_index[finite], distance[finite])
        elif finite.sum() == 1:
            distance = np.full(len(distance), distance[finite][0], dtype=float)
        else:
            distance = np.zeros(len(distance), dtype=float)
        distance = np.maximum.accumulate(distance - distance[0])
        total = float(distance[-1])
        if total <= 0.0:
            return np.linspace(0.0, 1.0, len(frame))
        return np.clip(distance / total, 0.0, 1.0)

    def raw_channels(frame: pd.DataFrame) -> dict[str, np.ndarray]:
        return {
            "time_s": frame["Elapsed (s)"].to_numpy(dtype=float),
            "distance_m": frame["GPS Distance (m)"].to_numpy(dtype=float),
            "speed": _channel(frame, speed_column),
            "throttle": _channel(frame, "Throttle Position (%)"),
            "brake": np.clip(_channel(frame, "Brake Pressure (bar)"), 0.0, None),
            "lateral_g": _channel(frame, "Lateral Acceleration (g)"),
            "longitudinal_g": _channel(frame, "Longitudinal Acceleration (g)"),
            "power": _channel(frame, "Power Level (KW)"),
            "steering": _channel(frame, "Steering Angle (deg)"),
            "longitude": _channel(frame, "Longitude (decimal)"),
            "latitude": _channel(frame, "Latitude (decimal)"),
        }

    def interpolate_channel(
        source_progress: np.ndarray,
        values: np.ndarray,
        target_progress: np.ndarray,
    ) -> np.ndarray:
        finite = np.isfinite(source_progress) & np.isfinite(values)
        if not finite.any():
            return np.full(len(target_progress), np.nan, dtype=float)
        source = source_progress[finite]
        channel_values = values[finite]
        order = np.argsort(source, kind="stable")
        source = source[order]
        channel_values = channel_values[order]
        source, unique_indices = np.unique(source, return_index=True)
        channel_values = channel_values[unique_indices]
        if len(source) == 1:
            return np.full(len(target_progress), channel_values[0], dtype=float)
        return np.interp(target_progress, source, channel_values)

    primary_progress = lap_progress(lap_data)
    primary_channels = raw_channels(lap_data)
    primary_channels.update(
        {
            "progress_pct": primary_progress * 100.0,
            "east_m": east_m,
            "north_m": north_m,
        }
    )

    primary_label = f"Primary | {title}"
    series = [("Primary", primary_label, primary_channels)]
    if comparison_lap_data is not None:
        comparison_progress = lap_progress(comparison_lap_data)
        anchor_index = nearest_sample_index(
            comparison_lap_data,
            longitude=float(longitude[0]),
            latitude=float(latitude[0]),
        )
        anchor_progress = float(comparison_progress[anchor_index])
        aligned_progress = np.concatenate(
            (
                comparison_progress[anchor_index:] - anchor_progress,
                comparison_progress[: anchor_index + 1] + 1.0 - anchor_progress,
            )
        )
        comparison_raw = raw_channels(comparison_lap_data)
        comparison_channels: dict[str, np.ndarray] = {}
        for name, values in comparison_raw.items():
            if name == "time_s":
                duration = float(np.nanmax(values))
                anchor_time = float(values[anchor_index])
                aligned_values = np.concatenate(
                    (
                        values[anchor_index:] - anchor_time,
                        values[: anchor_index + 1] + duration - anchor_time,
                    )
                )
            else:
                aligned_values = np.concatenate(
                    (values[anchor_index:], values[: anchor_index + 1])
                )
            comparison_channels[name] = interpolate_channel(
                aligned_progress, aligned_values, primary_progress
            )
        comparison_total_distance = float(
            comparison_lap_data["GPS Distance (m)"].iloc[-1]
        )
        comparison_channels["distance_m"] = (
            primary_progress * comparison_total_distance
        )
        comparison_channels["progress_pct"] = primary_progress * 100.0
        comparison_channels["east_m"] = (
            comparison_channels["longitude"] - mean_longitude
        ) * longitude_scale
        comparison_channels["north_m"] = (
            comparison_channels["latitude"] - mean_latitude
        ) * latitude_scale
        comparison_label = f"Comparison | {comparison_title or 'comparison lap'}"
        series.append(("Comparison", comparison_label, comparison_channels))

    decimals = {
        "time_s": 3,
        "distance_m": 1,
        "progress_pct": 3,
        "speed": 2,
        "throttle": 2,
        "brake": 2,
        "lateral_g": 3,
        "longitudinal_g": 3,
        "power": 2,
        "steering": 2,
        "east_m": 2,
        "north_m": 2,
        "longitude": 7,
        "latitude": 7,
    }
    records: list[dict[str, Any]] = []
    series_labels: list[str] = []
    for role, label, channel_values in series:
        series_labels.append(label)
        json_channels = {
            name: _json_values(values, decimals[name])
            for name, values in channel_values.items()
        }
        records.extend(
            {
                "sample": index,
                "series": label,
                "series_role": role,
                **{name: values[index] for name, values in json_channels.items()},
            }
            for index in range(len(lap_data))
        )

    chart_width = 960
    x_span = max(float(np.nanmax(east_m) - np.nanmin(east_m)), 1.0)
    y_span = max(float(np.nanmax(north_m) - np.nanmin(north_m)), 1.0)
    track_height = int(np.clip(chart_width * y_span / x_span, 330, 600))
    x_padding = max(x_span * 0.06, 2.0)
    y_padding = max(y_span * 0.06, 2.0)
    east_domain = [
        float(np.nanmin(east_m) - x_padding),
        float(np.nanmax(east_m) + x_padding),
    ]
    north_domain = [
        float(np.nanmin(north_m) - y_padding),
        float(np.nanmax(north_m) + y_padding),
    ]

    has_comparison = comparison_lap_data is not None
    graph_x_encoding = {
        "field": "progress_pct" if has_comparison else "time_s",
        "type": "quantitative",
        "title": "Lap distance (%)" if has_comparison else "Elapsed lap time (s)",
        "scale": (
            {"domain": [0, 100], "nice": False, "zero": True}
            if has_comparison
            else {"zero": False}
        ),
    }
    series_colours = ["#1d4ed8", "#c026d3"]

    def series_colour_encoding(*, legend: bool) -> dict[str, Any]:
        if not has_comparison:
            return {"value": series_colours[0]}
        return {
            "field": "series",
            "type": "nominal",
            "title": "Lap",
            "scale": {"domain": series_labels, "range": series_colours},
            "legend": {"orient": "top", "title": "Lap"} if legend else None,
        }

    cursor_filter = {"filter": {"param": "telemetry_cursor", "empty": False}}
    cursor_rule = {
        "transform": [
            cursor_filter,
            {"filter": "datum.series_role === 'Primary'"},
        ],
        "mark": {
            "type": "rule",
            "color": "#f59e0b",
            "strokeWidth": 2,
            "opacity": 0.8,
        },
        "encoding": {"x": graph_x_encoding},
    }

    def single_channel_chart(
        field: str,
        chart_title: str,
        axis_title: str,
        *,
        height: int = 150,
    ) -> dict[str, Any]:
        return {
            "width": chart_width,
            "height": height,
            "title": chart_title,
            "layer": [
                {
                    "mark": {"type": "line", "strokeWidth": 1.8},
                    "encoding": {
                        "x": graph_x_encoding,
                        "y": {
                            "field": field,
                            "type": "quantitative",
                            "title": axis_title,
                            "scale": {"zero": False},
                        },
                        "color": series_colour_encoding(legend=True),
                        "order": {"field": "sample", "type": "quantitative"},
                    },
                },
                cursor_rule,
                {
                    "transform": [cursor_filter],
                    "mark": {
                        "type": "point",
                        "filled": True,
                        "stroke": "white",
                        "strokeWidth": 1,
                        "size": 75,
                    },
                    "encoding": {
                        "x": graph_x_encoding,
                        "y": {"field": field, "type": "quantitative"},
                        "color": series_colour_encoding(legend=False),
                    },
                },
            ],
        }

    def folded_chart(
        fields: list[str],
        labels: list[str],
        colours: list[str],
        chart_title: str,
        axis_title: str,
    ) -> dict[str, Any]:
        channel_colour_encoding = {
            "field": "channel",
            "type": "nominal",
            "title": None,
            "scale": {"domain": fields, "range": colours},
            "legend": {
                "orient": "top-right",
                "labelExpr": "{" + ",".join(
                    f"'{field}':'{label}'" for field, label in zip(fields, labels)
                ) + "}[datum.label]",
            },
        }
        channel_dash_encoding = {
            "field": "channel",
            "type": "nominal",
            "title": "Signal",
            "scale": {"domain": fields, "range": [[1, 0], [7, 3]]},
            "legend": {
                "orient": "top-right",
                "labelExpr": "{" + ",".join(
                    f"'{field}':'{label}'" for field, label in zip(fields, labels)
                ) + "}[datum.label]",
            },
        }
        fold = {"fold": fields, "as": ["channel", "value"]}
        line_colour = (
            series_colour_encoding(legend=True)
            if has_comparison
            else channel_colour_encoding
        )
        return {
            "width": chart_width,
            "height": 135,
            "title": chart_title,
            "layer": [
                {
                    "transform": [fold],
                    "mark": {"type": "line", "strokeWidth": 1.35},
                    "encoding": {
                        "x": graph_x_encoding,
                        "y": {
                            "field": "value",
                            "type": "quantitative",
                            "title": axis_title,
                            "scale": {"zero": False},
                        },
                        "color": line_colour,
                        "order": {"field": "sample", "type": "quantitative"},
                        **({"strokeDash": channel_dash_encoding} if has_comparison else {}),
                    },
                },
                cursor_rule,
                {
                    "transform": [cursor_filter, fold],
                    "mark": {
                        "type": "point",
                        "filled": True,
                        "stroke": "white",
                        "strokeWidth": 1,
                        "size": 65,
                    },
                    "encoding": {
                        "x": graph_x_encoding,
                        "y": {"field": "value", "type": "quantitative"},
                        "color": (
                            series_colour_encoding(legend=False)
                            if has_comparison
                            else channel_colour_encoding
                        ),
                        **(
                            {
                                "shape": {
                                    "field": "channel",
                                    "type": "nominal",
                                    "scale": {
                                        "domain": fields,
                                        "range": ["circle", "diamond"],
                                    },
                                    "legend": None,
                                }
                            }
                            if has_comparison
                            else {}
                        ),
                    },
                },
            ],
        }

    track_x = {
        "field": "east_m",
        "type": "quantitative",
        "title": "East from lap centre (m)",
        "scale": {"domain": east_domain, "nice": False, "zero": False},
    }
    track_y = {
        "field": "north_m",
        "type": "quantitative",
        "title": "North from lap centre (m)",
        "scale": {"domain": north_domain, "nice": False, "zero": False},
    }
    tooltip = [
        {"field": "series", "type": "nominal", "title": "Lap"},
        {"field": "time_s", "type": "quantitative", "title": "Lap time (s)", "format": ".3f"},
        {"field": "distance_m", "type": "quantitative", "title": "Distance (m)", "format": ".0f"},
        {"field": "progress_pct", "type": "quantitative", "title": "Lap progress (%)", "format": ".1f"},
        {
            "field": "speed",
            "type": "quantitative",
            "title": f"Speed ({speed_unit})",
            "format": ".1f",
        },
        {"field": "throttle", "type": "quantitative", "title": "Throttle (%)", "format": ".1f"},
        {"field": "brake", "type": "quantitative", "title": "Brake (bar)", "format": ".1f"},
        {"field": "lateral_g", "type": "quantitative", "title": "Lateral (g)", "format": "+.2f"},
        {
            "field": "longitudinal_g",
            "type": "quantitative",
            "title": "Longitudinal (g)",
            "format": "+.2f",
        },
        {"field": "power", "type": "quantitative", "title": "Power (kW)", "format": "+.1f"},
        {"field": "steering", "type": "quantitative", "title": "Steering (deg)", "format": "+.1f"},
        {"field": "latitude", "type": "quantitative", "title": "Latitude", "format": ".6f"},
        {"field": "longitude", "type": "quantitative", "title": "Longitude", "format": ".6f"},
    ]
    primary_filter = {"filter": "datum.series_role === 'Primary'"}
    last_sample = len(lap_data) - 1
    track_chart = {
        "width": chart_width,
        "height": track_height,
        "title": "Track map — move or drag the pointer along the trace",
        "layer": [
            {
                "transform": [primary_filter],
                "mark": {
                    "type": "line",
                    "color": "#94a3b8",
                    "strokeWidth": 5,
                    "strokeCap": "round",
                    "strokeJoin": "round",
                },
                "encoding": {
                    "x": track_x,
                    "y": track_y,
                    "order": {"field": "sample", "type": "quantitative"},
                },
            },
            {
                "transform": [primary_filter],
                "mark": {"type": "point", "filled": True, "size": 16, "opacity": 0.8},
                "encoding": {
                    "x": track_x,
                    "y": track_y,
                    "color": {
                        "field": "speed",
                        "type": "quantitative",
                        "title": f"Speed ({speed_unit})",
                        "scale": {"scheme": "turbo"},
                    },
                },
            },
            {
                "transform": [primary_filter],
                "params": [
                    {
                        "name": "telemetry_cursor",
                        "value": [{"sample": 0}],
                        "select": {
                            "type": "point",
                            "fields": ["sample"],
                            "on": "pointermove",
                            "nearest": True,
                            "clear": False,
                            "toggle": False,
                        },
                    }
                ],
                "mark": {"type": "point", "filled": True, "size": 110, "opacity": 0.001},
                "encoding": {"x": track_x, "y": track_y, "tooltip": tooltip},
            },
            {
                "transform": [
                    {"filter": "datum.series_role === 'Primary' && datum.sample === 0"}
                ],
                "mark": {
                    "type": "point",
                    "filled": True,
                    "size": 90,
                    "color": "white",
                    "stroke": "#111827",
                    "strokeWidth": 2,
                },
                "encoding": {"x": track_x, "y": track_y},
            },
            {
                "transform": [
                    {
                        "filter": (
                            "datum.series_role === 'Primary' && "
                            f"datum.sample === {last_sample}"
                        )
                    }
                ],
                "mark": {
                    "type": "point",
                    "filled": True,
                    "shape": "cross",
                    "size": 120,
                    "color": "#111827",
                },
                "encoding": {"x": track_x, "y": track_y},
            },
            {
                "transform": [primary_filter, cursor_filter],
                "mark": {
                    "type": "point",
                    "filled": True,
                    "size": 260,
                    "color": "#facc15",
                    "stroke": "#111827",
                    "strokeWidth": 2.5,
                },
                "encoding": {"x": track_x, "y": track_y, "tooltip": tooltip},
            },
            {
                "transform": [
                    primary_filter,
                    cursor_filter,
                    {
                        "calculate": (
                            f"'t ' + format(datum.time_s, '.3f') + ' s · ' + "
                            f"format(datum.speed, '.1f') + ' {speed_unit}'"
                        ),
                        "as": "cursor_label",
                    },
                ],
                "mark": {
                    "type": "text",
                    "align": "left",
                    "dx": 13,
                    "dy": -13,
                    "fontSize": 12,
                    "fontWeight": "bold",
                    "color": "#111827",
                },
                "encoding": {
                    "x": track_x,
                    "y": track_y,
                    "text": {"field": "cursor_label", "type": "nominal"},
                },
            },
        ],
    }

    optional_specs = {
        "driver_inputs": folded_chart(
            ["throttle", "brake"],
            ["Throttle (%)", "Brake (bar)"],
            ["#15803d", "#dc2626"],
            "Driver inputs",
            "Percent / bar",
        ),
        "vehicle_dynamics": folded_chart(
            ["lateral_g", "longitudinal_g"],
            ["Lateral", "Longitudinal"],
            ["#7c3aed", "#0891b2"],
            "Vehicle dynamics",
            "Acceleration (g)",
        ),
        "power_steering": folded_chart(
            ["power", "steering"],
            ["Power (kW)", "Steering (deg)"],
            ["#ea580c", "#475569"],
            "Power and steering",
            "kW / degrees",
        ),
    }
    speed_title = "Speed comparison" if has_comparison else f"{title} — Speed"
    speed_chart = single_channel_chart(
        "speed",
        speed_title,
        f"Speed ({speed_unit})",
    )
    charts = [
        optional_specs[group]
        for group in OPTIONAL_PLOT_GROUPS
        if group in selected_optional
    ]
    charts.extend([speed_chart, track_chart])

    return {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "description": "Linked Tesla Track Mode lap comparison and GPS trace.",
        "data": {"values": records},
        "vconcat": charts,
        "spacing": 12,
        "resolve": {"scale": {"color": "independent"}},
        "config": {
            "background": "white",
            "view": {"stroke": "#d8dee5"},
            "axis": {
                "gridColor": "#e5e7eb",
                "domainColor": "#94a3b8",
                "labelColor": "#334e68",
                "titleColor": "#243b53",
            },
            "title": {
                "anchor": "start",
                "color": "#243b53",
                "font": "system-ui",
                "fontSize": 14,
            },
            "legend": {
                "labelColor": "#334e68",
                "titleColor": "#243b53",
                "orient": "right",
            },
        },
    }


class TelemetryDashboard:
    """Jupyter dashboard with session/lap selectors and a draggable GPS handle."""

    def __init__(
        self,
        data_dir: str | Path = Path("data/20260714"),
        *,
        n_sectors: int = 3,
        default_file: str | Path | None = None,
    ) -> None:
        if not isinstance(n_sectors, int) or n_sectors < 1:
            raise ValueError("n_sectors must be a positive integer")

        self.data_dir = Path(data_dir)
        self.n_sectors = n_sectors
        self.telemetry = pd.DataFrame()
        self.timed = pd.DataFrame()
        self.lap_summary = pd.DataFrame()
        self.lap_data = pd.DataFrame()
        self.comparison_telemetry = pd.DataFrame()
        self.comparison_timed = pd.DataFrame()
        self.comparison_lap_summary = pd.DataFrame()
        self.comparison_lap_data = pd.DataFrame()
        self.current_index = 0
        self.plot_spec: dict[str, Any] | None = None
        self.plot_handle: Any | None = None
        self._changing_controls = False

        self.file_dropdown = widgets.Dropdown(
            description="Telemetry CSV",
            layout=widgets.Layout(width="620px"),
            style={"description_width": "110px"},
        )
        self.refresh_button = widgets.Button(
            description="Refresh files",
            icon="refresh",
            tooltip="Rescan the configured data directory",
            layout=widgets.Layout(width="145px"),
        )
        self.lap_dropdown = widgets.Dropdown(
            description="Lap",
            disabled=True,
            layout=widgets.Layout(width="330px"),
            style={"description_width": "50px"},
        )
        self.speed_unit = widgets.ToggleButtons(
            options=[("km/h", "km/h"), ("MPH", "MPH")],
            value="km/h",
            description="Speed",
            style={"description_width": "55px"},
        )
        self.optional_plot_checkboxes = {
            "driver_inputs": widgets.Checkbox(
                value=False,
                description="Driver inputs",
                indent=False,
                layout=widgets.Layout(width="145px"),
            ),
            "vehicle_dynamics": widgets.Checkbox(
                value=False,
                description="Vehicle dynamics",
                indent=False,
                layout=widgets.Layout(width="165px"),
            ),
            "power_steering": widgets.Checkbox(
                value=False,
                description="Power and steering",
                indent=False,
                layout=widgets.Layout(width="185px"),
            ),
        }
        self.comparison_enabled = widgets.Checkbox(
            value=False,
            description="Compare another lap",
            indent=False,
            layout=widgets.Layout(width="190px"),
        )
        self.comparison_file_dropdown = widgets.Dropdown(
            description="Comparison CSV",
            disabled=True,
            layout=widgets.Layout(width="620px"),
            style={"description_width": "110px"},
        )
        self.comparison_lap_dropdown = widgets.Dropdown(
            description="Comparison lap",
            disabled=True,
            layout=widgets.Layout(width="330px"),
            style={"description_width": "110px"},
        )
        self.message = widgets.HTML()
        self.summary = widgets.HTML()
        self.plot_output = widgets.Output(layout=widgets.Layout(width="100%"))

        self.interaction_note = widgets.HTML(
            value=(
                "<span style='color:#46635b'>Move or drag the pointer along the track trace. "
                "Speed is always directly above the map; optional graphs appear above speed. "
                "The yellow marker moves the cursor rule and value markers on every graph. "
                "A comparison marker represents the same GPS-progress point on the other lap. "
                "Comparison graph axes use lap distance so the same corners line up. "
                "Hover for exact telemetry. "
                "This uses VS Code's bundled Vega-Lite renderer—no widget CDN is required.</span>"
            )
        )

        optional_controls = widgets.HBox(
            [
                widgets.HTML(
                    value="<b>Graphs:</b> Speed (always shown)&nbsp;&nbsp; Optional:",
                    layout=widgets.Layout(width="285px"),
                ),
                *self.optional_plot_checkboxes.values(),
            ]
        )

        self.widget = widgets.VBox(
            [
                widgets.HBox([self.file_dropdown, self.refresh_button]),
                widgets.HBox([self.lap_dropdown, self.speed_unit]),
                optional_controls,
                self.comparison_enabled,
                widgets.HBox(
                    [self.comparison_file_dropdown, self.comparison_lap_dropdown]
                ),
                self.interaction_note,
                self.message,
                self.summary,
                self.plot_output,
            ]
        )

        self.file_dropdown.observe(self._on_file_change, names="value")
        self.lap_dropdown.observe(self._on_lap_change, names="value")
        self.speed_unit.observe(self._on_speed_unit_change, names="value")
        for checkbox in self.optional_plot_checkboxes.values():
            checkbox.observe(self._on_optional_plot_change, names="value")
        self.comparison_enabled.observe(
            self._on_comparison_enabled_change, names="value"
        )
        self.comparison_file_dropdown.observe(
            self._on_comparison_file_change, names="value"
        )
        self.comparison_lap_dropdown.observe(
            self._on_comparison_lap_change, names="value"
        )
        self.refresh_button.on_click(self._on_refresh_click)
        self.refresh_files(default_file=default_file)

    @property
    def selected_file(self) -> Path | None:
        value = self.file_dropdown.value
        return Path(value) if value else None

    @property
    def selected_lap(self) -> int | None:
        value = self.lap_dropdown.value
        return int(value) if value is not None else None

    @property
    def selected_comparison_file(self) -> Path | None:
        value = self.comparison_file_dropdown.value
        return Path(value) if value else None

    @property
    def selected_comparison_lap(self) -> int | None:
        value = self.comparison_lap_dropdown.value
        return int(value) if value is not None else None

    @property
    def selected_optional_charts(self) -> tuple[str, ...]:
        return tuple(
            group
            for group in OPTIONAL_PLOT_GROUPS
            if self.optional_plot_checkboxes[group].value
        )

    def display(self) -> None:
        """Display the dashboard in a notebook cell."""

        display(self.widget)
        self._update_plot_display(create=True)

    def refresh_files(self, default_file: str | Path | None = None) -> None:
        """Rescan ``data_dir`` and load the requested or newest CSV."""

        try:
            files = discover_telemetry_files(self.data_dir)
        except Exception as exc:
            self._set_error(str(exc))
            self._changing_controls = True
            self.file_dropdown.options = []
            self.comparison_file_dropdown.options = []
            self.comparison_file_dropdown.disabled = True
            self.comparison_lap_dropdown.options = []
            self.comparison_lap_dropdown.disabled = True
            self._changing_controls = False
            return

        requested = Path(default_file) if default_file is not None else files[-1]
        requested_name = requested.name
        selected = next((path for path in files if path.name == requested_name), files[-1])
        previous_comparison = self.selected_comparison_file
        comparison_name = previous_comparison.name if previous_comparison else selected.name
        comparison_selected = next(
            (path for path in files if path.name == comparison_name), selected
        )
        file_options = [(path.name, str(path)) for path in files]

        self._changing_controls = True
        self.file_dropdown.options = file_options
        self.file_dropdown.value = str(selected)
        self.comparison_file_dropdown.options = file_options
        self.comparison_file_dropdown.value = str(comparison_selected)
        self.comparison_file_dropdown.disabled = not self.comparison_enabled.value
        self._changing_controls = False
        if self.comparison_enabled.value:
            self._load_comparison_file(redraw=False)
        else:
            self.comparison_telemetry = pd.DataFrame()
            self.comparison_timed = pd.DataFrame()
            self.comparison_lap_summary = pd.DataFrame()
            self.comparison_lap_data = pd.DataFrame()
            self._changing_controls = True
            self.comparison_lap_dropdown.options = [
                ("Enable comparison to choose a lap", None)
            ]
            self.comparison_lap_dropdown.value = None
            self.comparison_lap_dropdown.disabled = True
            self._changing_controls = False
        self._load_selected_file()

    def _on_refresh_click(self, _: widgets.Button) -> None:
        self.refresh_files(default_file=self.selected_file)

    def _on_file_change(self, change: dict[str, Any]) -> None:
        if not self._changing_controls and change.get("new"):
            self._load_selected_file()

    def _on_lap_change(self, change: dict[str, Any]) -> None:
        if not self._changing_controls and change.get("new") is not None:
            self._draw_selected_lap()

    def _on_speed_unit_change(self, _: dict[str, Any]) -> None:
        if not self._changing_controls and self.selected_lap is not None:
            self._draw_selected_lap()

    def _on_optional_plot_change(self, _: dict[str, Any]) -> None:
        if not self._changing_controls and self.selected_lap is not None:
            self._draw_selected_lap()

    def _on_comparison_enabled_change(self, change: dict[str, Any]) -> None:
        if self._changing_controls:
            return
        enabled = bool(change.get("new"))
        self.comparison_file_dropdown.disabled = not enabled
        if enabled:
            self._load_comparison_file(redraw=False)
        else:
            self.comparison_lap_dropdown.disabled = True
            self.comparison_lap_data = pd.DataFrame()
            self.message.value = ""
        if self.selected_lap is not None:
            self._draw_selected_lap()

    def _on_comparison_file_change(self, change: dict[str, Any]) -> None:
        if not self._changing_controls and change.get("new"):
            self._load_comparison_file()

    def _on_comparison_lap_change(self, change: dict[str, Any]) -> None:
        if (
            not self._changing_controls
            and self.comparison_enabled.value
            and change.get("new") is not None
            and self.selected_lap is not None
        ):
            self._draw_selected_lap()

    @staticmethod
    def _lap_options(summary: pd.DataFrame) -> list[tuple[str, int]]:
        fastest = fastest_lap_id(summary)
        options = []
        for _, row in summary.iterrows():
            lap = int(row["Lap"])
            suffix = " · fastest" if lap == fastest else ""
            options.append((f"Lap {lap} — {row['Lap Time']}{suffix}", lap))
        return options

    def _default_comparison_lap(self, summary: pd.DataFrame) -> int:
        ordered = summary.sort_values("Lap Time (ms)")
        lap_ids = ordered["Lap"].astype(int).tolist()
        if (
            self.selected_file == self.selected_comparison_file
            and self.selected_lap in lap_ids
            and len(lap_ids) > 1
        ):
            return next(lap for lap in lap_ids if lap != self.selected_lap)
        return lap_ids[0]

    def _load_comparison_file(self, *, redraw: bool = True) -> None:
        path = self.selected_comparison_file
        if path is None:
            return

        previous_lap = self.selected_comparison_lap
        self.message.value = f"<i>Loading comparison {escape(path.name)}…</i>"
        try:
            telemetry = load_telemetry(path)
            timed = timed_laps(telemetry)
            summary = build_lap_summary(timed)
        except Exception as exc:
            self.comparison_telemetry = pd.DataFrame()
            self.comparison_timed = pd.DataFrame()
            self.comparison_lap_summary = pd.DataFrame()
            self.comparison_lap_data = pd.DataFrame()
            self._set_error(f"Comparison file: {exc}")
            self._changing_controls = True
            self.comparison_lap_dropdown.options = [("Comparison unavailable", None)]
            self.comparison_lap_dropdown.value = None
            self.comparison_lap_dropdown.disabled = True
            self._changing_controls = False
            if redraw and self.selected_lap is not None:
                self._draw_selected_lap()
            return

        self.comparison_telemetry = telemetry
        self.comparison_timed = timed
        self.comparison_lap_summary = summary
        self.comparison_lap_data = pd.DataFrame()
        self._changing_controls = True
        if summary.empty:
            self.comparison_lap_dropdown.options = [
                ("No timed laps in this recording", None)
            ]
            self.comparison_lap_dropdown.value = None
            self.comparison_lap_dropdown.disabled = True
        else:
            lap_ids = summary["Lap"].astype(int).tolist()
            selected_lap = (
                previous_lap
                if previous_lap in lap_ids
                else self._default_comparison_lap(summary)
            )
            self.comparison_lap_dropdown.options = self._lap_options(summary)
            self.comparison_lap_dropdown.value = selected_lap
            self.comparison_lap_dropdown.disabled = not self.comparison_enabled.value
        self._changing_controls = False

        if summary.empty:
            self.message.value = (
                "<b>The comparison file has no positive timed laps.</b> "
                "Choose another comparison CSV."
            )
        else:
            self.message.value = ""
        if redraw and self.selected_lap is not None:
            self._draw_selected_lap()

    def _load_selected_file(self) -> None:
        path = self.selected_file
        if path is None:
            return

        self.message.value = f"<i>Loading {escape(path.name)}…</i>"
        try:
            telemetry = load_telemetry(path)
            timed = timed_laps(telemetry)
            summary = build_lap_summary(timed)
        except Exception as exc:
            self._set_error(str(exc))
            self._clear_figure()
            return

        self.telemetry = telemetry
        self.timed = timed
        self.lap_summary = summary
        self.message.value = ""
        self._render_session_summary(path)

        self._changing_controls = True
        if summary.empty:
            self.lap_dropdown.options = [("No timed laps in this recording", None)]
            self.lap_dropdown.value = None
            self.lap_dropdown.disabled = True
            self._changing_controls = False
            self.message.value = (
                "<b>No positive lap IDs with elapsed time were recorded in this file.</b> "
                "Choose another CSV above."
            )
            self._clear_figure()
            return

        fastest = fastest_lap_id(summary)
        self.lap_dropdown.disabled = False
        self.lap_dropdown.options = self._lap_options(summary)
        self.lap_dropdown.value = fastest
        self._changing_controls = False
        self._draw_selected_lap()

    def _render_session_summary(self, path: Path) -> None:
        rows = len(self.telemetry)
        duplicate_rows = int(self.telemetry.drop(columns=["Source"]).duplicated().sum())
        observed_laps = sorted(self.telemetry["Lap"].dropna().astype(int).unique().tolist())
        timed_count = len(self.lap_summary)

        cards = (
            "<div style='display:flex;gap:12px;flex-wrap:wrap;margin:8px 0'>"
            f"<div><b>{rows:,}</b><br><small>samples</small></div>"
            f"<div><b>{timed_count}</b><br><small>timed laps</small></div>"
            f"<div><b>{duplicate_rows:,}</b><br><small>exact duplicates retained</small></div>"
            f"<div><b>{escape(', '.join(map(str, observed_laps)))}</b><br><small>observed lap IDs</small></div>"
            "</div>"
        )

        if self.lap_summary.empty:
            self.summary.value = f"<h4>{escape(path.name)}</h4>{cards}"
            return

        fastest = fastest_lap_id(self.lap_summary)
        fastest_row = self.lap_summary.loc[self.lap_summary["Lap"].eq(fastest)].iloc[0]

        table = self.lap_summary[
            [
                "Lap",
                "Lap Time",
                "Gap to Fastest (s)",
                "Average Speed (km/h)",
                "Maximum Speed (km/h)",
                "Average Throttle (%)",
                "Maximum Brake (bar)",
                "Samples",
            ]
        ].copy()
        table["Gap to Fastest (s)"] = table["Gap to Fastest (s)"].map(lambda value: f"{value:+.3f}")
        for column in (
            "Average Speed (km/h)",
            "Maximum Speed (km/h)",
            "Average Throttle (%)",
            "Maximum Brake (bar)",
        ):
            table[column] = table[column].map(lambda value: f"{value:.1f}")
        lap_table_html = table.to_html(index=False, border=0, classes="telemetry-table")

        sector_text = ""
        try:
            _, best_sectors = build_sector_summary(self.timed, self.n_sectors)
            theoretical_ms = float(best_sectors["Sector Time (ms)"].sum())
            gain_s = (float(fastest_row["Lap Time (ms)"]) - theoretical_ms) / 1_000.0
            sector_parts = [
                f"S{int(row['Sector'])} {format_lap_time(row['Sector Time (ms)'])} "
                f"(Lap {int(row['Lap'])})"
                for _, row in best_sectors.iterrows()
            ]
            sector_text = (
                f"<p><b>{self.n_sectors}-sector theoretical lap:</b> "
                f"{format_lap_time(theoretical_ms)} · potential gain {gain_s:.3f} s<br>"
                f"<small>{escape(' · '.join(sector_parts))}</small></p>"
            )
        except ValueError as exc:
            sector_text = f"<p><small>Sector analysis unavailable: {escape(str(exc))}</small></p>"

        style = """
        <style>
        .telemetry-table {border-collapse:collapse;font-size:12px;}
        .telemetry-table th {background:#243b53;color:white;padding:5px 8px;}
        .telemetry-table td {padding:4px 8px;border-bottom:1px solid #d8dee5;text-align:right;}
        .telemetry-table tr:nth-child(even) {background:#f4f7f9;}
        </style>
        """
        self.summary.value = (
            f"{style}<h4>{escape(path.name)}</h4>{cards}"
            f"<p><b>Fastest recorded lap:</b> Lap {fastest} — {fastest_row['Lap Time']}</p>"
            f"{sector_text}{lap_table_html}"
        )

    def _draw_selected_lap(self) -> None:
        lap = self.selected_lap
        if lap is None:
            return
        try:
            lap_data = prepare_lap(self.timed, lap)
        except Exception as exc:
            self._set_error(str(exc))
            self._clear_figure()
            return

        self.lap_data = lap_data
        comparison_lap_data: pd.DataFrame | None = None
        comparison_title: str | None = None
        if self.comparison_enabled.value:
            comparison_lap = self.selected_comparison_lap
            if comparison_lap is not None:
                try:
                    comparison_lap_data = prepare_lap(
                        self.comparison_timed, comparison_lap
                    )
                    comparison_time = self.comparison_lap_summary.loc[
                        self.comparison_lap_summary["Lap"].eq(comparison_lap),
                        "Lap Time",
                    ].iloc[0]
                    comparison_source = (
                        self.selected_comparison_file.name
                        if self.selected_comparison_file
                        else "telemetry"
                    )
                    comparison_title = (
                        f"{comparison_source} · Lap {comparison_lap} · {comparison_time}"
                    )
                    self.message.value = ""
                except Exception as exc:
                    self._set_error(f"Comparison lap: {exc}")
                    comparison_lap_data = None
        self.comparison_lap_data = (
            comparison_lap_data
            if comparison_lap_data is not None
            else pd.DataFrame()
        )

        lap_time = self.lap_summary.loc[
            self.lap_summary["Lap"].eq(lap), "Lap Time"
        ].iloc[0]
        source = self.selected_file.name if self.selected_file else "telemetry"
        try:
            self.plot_spec = build_telemetry_plot_spec(
                lap_data,
                title=f"{source} · Lap {lap} · {lap_time}",
                speed_unit=self.speed_unit.value,
                optional_charts=self.selected_optional_charts,
                comparison_lap_data=comparison_lap_data,
                comparison_title=comparison_title,
            )
        except Exception as exc:
            self._set_error(str(exc))
            self._clear_figure()
            return
        self._update_plot_display()

    def _update_plot_display(self, *, create: bool = False) -> None:
        if self.plot_spec is None:
            return
        bundle = {
            "application/vnd.vegalite.v5+json": self.plot_spec,
            "text/plain": (
                "Interactive telemetry view. Open this notebook in VS Code or "
                "JupyterLab with Vega-Lite support."
            ),
        }
        self.plot_output.clear_output(wait=True)
        with self.plot_output:
            display(bundle, raw=True)

    def _clear_figure(self) -> None:
        """Clear the current plot while retaining the historical method name."""

        self.plot_spec = None
        self.lap_data = pd.DataFrame()
        self.comparison_lap_data = pd.DataFrame()
        self.plot_output.clear_output(wait=True)
        if self.plot_handle is not None:
            self.plot_handle.update({"text/plain": ""}, raw=True)

    def _set_error(self, message: str) -> None:
        self.message.value = (
            "<div style='padding:8px;border-left:4px solid #b42318;background:#fff1f0'>"
            f"<b>Telemetry error:</b> {escape(message)}</div>"
        )


__all__ = [
    "GRAPH_CHANNELS",
    "OPTIONAL_PLOT_GROUPS",
    "REQUIRED_COLUMNS",
    "TelemetryDashboard",
    "build_lap_summary",
    "build_sector_summary",
    "build_telemetry_plot_spec",
    "discover_telemetry_files",
    "fastest_lap_id",
    "format_lap_time",
    "lap_sector_times",
    "load_telemetry",
    "nearest_sample_index",
    "prepare_lap",
    "timed_laps",
]

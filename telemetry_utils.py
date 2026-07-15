"""Utilities and interactive widgets for Tesla Track Mode telemetry.

The data-processing functions in this module are deliberately independent from
the notebook so they can be tested and reused from scripts.  ``TelemetryDashboard``
adds the Jupyter-specific controls and synchronized Matplotlib views.
"""

from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import clear_output, display
from matplotlib.collections import LineCollection
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
        self.current_index = 0
        self.fig: Any | None = None
        self.ax_track: Any | None = None
        self.position_handle: Any | None = None
        self._connection_ids: list[int] = []
        self._cursor_lines: list[Any] = []
        self._value_markers: list[tuple[Any, np.ndarray]] = []
        self._dragging = False
        self._changing_controls = False
        self._syncing_slider = False

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
        self.time_slider = widgets.FloatSlider(
            description="Time (s)",
            disabled=True,
            continuous_update=True,
            readout_format=".3f",
            layout=widgets.Layout(width="760px"),
            style={"description_width": "65px"},
        )
        self.message = widgets.HTML()
        self.summary = widgets.HTML()
        self.position_readout = widgets.HTML()
        self.figure_output = widgets.Output()

        backend = matplotlib.get_backend().lower()
        if "ipympl" in backend or "widget" in backend:
            backend_note = (
                "<span style='color:#46635b'>Drag the yellow handle on the map; "
                "it snaps to the nearest recorded GPS sample. Click the track or drag the time "
                "slider for precise positioning.</span>"
            )
        else:
            backend_note = (
                "<b style='color:#b42318'>The current Matplotlib backend is not interactive. "
                "Install ipympl and run <code>%matplotlib widget</code> before creating the dashboard.</b>"
            )
        self.interaction_note = widgets.HTML(value=backend_note)

        self.widget = widgets.VBox(
            [
                widgets.HBox([self.file_dropdown, self.refresh_button]),
                widgets.HBox([self.lap_dropdown, self.speed_unit]),
                self.time_slider,
                self.interaction_note,
                self.message,
                self.position_readout,
                self.summary,
                self.figure_output,
            ]
        )

        self.file_dropdown.observe(self._on_file_change, names="value")
        self.lap_dropdown.observe(self._on_lap_change, names="value")
        self.speed_unit.observe(self._on_speed_unit_change, names="value")
        self.time_slider.observe(self._on_time_change, names="value")
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

    def display(self) -> None:
        """Display the dashboard in a notebook cell."""

        display(self.widget)

    def refresh_files(self, default_file: str | Path | None = None) -> None:
        """Rescan ``data_dir`` and load the requested or newest CSV."""

        try:
            files = discover_telemetry_files(self.data_dir)
        except Exception as exc:
            self._set_error(str(exc))
            self._changing_controls = True
            self.file_dropdown.options = []
            self._changing_controls = False
            return

        requested = Path(default_file) if default_file is not None else files[-1]
        requested_name = requested.name
        selected = next((path for path in files if path.name == requested_name), files[-1])

        self._changing_controls = True
        self.file_dropdown.options = [(path.name, str(path)) for path in files]
        self.file_dropdown.value = str(selected)
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

    def _on_time_change(self, change: dict[str, Any]) -> None:
        if self._syncing_slider or self.lap_data.empty:
            return
        time_values = self.lap_data["Elapsed (s)"].to_numpy(dtype=float)
        requested = float(change["new"])
        insertion = int(np.searchsorted(time_values, requested, side="left"))
        candidates = [max(0, insertion - 1), min(len(time_values) - 1, insertion)]
        index = min(candidates, key=lambda candidate: abs(time_values[candidate] - requested))
        self._update_selection(index, sync_slider=False)

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
            self.time_slider.disabled = True
            self._changing_controls = False
            self.position_readout.value = (
                "<b>No positive lap IDs with elapsed time were recorded in this file.</b> "
                "Choose another CSV above."
            )
            self._clear_figure()
            return

        fastest = fastest_lap_id(summary)
        options = []
        for _, row in summary.iterrows():
            lap = int(row["Lap"])
            suffix = " · fastest" if lap == fastest else ""
            options.append((f"Lap {lap} — {row['Lap Time']}{suffix}", lap))
        self.lap_dropdown.disabled = False
        self.lap_dropdown.options = options
        self.lap_dropdown.value = fastest
        self.time_slider.disabled = False
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

        self._disconnect_figure()
        if self.fig is not None:
            plt.close(self.fig)

        self.lap_data = lap_data
        time_values = lap_data["Elapsed (s)"].to_numpy(dtype=float)
        speed_column = "Speed (km/h)" if self.speed_unit.value == "km/h" else "Speed (MPH)"
        speed_values = _channel(lap_data, speed_column)
        throttle_values = _channel(lap_data, "Throttle Position (%)")
        brake_values = np.clip(_channel(lap_data, "Brake Pressure (bar)"), 0.0, None)
        lateral_g = _channel(lap_data, "Lateral Acceleration (g)")
        longitudinal_g = _channel(lap_data, "Longitudinal Acceleration (g)")
        power_values = _channel(lap_data, "Power Level (KW)")
        steering_values = _channel(lap_data, "Steering Angle (deg)")
        longitude = lap_data["Longitude (decimal)"].to_numpy(dtype=float)
        latitude = lap_data["Latitude (decimal)"].to_numpy(dtype=float)

        with plt.ioff():
            fig = plt.figure(figsize=(14, 13), constrained_layout=True)
        grid = fig.add_gridspec(5, 1, height_ratios=[1.35, 1.0, 1.0, 1.0, 3.1])
        ax_speed = fig.add_subplot(grid[0])
        ax_inputs = fig.add_subplot(grid[1], sharex=ax_speed)
        ax_dynamics = fig.add_subplot(grid[2], sharex=ax_speed)
        ax_power = fig.add_subplot(grid[3], sharex=ax_speed)
        ax_track = fig.add_subplot(grid[4])
        ax_brake = ax_inputs.twinx()
        ax_steering = ax_power.twinx()

        lap_time = self.lap_summary.loc[self.lap_summary["Lap"].eq(lap), "Lap Time"].iloc[0]
        source = self.selected_file.name if self.selected_file else "telemetry"
        fig.suptitle(f"{source} · Lap {lap} · {lap_time}", fontsize=15, fontweight="bold")

        ax_speed.plot(time_values, speed_values, color="#1d4ed8", linewidth=1.7)
        speed_point, = ax_speed.plot([], [], "o", color="#f59e0b", markersize=6, zorder=5)
        ax_speed.set_ylabel(f"Speed ({self.speed_unit.value})")
        ax_speed.set_title("Speed")

        ax_inputs.plot(time_values, throttle_values, color="#15803d", linewidth=1.4, label="Throttle")
        ax_inputs.fill_between(time_values, 0, throttle_values, color="#22c55e", alpha=0.12)
        throttle_point, = ax_inputs.plot([], [], "o", color="#15803d", markersize=5, zorder=5)
        ax_inputs.set_ylabel("Throttle (%)", color="#15803d")
        ax_inputs.set_ylim(-3, 103)
        ax_brake.plot(time_values, brake_values, color="#dc2626", linewidth=1.25, label="Brake")
        brake_point, = ax_brake.plot([], [], "o", color="#dc2626", markersize=5, zorder=5)
        ax_brake.set_ylabel("Brake (bar)", color="#dc2626")
        ax_inputs.set_title("Driver inputs")

        ax_dynamics.plot(time_values, lateral_g, color="#7c3aed", linewidth=1.25, label="Lateral")
        ax_dynamics.plot(
            time_values, longitudinal_g, color="#0891b2", linewidth=1.25, label="Longitudinal"
        )
        lateral_point, = ax_dynamics.plot([], [], "o", color="#7c3aed", markersize=5, zorder=5)
        longitudinal_point, = ax_dynamics.plot([], [], "o", color="#0891b2", markersize=5, zorder=5)
        ax_dynamics.axhline(0, color="#94a3b8", linewidth=0.8)
        ax_dynamics.set_ylabel("Acceleration (g)")
        ax_dynamics.set_title("Vehicle dynamics")
        ax_dynamics.legend(loc="upper right", ncol=2, frameon=True)

        ax_power.plot(time_values, power_values, color="#ea580c", linewidth=1.25)
        power_point, = ax_power.plot([], [], "o", color="#ea580c", markersize=5, zorder=5)
        ax_power.axhline(0, color="#94a3b8", linewidth=0.8)
        ax_power.set_ylabel("Power (kW)", color="#ea580c")
        ax_power.set_xlabel("Elapsed lap time (s)")
        ax_power.set_title("Power and steering")
        ax_steering.plot(time_values, steering_values, color="#475569", linewidth=0.95, alpha=0.8)
        steering_point, = ax_steering.plot([], [], "o", color="#475569", markersize=4, zorder=5)
        ax_steering.set_ylabel("Steering (deg)", color="#475569")

        for axis in (ax_speed, ax_inputs, ax_dynamics, ax_power):
            axis.grid(True, color="#d8dee5", linewidth=0.7, alpha=0.8)
            axis.margins(x=0)

        points = np.column_stack([longitude, latitude])
        segments = np.stack([points[:-1], points[1:]], axis=1)
        segment_speeds = (speed_values[:-1] + speed_values[1:]) / 2.0
        finite_speeds = segment_speeds[np.isfinite(segment_speeds)]
        speed_min = float(np.nanmin(finite_speeds)) if len(finite_speeds) else 0.0
        speed_max = float(np.nanmax(finite_speeds)) if len(finite_speeds) else 1.0
        if np.isclose(speed_min, speed_max):
            speed_max = speed_min + 1.0
        track_collection = LineCollection(
            segments,
            cmap="turbo",
            norm=plt.Normalize(speed_min, speed_max),
            linewidth=4.0,
            capstyle="round",
            zorder=2,
        )
        track_collection.set_array(segment_speeds)
        ax_track.add_collection(track_collection)
        colorbar = fig.colorbar(track_collection, ax=ax_track, pad=0.015, fraction=0.035)
        colorbar.set_label(f"Speed ({self.speed_unit.value})")

        ax_track.plot(
            longitude[0], latitude[0], marker="o", markersize=8, markerfacecolor="white",
            markeredgecolor="black", linestyle="None", label="Start", zorder=5
        )
        ax_track.plot(
            longitude[-1], latitude[-1], marker="X", markersize=8, color="black",
            linestyle="None", label="End", zorder=5
        )
        position_handle, = ax_track.plot(
            [longitude[0]],
            [latitude[0]],
            marker="o",
            markersize=14,
            markerfacecolor="#facc15",
            markeredgecolor="#111827",
            markeredgewidth=2.0,
            linestyle="None",
            label="Drag position",
            picker=12,
            zorder=8,
        )
        longitude_span = max(float(np.ptp(longitude)), 1e-6)
        latitude_span = max(float(np.ptp(latitude)), 1e-6)
        ax_track.set_xlim(float(np.min(longitude)) - longitude_span * 0.04, float(np.max(longitude)) + longitude_span * 0.04)
        ax_track.set_ylim(float(np.min(latitude)) - latitude_span * 0.04, float(np.max(latitude)) + latitude_span * 0.04)
        mean_latitude = float(np.mean(latitude))
        ax_track.set_aspect(1.0 / max(np.cos(np.radians(mean_latitude)), 0.1))
        ax_track.set_xlabel("Longitude")
        ax_track.set_ylabel("Latitude")
        ax_track.set_title("Track map — drag the yellow handle or click anywhere on the trace")
        ax_track.grid(True, color="#d8dee5", linewidth=0.7, alpha=0.7)
        ax_track.legend(loc="best", frameon=True)

        cursor_lines = [
            axis.axvline(time_values[0], color="#f59e0b", linewidth=1.35, alpha=0.95, zorder=4)
            for axis in (ax_speed, ax_inputs, ax_dynamics, ax_power)
        ]

        self.fig = fig
        self.ax_track = ax_track
        self.position_handle = position_handle
        self._cursor_lines = cursor_lines
        self._value_markers = [
            (speed_point, speed_values),
            (throttle_point, throttle_values),
            (brake_point, brake_values),
            (lateral_point, lateral_g),
            (longitudinal_point, longitudinal_g),
            (power_point, power_values),
            (steering_point, steering_values),
        ]
        self._connection_ids = [
            fig.canvas.mpl_connect("button_press_event", self._on_map_press),
            fig.canvas.mpl_connect("motion_notify_event", self._on_map_motion),
            fig.canvas.mpl_connect("button_release_event", self._on_map_release),
        ]

        positive_steps = np.diff(time_values)
        positive_steps = positive_steps[positive_steps > 0]
        slider_step = float(np.median(positive_steps)) if len(positive_steps) else 0.001
        self._syncing_slider = True
        self.time_slider.min = float(time_values[0])
        self.time_slider.max = float(time_values[-1])
        self.time_slider.step = max(slider_step, 0.001)
        self.time_slider.value = float(time_values[0])
        self._syncing_slider = False

        self._update_selection(0)
        with self.figure_output:
            clear_output(wait=True)
            backend = matplotlib.get_backend().lower()
            if "ipympl" in backend or "widget" in backend:
                display(fig.canvas)
            else:
                display(fig)

    def _toolbar_is_active(self) -> bool:
        if self.fig is None:
            return False
        toolbar = getattr(self.fig.canvas, "toolbar", None)
        mode = getattr(toolbar, "mode", "") if toolbar is not None else ""
        return bool(str(mode))

    def _on_map_press(self, event: Any) -> None:
        if (
            event.button != 1
            or event.inaxes is not self.ax_track
            or event.xdata is None
            or event.ydata is None
            or self._toolbar_is_active()
        ):
            return
        self._dragging = True
        self._select_nearest(float(event.xdata), float(event.ydata))

    def _on_map_motion(self, event: Any) -> None:
        if (
            not self._dragging
            or event.inaxes is not self.ax_track
            or event.xdata is None
            or event.ydata is None
        ):
            return
        self._select_nearest(float(event.xdata), float(event.ydata))

    def _on_map_release(self, _: Any) -> None:
        self._dragging = False

    def _select_nearest(self, longitude: float, latitude: float) -> None:
        index = nearest_sample_index(
            self.lap_data,
            longitude,
            latitude,
            current_index=self.current_index,
        )
        self._update_selection(index)

    def _update_selection(self, index: int, *, sync_slider: bool = True) -> None:
        if self.lap_data.empty or self.fig is None:
            return
        index = int(np.clip(index, 0, len(self.lap_data) - 1))
        self.current_index = index
        row = self.lap_data.iloc[index]
        elapsed_s = float(row["Elapsed (s)"])
        longitude = float(row["Longitude (decimal)"])
        latitude = float(row["Latitude (decimal)"])

        self.position_handle.set_data([longitude], [latitude])
        for cursor in self._cursor_lines:
            cursor.set_xdata([elapsed_s, elapsed_s])
        for marker, values in self._value_markers:
            marker.set_data([elapsed_s], [values[index]])

        if sync_slider:
            self._syncing_slider = True
            self.time_slider.value = elapsed_s
            self._syncing_slider = False

        speed_column = "Speed (km/h)" if self.speed_unit.value == "km/h" else "Speed (MPH)"
        speed = float(row[speed_column])
        throttle = float(row["Throttle Position (%)"])
        brake = max(0.0, float(row["Brake Pressure (bar)"]))
        lateral_g = float(row.get("Lateral Acceleration (g)", np.nan))
        longitudinal_g = float(row.get("Longitudinal Acceleration (g)", np.nan))
        power = float(row.get("Power Level (KW)", np.nan))
        steering = float(row.get("Steering Angle (deg)", np.nan))
        distance = float(row["GPS Distance (m)"])

        self.position_readout.value = (
            "<div style='padding:7px 10px;background:#f4f7f9;border-left:4px solid #facc15'>"
            f"<b>Lap {self.selected_lap} · {elapsed_s:.3f} s · {distance:.0f} m</b> &nbsp; "
            f"Speed {speed:.1f} {self.speed_unit.value} · Throttle {throttle:.1f}% · "
            f"Brake {brake:.1f} bar · Lat {lateral_g:+.2f} g · Long {longitudinal_g:+.2f} g · "
            f"Power {power:+.1f} kW · Steering {steering:+.1f}°"
            "</div>"
        )
        self.fig.canvas.draw_idle()

    def _disconnect_figure(self) -> None:
        if self.fig is not None:
            for connection_id in self._connection_ids:
                self.fig.canvas.mpl_disconnect(connection_id)
        self._connection_ids = []
        self._dragging = False

    def _clear_figure(self) -> None:
        self._disconnect_figure()
        if self.fig is not None:
            plt.close(self.fig)
        self.fig = None
        self.lap_data = pd.DataFrame()
        with self.figure_output:
            clear_output(wait=True)

    def _set_error(self, message: str) -> None:
        self.message.value = (
            "<div style='padding:8px;border-left:4px solid #b42318;background:#fff1f0'>"
            f"<b>Telemetry error:</b> {escape(message)}</div>"
        )


__all__ = [
    "GRAPH_CHANNELS",
    "REQUIRED_COLUMNS",
    "TelemetryDashboard",
    "build_lap_summary",
    "build_sector_summary",
    "discover_telemetry_files",
    "fastest_lap_id",
    "format_lap_time",
    "lap_sector_times",
    "load_telemetry",
    "nearest_sample_index",
    "prepare_lap",
    "timed_laps",
]

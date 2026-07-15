"""Video overlays for Tesla Track Mode telemetry recordings.

The CSV export contains lap-relative elapsed time rather than a continuous
session timestamp.  This module therefore keeps every CSV row (including exact
duplicates), builds a monotonic session clock, pairs recordings by their
filename timestamps, and samples telemetry using each video's presentation
timestamps.

Video support is provided by PyAV and imported lazily so the data-processing
and track-projection helpers remain usable without opening a video.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from fractions import Fraction
from pathlib import Path
import re
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


EARTH_RADIUS_M = 6_371_000.0
DEFAULT_SAMPLE_RATE_HZ = 50.0

TELEMETRY_FILENAME = re.compile(
    r"^telemetry-v\d+-(?P<stamp>\d{4}-\d{2}-\d{2}-\d{2}_\d{2}_\d{2})\.csv$",
    re.IGNORECASE,
)
VIDEO_FILENAME = re.compile(
    r"^laps-(?P<stamp>\d{4}-\d{2}-\d{2}-\d{2}_\d{2}_\d{2})\.mp4$",
    re.IGNORECASE,
)
TIMESTAMP_FORMAT = "%Y-%m-%d-%H_%M_%S"

REQUIRED_OVERLAY_COLUMNS = (
    "Lap",
    "Elapsed Time (ms)",
    "Speed (MPH)",
    "Latitude (decimal)",
    "Longitude (decimal)",
    "Throttle Position (%)",
    "Brake Pressure (bar)",
)

LINEAR_CHANNELS = (
    "Speed (MPH)",
    "Latitude (decimal)",
    "Longitude (decimal)",
    "Lateral Acceleration (m/s^2)",
    "Longitudinal Acceleration (m/s^2)",
    "Throttle Position (%)",
    "Brake Pressure (bar)",
    "Steering Angle (deg)",
    "Steering Angle Rate (deg/s)",
    "Yaw Rate (rad/s)",
    "Power Level (KW)",
)

HELD_CHANNELS = (
    "Lap",
    "Elapsed Time (ms)",
    "State of Charge (%)",
    "Tire Pressure Front Left (bar)",
    "Tire Pressure Front Right (bar)",
    "Tire Pressure Rear Left (bar)",
    "Tire Pressure Rear Right (bar)",
    "Brake Temperature Front Left (% est.)",
    "Brake Temperature Front Right (% est.)",
    "Brake Temperature Rear Left (% est.)",
    "Brake Temperature Rear Right (% est.)",
    "Front Inverter Temp (%)",
    "Rear Inverter Temp (%)",
    "Battery Temp (%)",
    "Tire Slip Front Left (% est.)",
    "Tire Slip Front Right (% est.)",
    "Tire Slip Rear Left (% est.)",
    "Tire Slip Rear Right (% est.)",
)


def _import_av() -> Any:
    try:
        import av  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - exercised without video extras
        raise ModuleNotFoundError(
            "Video support requires PyAV. Install the repository requirements "
            "in the car_track Conda environment."
        ) from exc
    return av


def _parse_filename_timestamp(path: Path, pattern: re.Pattern[str]) -> datetime | None:
    match = pattern.match(path.name)
    if match is None:
        return None
    return datetime.strptime(match.group("stamp"), TIMESTAMP_FORMAT)


@dataclass(frozen=True)
class RecordingPair:
    """One telemetry CSV and its nearest timestamped Tesla video."""

    telemetry_path: Path
    video_path: Path
    telemetry_started_at: datetime
    video_started_at: datetime
    thumbnail_path: Path | None = None

    @property
    def session_id(self) -> str:
        return self.telemetry_started_at.strftime("%Y%m%d_%H%M%S")

    @property
    def filename_offset_s(self) -> float:
        """Seconds from telemetry start to video start.

        Positive values mean the CSV started before the video.
        """

        return (self.video_started_at - self.telemetry_started_at).total_seconds()


def discover_recording_pairs(
    data_dir: str | Path,
    *,
    max_start_delta_s: float = 5.0,
) -> list[RecordingPair]:
    """Pair telemetry and MP4 files using nearest unique start timestamps.

    Exact stem matching is insufficient for Tesla exports because the CSV and
    MP4 filenames in a recording may differ by several seconds.
    """

    directory = Path(data_dir).expanduser()
    if not directory.is_dir():
        raise FileNotFoundError(f"Recording directory not found: {directory.resolve()}")
    if max_start_delta_s < 0:
        raise ValueError("max_start_delta_s must be non-negative")

    telemetry_files = [
        (path, timestamp)
        for path in directory.glob("*.csv")
        if (timestamp := _parse_filename_timestamp(path, TELEMETRY_FILENAME)) is not None
    ]
    video_files = [
        (path, timestamp)
        for path in directory.glob("*.mp4")
        if (timestamp := _parse_filename_timestamp(path, VIDEO_FILENAME)) is not None
    ]
    if not telemetry_files:
        raise FileNotFoundError(f"No timestamped Tesla telemetry CSVs found in {directory.resolve()}")
    if not video_files:
        raise FileNotFoundError(f"No timestamped Tesla MP4s found in {directory.resolve()}")

    candidates: list[tuple[float, datetime, datetime, Path, Path]] = []
    for telemetry_path, telemetry_time in telemetry_files:
        for video_path, video_time in video_files:
            delta = abs((video_time - telemetry_time).total_seconds())
            if delta <= max_start_delta_s:
                candidates.append(
                    (delta, telemetry_time, video_time, telemetry_path, video_path)
                )

    used_telemetry: set[Path] = set()
    used_videos: set[Path] = set()
    pairs: list[RecordingPair] = []
    for _, telemetry_time, video_time, telemetry_path, video_path in sorted(candidates):
        if telemetry_path in used_telemetry or video_path in used_videos:
            continue
        thumbnail = video_path.with_name(f"{video_path.stem}-thumb.png")
        pairs.append(
            RecordingPair(
                telemetry_path=telemetry_path,
                video_path=video_path,
                telemetry_started_at=telemetry_time,
                video_started_at=video_time,
                thumbnail_path=thumbnail if thumbnail.is_file() else None,
            )
        )
        used_telemetry.add(telemetry_path)
        used_videos.add(video_path)

    if not pairs:
        raise FileNotFoundError(
            f"No CSV/MP4 pairs were within {max_start_delta_s:g} seconds in {directory.resolve()}"
        )
    return sorted(pairs, key=lambda pair: pair.telemetry_started_at)


def select_recording(
    recordings: Iterable[RecordingPair],
    session: str | None = None,
) -> RecordingPair:
    """Select a recording by session ID, CSV stem, or CSV filename."""

    items = list(recordings)
    if not items:
        raise ValueError("No recordings are available")
    if session is None:
        return items[-1]
    requested = Path(session).name.casefold()
    for item in items:
        candidates = {
            item.session_id.casefold(),
            item.telemetry_path.stem.casefold(),
            item.telemetry_path.name.casefold(),
        }
        if requested in candidates:
            return item
    available = ", ".join(item.session_id for item in items)
    raise KeyError(f"Unknown recording {session!r}. Available session IDs: {available}")


@dataclass(frozen=True)
class VideoInfo:
    path: Path
    width: int
    height: int
    duration_s: float
    average_rate: float
    frames: int
    codec: str
    time_base: float
    has_audio: bool


def probe_video(video_path: str | Path) -> VideoInfo:
    """Read video metadata through PyAV without decoding every frame."""

    av = _import_av()
    path = Path(video_path)
    if not path.is_file():
        raise FileNotFoundError(f"Video not found: {path.resolve()}")

    with av.open(str(path)) as container:
        if not container.streams.video:
            raise ValueError(f"No video stream found in {path.name}")
        stream = container.streams.video[0]
        if stream.duration is not None and stream.time_base is not None:
            duration_s = float(stream.duration * stream.time_base)
        elif container.duration is not None:
            duration_s = float(container.duration / av.time_base)
        elif stream.frames and stream.average_rate:
            duration_s = float(stream.frames / stream.average_rate)
        else:
            raise ValueError(f"Could not determine video duration for {path.name}")

        average_rate = float(stream.average_rate) if stream.average_rate else float("nan")
        time_base = float(stream.time_base) if stream.time_base else float("nan")
        codec_context = stream.codec_context
        return VideoInfo(
            path=path,
            width=int(codec_context.width),
            height=int(codec_context.height),
            duration_s=duration_s,
            average_rate=average_rate,
            frames=int(stream.frames or 0),
            codec=str(codec_context.name or stream.codec.name),
            time_base=time_base,
            has_audio=bool(container.streams.audio),
        )


def load_overlay_telemetry(csv_path: str | Path) -> pd.DataFrame:
    """Load Tesla telemetry in original row order and retain duplicate rows."""

    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"Telemetry CSV not found: {path.resolve()}")
    telemetry = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    if telemetry.empty:
        raise ValueError(f"Telemetry CSV is empty: {path.name}")
    missing = sorted(set(REQUIRED_OVERLAY_COLUMNS).difference(telemetry.columns))
    if missing:
        raise ValueError(f"{path.name} is missing required columns: {missing}")

    for column in telemetry.columns:
        telemetry[column] = pd.to_numeric(telemetry[column], errors="coerce")
    telemetry.insert(0, "Sample Index", np.arange(len(telemetry), dtype=np.int64))
    telemetry["Source"] = path.name
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


def estimate_lap_sample_rate_hz(telemetry: pd.DataFrame) -> float | None:
    """Estimate stream rate from positive laps without using median deltas."""

    if not {"Lap", "Elapsed Time (ms)"}.issubset(telemetry.columns):
        return None
    intervals = 0
    duration_s = 0.0
    timed = telemetry.loc[
        telemetry["Lap"].gt(0) & telemetry["Elapsed Time (ms)"].gt(0),
        ["Lap", "Elapsed Time (ms)"],
    ]
    for _, group in timed.groupby("Lap", sort=False):
        elapsed = group["Elapsed Time (ms)"].to_numpy(dtype=float)
        elapsed = elapsed[np.isfinite(elapsed)]
        if len(elapsed) < 100:
            continue
        span_s = (float(np.max(elapsed)) - float(np.min(elapsed))) / 1_000.0
        if span_s <= 0:
            continue
        intervals += len(elapsed) - 1
        duration_s += span_s
    if not intervals or duration_s <= 0:
        return None
    rate = intervals / duration_s
    return float(rate) if 20.0 <= rate <= 100.0 else None


@dataclass(frozen=True)
class SyncInfo:
    telemetry_rows: int
    sample_rate_hz: float
    video_start_offset_s: float
    telemetry_span_s: float
    rate_source: str
    lap_rate_hz: float | None = None

    def session_time_for_video(self, video_time_s: float, sync_adjust_s: float = 0.0) -> float:
        return float(video_time_s + self.video_start_offset_s + sync_adjust_s)

    def video_time_for_session(self, session_time_s: float, sync_adjust_s: float = 0.0) -> float:
        return float(session_time_s - self.video_start_offset_s - sync_adjust_s)


def build_sync_info(
    telemetry: pd.DataFrame,
    video_info: VideoInfo,
    *,
    video_start_offset_s: float,
    default_rate_hz: float = DEFAULT_SAMPLE_RATE_HZ,
) -> SyncInfo:
    """Fit a continuous telemetry clock to the paired recording duration."""

    rows = len(telemetry)
    if rows < 2:
        raise ValueError("At least two telemetry rows are required for synchronization")
    if default_rate_hz <= 0:
        raise ValueError("default_rate_hz must be positive")

    lap_rate = estimate_lap_sample_rate_hz(telemetry)
    end_aligned_span = video_info.duration_s + float(video_start_offset_s)
    end_aligned_rate = (
        (rows - 1) / end_aligned_span if end_aligned_span > 0 else float("nan")
    )

    plausible = 30.0 <= end_aligned_rate <= 80.0
    consistent_with_laps = (
        lap_rate is None or abs(end_aligned_rate - lap_rate) / lap_rate <= 0.12
    )
    if plausible and consistent_with_laps:
        sample_rate = float(end_aligned_rate)
        source = "video-end alignment"
    elif lap_rate is not None:
        sample_rate = float(lap_rate)
        source = "timed-lap elapsed clock"
    else:
        sample_rate = float(default_rate_hz)
        source = "configured default"

    return SyncInfo(
        telemetry_rows=rows,
        sample_rate_hz=sample_rate,
        video_start_offset_s=float(video_start_offset_s),
        telemetry_span_s=(rows - 1) / sample_rate,
        rate_source=source,
        lap_rate_hz=lap_rate,
    )


def attach_session_clock(telemetry: pd.DataFrame, sync: SyncInfo) -> pd.DataFrame:
    """Return a copy with monotonic session and initial video timestamps."""

    if len(telemetry) != sync.telemetry_rows:
        raise ValueError("Telemetry row count does not match SyncInfo")
    result = telemetry.copy()
    result["Session Time (s)"] = np.arange(len(result), dtype=float) / sync.sample_rate_hz
    result["Video Time (s)"] = result["Session Time (s)"] - sync.video_start_offset_s
    return result


@dataclass(frozen=True)
class TrackProjection:
    progress: float
    x_m: float
    y_m: float
    distance_from_reference_m: float


@dataclass(frozen=True)
class TrackReference:
    """A smoothed, distance-resampled reference lap in local metres."""

    origin_latitude: float
    origin_longitude: float
    x_m: np.ndarray
    y_m: np.ndarray
    cumulative_m: np.ndarray
    total_distance_m: float
    lap: int
    source: str

    def to_local(self, latitude: float, longitude: float) -> tuple[float, float]:
        metres_per_degree = np.pi * EARTH_RADIUS_M / 180.0
        x = (
            (longitude - self.origin_longitude)
            * metres_per_degree
            * np.cos(np.radians(self.origin_latitude))
        )
        y = (latitude - self.origin_latitude) * metres_per_degree
        return float(x), float(y)

    def project(self, latitude: float, longitude: float) -> TrackProjection:
        if not np.isfinite(latitude) or not np.isfinite(longitude):
            return TrackProjection(*(float("nan"),) * 4)
        point = np.asarray(self.to_local(latitude, longitude), dtype=float)
        starts = np.column_stack([self.x_m[:-1], self.y_m[:-1]])
        ends = np.column_stack([self.x_m[1:], self.y_m[1:]])
        vectors = ends - starts
        length_squared = np.einsum("ij,ij->i", vectors, vectors)
        safe_length_squared = np.where(length_squared > 0, length_squared, 1.0)
        fractions = np.einsum("ij,ij->i", point - starts, vectors) / safe_length_squared
        fractions = np.clip(fractions, 0.0, 1.0)
        projected = starts + vectors * fractions[:, None]
        errors = projected - point
        index = int(np.argmin(np.einsum("ij,ij->i", errors, errors)))
        segment_length = float(np.sqrt(length_squared[index]))
        distance = float(np.sqrt(np.dot(errors[index], errors[index])))
        along = float(self.cumulative_m[index] + fractions[index] * segment_length)
        progress = along / self.total_distance_m if self.total_distance_m > 0 else 0.0
        return TrackProjection(
            progress=float(progress % 1.0),
            x_m=float(projected[index, 0]),
            y_m=float(projected[index, 1]),
            distance_from_reference_m=distance,
        )


def _local_coordinates(
    latitude: np.ndarray,
    longitude: np.ndarray,
    origin_latitude: float,
    origin_longitude: float,
) -> tuple[np.ndarray, np.ndarray]:
    metres_per_degree = np.pi * EARTH_RADIUS_M / 180.0
    x = (
        (longitude - origin_longitude)
        * metres_per_degree
        * np.cos(np.radians(origin_latitude))
    )
    y = (latitude - origin_latitude) * metres_per_degree
    return x, y


def _polyline_distance(latitude: np.ndarray, longitude: np.ndarray) -> float:
    if len(latitude) < 2:
        return 0.0
    origin_latitude = float(np.nanmean(latitude))
    origin_longitude = float(np.nanmean(longitude))
    x, y = _local_coordinates(
        latitude, longitude, origin_latitude, origin_longitude
    )
    return float(np.sum(np.hypot(np.diff(x), np.diff(y))))


def choose_reference_lap(telemetry: pd.DataFrame) -> int:
    """Choose the quickest lap whose GPS distance is near the session median."""

    candidates: list[tuple[int, float, float]] = []
    for lap, group in telemetry.loc[telemetry["Lap"].gt(0)].groupby("Lap", sort=True):
        clean = group.dropna(subset=["Latitude (decimal)", "Longitude (decimal)"])
        if len(clean) < 100:
            continue
        latitude = clean["Latitude (decimal)"].to_numpy(dtype=float)
        longitude = clean["Longitude (decimal)"].to_numpy(dtype=float)
        distance = _polyline_distance(latitude, longitude)
        duration = float(clean["Elapsed Time (ms)"].max())
        if distance > 0 and duration > 0:
            candidates.append((int(lap), distance, duration))
    if not candidates:
        raise ValueError(
            "No complete positive lap is available for a reference track. "
            "Load a timed recording as the reference source."
        )

    median_distance = float(np.median([candidate[1] for candidate in candidates]))
    clean_candidates = [
        candidate
        for candidate in candidates
        if abs(candidate[1] - median_distance) / median_distance <= 0.12
    ]
    return min(clean_candidates or candidates, key=lambda candidate: candidate[2])[0]


def build_track_reference(
    telemetry: pd.DataFrame,
    *,
    lap: int | None = None,
    points: int = 600,
    smoothing_window: int = 7,
) -> TrackReference:
    """Build a fixed reference centreline from one clean timed lap."""

    if points < 50:
        raise ValueError("points must be at least 50")
    selected_lap = choose_reference_lap(telemetry) if lap is None else int(lap)
    clean = (
        telemetry.loc[telemetry["Lap"].eq(selected_lap)]
        .dropna(subset=["Latitude (decimal)", "Longitude (decimal)"])
        .sort_values("Sample Index", kind="stable")
    )
    if len(clean) < 100:
        raise ValueError(f"Lap {selected_lap} does not have enough GPS samples")

    latitude = clean["Latitude (decimal)"].to_numpy(dtype=float)
    longitude = clean["Longitude (decimal)"].to_numpy(dtype=float)
    origin_latitude = float(np.mean(latitude))
    origin_longitude = float(np.mean(longitude))
    x, y = _local_coordinates(
        latitude, longitude, origin_latitude, origin_longitude
    )

    # The GPS channels refresh more slowly than the CSV stream, so first remove
    # repeated points and then smooth the remaining updates.
    steps = np.r_[True, np.hypot(np.diff(x), np.diff(y)) > 0.25]
    x = x[steps]
    y = y[steps]
    if len(x) < 20:
        raise ValueError(f"Lap {selected_lap} does not have enough distinct GPS points")

    window = max(1, int(smoothing_window))
    if window > 1:
        x = pd.Series(x).rolling(window, center=True, min_periods=1).median().to_numpy()
        y = pd.Series(y).rolling(window, center=True, min_periods=1).median().to_numpy()

    initial_distance = float(np.sum(np.hypot(np.diff(x), np.diff(y))))
    closure_distance = float(np.hypot(x[-1] - x[0], y[-1] - y[0]))
    if closure_distance <= max(75.0, initial_distance * 0.03):
        x = np.r_[x, x[0]]
        y = np.r_[y, y[0]]

    segment_lengths = np.hypot(np.diff(x), np.diff(y))
    moving = np.r_[True, segment_lengths > 0.05]
    x = x[moving]
    y = y[moving]
    cumulative = np.r_[0.0, np.cumsum(np.hypot(np.diff(x), np.diff(y)))]
    total_distance = float(cumulative[-1])
    if total_distance <= 0:
        raise ValueError(f"Lap {selected_lap} has no usable GPS distance")

    resampled_distance = np.linspace(0.0, total_distance, points)
    resampled_x = np.interp(resampled_distance, cumulative, x)
    resampled_y = np.interp(resampled_distance, cumulative, y)
    return TrackReference(
        origin_latitude=origin_latitude,
        origin_longitude=origin_longitude,
        x_m=resampled_x,
        y_m=resampled_y,
        cumulative_m=resampled_distance,
        total_distance_m=total_distance,
        lap=selected_lap,
        source=str(clean["Source"].iloc[0]) if "Source" in clean else "telemetry",
    )


@dataclass(frozen=True)
class TelemetrySample:
    video_time_s: float
    session_time_s: float
    lap: int
    lap_elapsed_ms: float
    speed_mph: float
    speed_kmh: float
    latitude: float
    longitude: float
    lateral_g: float
    longitudinal_g: float
    throttle_pct: float
    brake_bar: float
    steering_deg: float
    yaw_rate: float
    power_kw: float
    state_of_charge_pct: float
    battery_thermal_pct: float
    front_inverter_thermal_pct: float
    rear_inverter_thermal_pct: float
    tire_pressures_bar: tuple[float, float, float, float]
    brake_temperature_est_pct: tuple[float, float, float, float]
    tire_slip_est_pct: tuple[float, float, float, float]


class TelemetrySampler:
    """Efficient interpolation of prepared telemetry at video timestamps."""

    def __init__(self, telemetry: pd.DataFrame, sync: SyncInfo) -> None:
        if len(telemetry) != sync.telemetry_rows:
            raise ValueError("Telemetry row count does not match SyncInfo")
        self.telemetry = telemetry
        self.sync = sync
        self.times = np.arange(len(telemetry), dtype=float) / sync.sample_rate_hz
        self._values: dict[str, np.ndarray] = {}
        for column in set(LINEAR_CHANNELS).union(HELD_CHANNELS):
            if column in telemetry:
                self._values[column] = telemetry[column].to_numpy(dtype=float)

        self._interpolation: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for column in LINEAR_CHANNELS:
            if column not in self._values:
                continue
            values = self._values[column]
            finite = np.isfinite(values)
            if np.any(finite):
                self._interpolation[column] = (self.times[finite], values[finite])

    def _linear(self, column: str, time_s: float) -> float:
        values = self._interpolation.get(column)
        if values is None:
            return float("nan")
        times, channel = values
        return float(np.interp(time_s, times, channel, left=channel[0], right=channel[-1]))

    def _held(self, column: str, index: int) -> float:
        values = self._values.get(column)
        if values is None:
            return float("nan")
        value = float(values[index])
        if np.isfinite(value):
            return value
        finite_indices = np.flatnonzero(np.isfinite(values[: index + 1]))
        return float(values[finite_indices[-1]]) if len(finite_indices) else float("nan")

    @staticmethod
    def _thermal_percent(value: float) -> float:
        return float(value * 100.0) if np.isfinite(value) else float("nan")

    @staticmethod
    def _valid_pressure(value: float) -> float:
        return float(value) if np.isfinite(value) and value > 0 else float("nan")

    def sample(self, video_time_s: float, *, sync_adjust_s: float = 0.0) -> TelemetrySample:
        session_time = self.sync.session_time_for_video(video_time_s, sync_adjust_s)
        index = int(np.searchsorted(self.times, session_time, side="right") - 1)
        index = int(np.clip(index, 0, len(self.times) - 1))

        speed_mph = self._linear("Speed (MPH)", session_time)
        lateral_ms2 = self._linear("Lateral Acceleration (m/s^2)", session_time)
        longitudinal_ms2 = self._linear("Longitudinal Acceleration (m/s^2)", session_time)
        pressures = tuple(
            self._valid_pressure(self._held(column, index))
            for column in (
                "Tire Pressure Front Left (bar)",
                "Tire Pressure Front Right (bar)",
                "Tire Pressure Rear Left (bar)",
                "Tire Pressure Rear Right (bar)",
            )
        )
        brake_temperatures = tuple(
            self._thermal_percent(self._held(column, index))
            for column in (
                "Brake Temperature Front Left (% est.)",
                "Brake Temperature Front Right (% est.)",
                "Brake Temperature Rear Left (% est.)",
                "Brake Temperature Rear Right (% est.)",
            )
        )
        slips = tuple(
            self._thermal_percent(self._held(column, index))
            for column in (
                "Tire Slip Front Left (% est.)",
                "Tire Slip Front Right (% est.)",
                "Tire Slip Rear Left (% est.)",
                "Tire Slip Rear Right (% est.)",
            )
        )
        return TelemetrySample(
            video_time_s=float(video_time_s),
            session_time_s=session_time,
            lap=max(0, int(round(self._held("Lap", index)))),
            lap_elapsed_ms=max(0.0, self._held("Elapsed Time (ms)", index)),
            speed_mph=speed_mph,
            speed_kmh=speed_mph * 1.609344,
            latitude=self._linear("Latitude (decimal)", session_time),
            longitude=self._linear("Longitude (decimal)", session_time),
            lateral_g=lateral_ms2 / 9.80665,
            longitudinal_g=longitudinal_ms2 / 9.80665,
            throttle_pct=float(np.clip(self._linear("Throttle Position (%)", session_time), 0, 100)),
            brake_bar=max(0.0, self._linear("Brake Pressure (bar)", session_time)),
            steering_deg=self._linear("Steering Angle (deg)", session_time),
            yaw_rate=self._linear("Yaw Rate (rad/s)", session_time),
            power_kw=self._linear("Power Level (KW)", session_time),
            state_of_charge_pct=self._held("State of Charge (%)", index),
            battery_thermal_pct=self._thermal_percent(self._held("Battery Temp (%)", index)),
            front_inverter_thermal_pct=self._thermal_percent(
                self._held("Front Inverter Temp (%)", index)
            ),
            rear_inverter_thermal_pct=self._thermal_percent(
                self._held("Rear Inverter Temp (%)", index)
            ),
            tire_pressures_bar=pressures,  # type: ignore[arg-type]
            brake_temperature_est_pct=brake_temperatures,  # type: ignore[arg-type]
            tire_slip_est_pct=slips,  # type: ignore[arg-type]
        )


def format_overlay_time(milliseconds: float) -> str:
    if not np.isfinite(milliseconds) or milliseconds <= 0:
        return "--:--.---"
    total_ms = int(round(milliseconds))
    minutes, remainder = divmod(total_ms, 60_000)
    seconds, millis = divmod(remainder, 1_000)
    return f"{minutes}:{seconds:02d}.{millis:03d}"


@dataclass(frozen=True)
class OverlayStyle:
    speed_unit: str = "km/h"
    panel_alpha: int = 205
    show_tire_pressures: bool = False
    show_estimated_temperatures: bool = False
    max_brake_bar: float = 80.0
    max_g: float = 1.5
    accent: tuple[int, int, int] = (250, 204, 21)

    def __post_init__(self) -> None:
        if self.speed_unit not in {"km/h", "MPH"}:
            raise ValueError("speed_unit must be 'km/h' or 'MPH'")


class TelemetryOverlayRenderer:
    """Render a scalable, high-contrast telemetry HUD onto Pillow images."""

    def __init__(self, track: TrackReference, style: OverlayStyle | None = None) -> None:
        self.track = track
        self.style = style or OverlayStyle()
        self._font_cache: dict[tuple[int, bool], ImageFont.ImageFont] = {}

    def _font(self, size: int, *, bold: bool = False) -> ImageFont.ImageFont:
        key = (size, bold)
        if key in self._font_cache:
            return self._font_cache[key]
        names = (
            ["C:/Windows/Fonts/segoeuib.ttf", "C:/Windows/Fonts/arialbd.ttf", "DejaVuSans-Bold.ttf"]
            if bold
            else ["C:/Windows/Fonts/segoeui.ttf", "C:/Windows/Fonts/arial.ttf", "DejaVuSans.ttf"]
        )
        for name in names:
            try:
                font = ImageFont.truetype(name, size=size)
                self._font_cache[key] = font
                return font
            except OSError:
                continue
        font = ImageFont.load_default()
        self._font_cache[key] = font
        return font

    @staticmethod
    def _thermal_color(value: float) -> tuple[int, int, int]:
        if not np.isfinite(value):
            return (148, 163, 184)
        if value >= 100:
            return (239, 68, 68)
        if value >= 85:
            return (249, 115, 22)
        if value >= 75:
            return (250, 204, 21)
        return (34, 197, 94)

    @staticmethod
    def _fmt(value: float, pattern: str, missing: str = "--") -> str:
        return format(value, pattern) if np.isfinite(value) else missing

    def _panel(
        self,
        draw: ImageDraw.ImageDraw,
        box: tuple[int, int, int, int],
        radius: int,
    ) -> None:
        draw.rounded_rectangle(
            box,
            radius=radius,
            fill=(7, 12, 20, self.style.panel_alpha),
            outline=(148, 163, 184, 135),
            width=max(1, radius // 6),
        )

    def _bar(
        self,
        draw: ImageDraw.ImageDraw,
        box: tuple[int, int, int, int],
        fraction: float,
        color: tuple[int, int, int],
    ) -> None:
        x0, y0, x1, y1 = box
        fraction = float(np.clip(fraction, 0.0, 1.0)) if np.isfinite(fraction) else 0.0
        radius = max(2, (y1 - y0) // 2)
        draw.rounded_rectangle(box, radius=radius, fill=(51, 65, 85, 210))
        if fraction > 0:
            fill_x = max(x0 + 1, int(round(x0 + (x1 - x0) * fraction)))
            draw.rounded_rectangle(
                (x0, y0, fill_x, y1), radius=radius, fill=(*color, 240)
            )

    def _draw_track(
        self,
        draw: ImageDraw.ImageDraw,
        box: tuple[int, int, int, int],
        sample: TelemetrySample,
        font_small: ImageFont.ImageFont,
        line_width: int,
    ) -> None:
        x0, y0, x1, y1 = box
        padding = max(8, int((x1 - x0) * 0.05))
        label_height = max(20, int((y1 - y0) * 0.15))
        map_box = (x0 + padding, y0 + padding + label_height, x1 - padding, y1 - padding)
        min_x, max_x = float(np.min(self.track.x_m)), float(np.max(self.track.x_m))
        min_y, max_y = float(np.min(self.track.y_m)), float(np.max(self.track.y_m))
        span_x = max(max_x - min_x, 1.0)
        span_y = max(max_y - min_y, 1.0)
        map_width = map_box[2] - map_box[0]
        map_height = map_box[3] - map_box[1]
        scale = min(map_width / span_x, map_height / span_y)
        offset_x = map_box[0] + (map_width - span_x * scale) / 2
        offset_y = map_box[1] + (map_height - span_y * scale) / 2

        def screen(x: float, y: float) -> tuple[int, int]:
            return (
                int(round(offset_x + (x - min_x) * scale)),
                int(round(offset_y + (max_y - y) * scale)),
            )

        points = [screen(float(x), float(y)) for x, y in zip(self.track.x_m, self.track.y_m)]
        if len(points) > 1:
            draw.line(points, fill=(226, 232, 240, 235), width=line_width, joint="curve")
        start = screen(float(self.track.x_m[0]), float(self.track.y_m[0]))
        marker_radius = max(3, line_width)
        draw.ellipse(
            (start[0] - marker_radius, start[1] - marker_radius, start[0] + marker_radius, start[1] + marker_radius),
            fill=(255, 255, 255, 255),
            outline=(15, 23, 42, 255),
        )

        projection = self.track.project(sample.latitude, sample.longitude)
        if np.isfinite(projection.x_m):
            current = screen(projection.x_m, projection.y_m)
            radius = max(6, line_width * 2)
            draw.ellipse(
                (current[0] - radius, current[1] - radius, current[0] + radius, current[1] + radius),
                fill=(*self.style.accent, 255),
                outline=(15, 23, 42, 255),
                width=max(2, line_width // 2),
            )
            progress_text = f"TRACK  {projection.progress * 100:5.1f}%"
        else:
            progress_text = "TRACK  --.-%"
        draw.text((x0 + padding, y0 + padding), progress_text, font=font_small, fill=(241, 245, 249, 255))

    def render(self, image: Image.Image, sample: TelemetrySample) -> Image.Image:
        base = image.convert("RGBA")
        width, height = base.size
        scale = max(0.45, min(width / 1920.0, height / 1080.0))
        margin = max(12, int(round(min(width, height) * 0.025)))
        gap = max(8, int(round(14 * scale)))
        radius = max(8, int(round(14 * scale)))
        overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay, "RGBA")

        font_tiny = self._font(max(10, int(round(17 * scale))))
        font_small = self._font(max(12, int(round(23 * scale))), bold=True)
        font_medium = self._font(max(14, int(round(32 * scale))), bold=True)
        font_speed = self._font(max(30, int(round(78 * scale))), bold=True)

        speed_box = (
            margin,
            margin,
            margin + int(width * 0.235),
            margin + int(height * 0.185),
        )
        track_box = (
            width - margin - int(width * 0.27),
            margin,
            width - margin,
            margin + int(height * 0.31),
        )
        bottom_y = height - margin
        inputs_box = (
            margin,
            bottom_y - int(height * 0.16),
            margin + int(width * 0.31),
            bottom_y,
        )
        dynamics_box = (
            inputs_box[2] + gap,
            bottom_y - int(height * 0.16),
            inputs_box[2] + gap + int(width * 0.245),
            bottom_y,
        )
        thermal_extra_rows = int(self.style.show_tire_pressures) + int(
            self.style.show_estimated_temperatures
        )
        thermal_box = (
            width - margin - int(width * 0.295),
            bottom_y - int(height * (0.18 + 0.035 * thermal_extra_rows)),
            width - margin,
            bottom_y,
        )
        for box in (speed_box, track_box, inputs_box, dynamics_box, thermal_box):
            self._panel(draw, box, radius)

        # Speed and lap clock.
        pad = max(8, int(round(15 * scale)))
        speed = sample.speed_kmh if self.style.speed_unit == "km/h" else sample.speed_mph
        speed_text = self._fmt(speed, ".0f")
        draw.text(
            (speed_box[0] + pad, speed_box[1] + int(2 * scale)),
            speed_text,
            font=font_speed,
            fill=(248, 250, 252, 255),
        )
        speed_width = draw.textbbox((0, 0), speed_text, font=font_speed)[2]
        draw.text(
            (speed_box[0] + pad + speed_width + int(8 * scale), speed_box[1] + int(47 * scale)),
            self.style.speed_unit,
            font=font_small,
            fill=(203, 213, 225, 255),
        )
        lap_label = f"LAP {sample.lap}" if sample.lap > 0 else "SESSION"
        lap_time = format_overlay_time(sample.lap_elapsed_ms)
        draw.text(
            (speed_box[0] + pad, speed_box[3] - pad - int(25 * scale)),
            f"{lap_label}   {lap_time}",
            font=font_small,
            fill=(*self.style.accent, 255),
        )

        self._draw_track(
            draw,
            track_box,
            sample,
            font_small,
            max(3, int(round(5 * scale))),
        )

        # Driver inputs.
        ix0, iy0, ix1, iy1 = inputs_box
        label_x = ix0 + pad
        bar_x0 = ix0 + int((ix1 - ix0) * 0.31)
        bar_x1 = ix1 - pad
        row_height = (iy1 - iy0 - 2 * pad) // 2
        throttle_y = iy0 + pad
        brake_y = throttle_y + row_height
        draw.text((label_x, throttle_y), "THROTTLE", font=font_tiny, fill=(226, 232, 240, 255))
        draw.text((label_x, brake_y), "BRAKE", font=font_tiny, fill=(226, 232, 240, 255))
        bar_height = max(8, int(round(16 * scale)))
        self._bar(
            draw,
            (bar_x0, throttle_y + int(4 * scale), bar_x1, throttle_y + int(4 * scale) + bar_height),
            sample.throttle_pct / 100.0,
            (34, 197, 94),
        )
        self._bar(
            draw,
            (bar_x0, brake_y + int(4 * scale), bar_x1, brake_y + int(4 * scale) + bar_height),
            sample.brake_bar / self.style.max_brake_bar,
            (239, 68, 68),
        )
        draw.text(
            (bar_x1, throttle_y),
            f" {sample.throttle_pct:3.0f}%",
            font=font_tiny,
            fill=(241, 245, 249, 255),
            anchor="ra",
        )
        draw.text(
            (bar_x1, brake_y),
            f" {sample.brake_bar:4.1f} bar",
            font=font_tiny,
            fill=(241, 245, 249, 255),
            anchor="ra",
        )

        # Power and g meter.
        dx0, dy0, dx1, dy1 = dynamics_box
        draw.text((dx0 + pad, dy0 + pad), "POWER", font=font_tiny, fill=(203, 213, 225, 255))
        power_text = self._fmt(sample.power_kw, "+.0f") + " kW"
        power_color = (249, 115, 22) if sample.power_kw >= 0 else (56, 189, 248)
        draw.text(
            (dx0 + pad, dy0 + pad + int(24 * scale)),
            power_text,
            font=font_medium,
            fill=(*power_color, 255),
        )
        center_x = dx1 - pad - int((dy1 - dy0) * 0.35)
        center_y = (dy0 + dy1) // 2
        g_radius = max(18, int((dy1 - dy0) * 0.29))
        draw.ellipse(
            (center_x - g_radius, center_y - g_radius, center_x + g_radius, center_y + g_radius),
            outline=(148, 163, 184, 220),
            width=max(1, int(round(2 * scale))),
        )
        draw.line((center_x - g_radius, center_y, center_x + g_radius, center_y), fill=(71, 85, 105, 220))
        draw.line((center_x, center_y - g_radius, center_x, center_y + g_radius), fill=(71, 85, 105, 220))
        dot_x = center_x + int(np.clip(sample.lateral_g / self.style.max_g, -1, 1) * g_radius)
        dot_y = center_y - int(np.clip(sample.longitudinal_g / self.style.max_g, -1, 1) * g_radius)
        dot_radius = max(4, int(round(6 * scale)))
        draw.ellipse(
            (dot_x - dot_radius, dot_y - dot_radius, dot_x + dot_radius, dot_y + dot_radius),
            fill=(*self.style.accent, 255),
        )

        # Thermal block. These are normalized Tesla thermal indicators, not °C.
        tx0, ty0, tx1, ty1 = thermal_box
        draw.text((tx0 + pad, ty0 + pad), "THERMAL LOAD", font=font_tiny, fill=(203, 213, 225, 255))
        thermal_rows = (
            ("BATTERY", sample.battery_thermal_pct),
            ("FRONT INV", sample.front_inverter_thermal_pct),
            ("REAR INV", sample.rear_inverter_thermal_pct),
        )
        thermal_row_height = max(18, int(round(27 * scale)))
        thermal_start = ty0 + pad + int(round(25 * scale))
        for row, (label, value) in enumerate(thermal_rows):
            y = thermal_start + row * thermal_row_height
            draw.text((tx0 + pad, y), label, font=font_tiny, fill=(226, 232, 240, 255))
            draw.text(
                (tx1 - pad, y),
                self._fmt(value, ".0f") + ("%" if np.isfinite(value) else ""),
                font=font_small,
                fill=(*self._thermal_color(value), 255),
                anchor="ra",
            )
        soc_text = self._fmt(sample.state_of_charge_pct, ".1f")
        detail_y = thermal_start + len(thermal_rows) * thermal_row_height
        draw.text(
            (tx0 + pad, detail_y),
            f"SOC  {soc_text}%",
            font=font_small,
            fill=(56, 189, 248, 255),
        )

        if self.style.show_tire_pressures:
            pressure_text = "  ".join(
                self._fmt(value, ".2f") for value in sample.tire_pressures_bar
            )
            detail_y += thermal_row_height
            draw.text(
                (tx0 + pad, detail_y),
                f"TYRES  {pressure_text} bar",
                font=font_tiny,
                fill=(226, 232, 240, 255),
            )

        if self.style.show_estimated_temperatures:
            detail_y += thermal_row_height
            brake_text = "  ".join(
                self._fmt(value, ".0f") for value in sample.brake_temperature_est_pct
            )
            draw.text(
                (tx0 + pad, detail_y),
                f"BRAKE EST  {brake_text}%",
                font=font_tiny,
                fill=(226, 232, 240, 255),
            )

        return Image.alpha_composite(base, overlay).convert("RGB")


def _frame_time_seconds(frame: Any, stream: Any, fallback_index: int) -> float:
    if frame.pts is not None and frame.time_base is not None:
        return float(frame.pts * frame.time_base)
    if frame.time is not None:
        return float(frame.time)
    rate = float(stream.average_rate) if stream.average_rate else 30.0
    return fallback_index / rate


def _target_dimensions(width: int, height: int, output_width: int | None) -> tuple[int, int]:
    if output_width is None or output_width >= width:
        target_width, target_height = width, height
    else:
        target_width = int(output_width)
        target_height = int(round(height * target_width / width))
    target_width += target_width % 2
    target_height += target_height % 2
    return target_width, target_height


@dataclass(frozen=True)
class PreviewResult:
    image: Image.Image
    requested_time_s: float
    actual_time_s: float
    sample: TelemetrySample


def render_overlay_preview(
    video_path: str | Path,
    sampler: TelemetrySampler,
    renderer: TelemetryOverlayRenderer,
    *,
    video_time_s: float,
    sync_adjust_s: float = 0.0,
    output_width: int | None = 1440,
) -> PreviewResult:
    """Decode and overlay the first frame at or after ``video_time_s``."""

    av = _import_av()
    path = Path(video_path)
    with av.open(str(path)) as container:
        if not container.streams.video:
            raise ValueError(f"No video stream found in {path.name}")
        stream = container.streams.video[0]
        if stream.time_base:
            seek_time = max(0.0, float(video_time_s) - 1.0)
            container.seek(
                int(seek_time / float(stream.time_base)),
                stream=stream,
                backward=True,
                any_frame=False,
            )
        for index, frame in enumerate(container.decode(stream)):
            actual_time = _frame_time_seconds(frame, stream, index)
            if actual_time + 1e-9 < video_time_s:
                continue
            image = frame.to_image().convert("RGB")
            target_size = _target_dimensions(image.width, image.height, output_width)
            if image.size != target_size:
                image = image.resize(target_size, Image.Resampling.LANCZOS)
            sample = sampler.sample(actual_time, sync_adjust_s=sync_adjust_s)
            return PreviewResult(
                image=renderer.render(image, sample),
                requested_time_s=float(video_time_s),
                actual_time_s=actual_time,
                sample=sample,
            )
    raise ValueError(f"No frame found at {video_time_s:.3f} s in {path.name}")


ProgressCallback = Callable[[int, float, float | None], None]


@dataclass(frozen=True)
class RenderResult:
    output_path: Path
    frames: int
    start_time_s: float
    end_time_s: float
    width: int
    height: int

    @property
    def duration_s(self) -> float:
        return max(0.0, self.end_time_s - self.start_time_s)


def render_overlay_video(
    video_path: str | Path,
    sampler: TelemetrySampler,
    renderer: TelemetryOverlayRenderer,
    output_path: str | Path,
    *,
    start_s: float = 0.0,
    duration_s: float | None = None,
    sync_adjust_s: float = 0.0,
    output_width: int | None = None,
    codec: str = "libx264",
    crf: int = 20,
    preset: str = "medium",
    progress: ProgressCallback | None = None,
) -> RenderResult:
    """Stream an overlaid MP4 while preserving source frame timestamps."""

    av = _import_av()
    source_path = Path(video_path)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if start_s < 0:
        raise ValueError("start_s must be non-negative")
    if duration_s is not None and duration_s <= 0:
        raise ValueError("duration_s must be positive when provided")

    input_container = av.open(str(source_path))
    output_container: Any | None = None
    frames_written = 0
    first_time: float | None = None
    last_time: float | None = None
    try:
        if not input_container.streams.video:
            raise ValueError(f"No video stream found in {source_path.name}")
        input_stream = input_container.streams.video[0]
        source_width = int(input_stream.codec_context.width)
        source_height = int(input_stream.codec_context.height)
        target_width, target_height = _target_dimensions(
            source_width, source_height, output_width
        )
        input_rate = input_stream.average_rate or Fraction(36, 1)
        input_time_base = input_stream.time_base or Fraction(1, 10_000)

        output_container = av.open(
            str(destination), "w", options={"movflags": "+faststart"}
        )
        output_stream = output_container.add_stream(codec, rate=input_rate)
        output_stream.width = target_width
        output_stream.height = target_height
        output_stream.pix_fmt = "yuv420p"
        output_stream.time_base = input_time_base
        output_stream.options = {"crf": str(int(crf)), "preset": str(preset)}

        if input_stream.time_base and start_s > 0:
            input_container.seek(
                int(max(0.0, start_s - 1.0) / float(input_stream.time_base)),
                stream=input_stream,
                backward=True,
                any_frame=False,
            )

        end_s = start_s + duration_s if duration_s is not None else None
        total_duration = duration_s
        if total_duration is None and input_stream.duration is not None:
            total_duration = max(
                0.0, float(input_stream.duration * input_time_base) - start_s
            )

        for decoded_index, frame in enumerate(input_container.decode(input_stream)):
            frame_time = _frame_time_seconds(frame, input_stream, decoded_index)
            if frame_time + 1e-9 < start_s:
                continue
            if end_s is not None and frame_time >= end_s:
                break
            if first_time is None:
                first_time = frame_time

            image = frame.to_image().convert("RGB")
            if image.size != (target_width, target_height):
                image = image.resize((target_width, target_height), Image.Resampling.LANCZOS)
            sample = sampler.sample(frame_time, sync_adjust_s=sync_adjust_s)
            rendered = renderer.render(image, sample)
            output_frame = av.VideoFrame.from_image(rendered)
            output_frame.pts = int(
                round((frame_time - first_time) / float(input_time_base))
            )
            output_frame.time_base = input_time_base
            for packet in output_stream.encode(output_frame):
                output_container.mux(packet)

            frames_written += 1
            last_time = frame_time
            if progress is not None and (
                frames_written == 1 or frames_written % max(1, int(float(input_rate))) == 0
            ):
                fraction = (
                    min(1.0, (frame_time - start_s) / total_duration)
                    if total_duration and total_duration > 0
                    else None
                )
                progress(frames_written, frame_time, fraction)

        for packet in output_stream.encode():
            output_container.mux(packet)
    except Exception:
        if output_container is not None:
            output_container.close()
            output_container = None
        if destination.exists():
            destination.unlink()
        raise
    finally:
        input_container.close()
        if output_container is not None:
            output_container.close()

    if frames_written == 0 or first_time is None or last_time is None:
        if destination.exists():
            destination.unlink()
        raise ValueError("The requested render range did not contain any video frames")
    return RenderResult(
        output_path=destination,
        frames=frames_written,
        start_time_s=first_time,
        end_time_s=last_time,
        width=target_width,
        height=target_height,
    )


def recording_summary(recordings: Iterable[RecordingPair]) -> pd.DataFrame:
    """Return a notebook-friendly table without opening the large MP4s."""

    rows = []
    for recording in recordings:
        rows.append(
            {
                "Session": recording.session_id,
                "Telemetry CSV": recording.telemetry_path.name,
                "Video": recording.video_path.name,
                "Filename offset (s)": recording.filename_offset_s,
                "Thumbnail": recording.thumbnail_path.name if recording.thumbnail_path else "",
            }
        )
    return pd.DataFrame(rows)


__all__ = [
    "DEFAULT_SAMPLE_RATE_HZ",
    "OverlayStyle",
    "PreviewResult",
    "RecordingPair",
    "RenderResult",
    "SyncInfo",
    "TelemetryOverlayRenderer",
    "TelemetrySample",
    "TelemetrySampler",
    "TrackProjection",
    "TrackReference",
    "VideoInfo",
    "attach_session_clock",
    "build_sync_info",
    "build_track_reference",
    "choose_reference_lap",
    "discover_recording_pairs",
    "estimate_lap_sample_rate_hz",
    "format_overlay_time",
    "load_overlay_telemetry",
    "probe_video",
    "recording_summary",
    "render_overlay_preview",
    "render_overlay_video",
    "select_recording",
]

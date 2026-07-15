from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import pytest

from telemetry_overlay import (
    OverlayStyle,
    RecordingPair,
    SyncInfo,
    TelemetryOverlayRenderer,
    TelemetrySampler,
    VideoInfo,
    attach_session_clock,
    build_sync_info,
    build_track_reference,
    discover_recording_pairs,
    format_overlay_delta,
    probe_video,
    render_overlay_preview,
    render_overlay_video,
    select_recording,
)


def synthetic_telemetry(rows: int = 400) -> pd.DataFrame:
    theta = np.linspace(0.0, 2.0 * np.pi, rows)
    row_index = np.arange(rows)
    latitude_origin = -33.805
    longitude_origin = 150.870
    metres_per_degree = np.pi * 6_371_000.0 / 180.0
    radius_m = 500.0
    latitude = latitude_origin + radius_m * np.sin(theta) / metres_per_degree
    longitude = longitude_origin + radius_m * np.cos(theta) / (
        metres_per_degree * np.cos(np.radians(latitude_origin))
    )
    corner = (row_index >= rows // 3) & (row_index < 2 * rows // 3)
    corner_index = np.arange(int(np.count_nonzero(corner)))
    tire_slips = []
    for phase in range(4):
        slip = np.full(rows, 0.027778)
        slip[corner] += 0.034722 * (1 + (corner_index + phase) % 12)
        slip[: max(1, rows // 20)] = 0.965278
        tire_slips.append(slip)
    frame = pd.DataFrame(
        {
            "Sample Index": np.arange(rows),
            "Lap": np.ones(rows),
            "Elapsed Time (ms)": np.arange(rows) * 20,
            "Speed (MPH)": np.linspace(0, 100, rows),
            "Latitude (decimal)": latitude,
            "Longitude (decimal)": longitude,
            "Lateral Acceleration (m/s^2)": np.sin(theta) * 9.80665,
            "Longitudinal Acceleration (m/s^2)": np.cos(theta) * 4.903325,
            "Throttle Position (%)": np.linspace(0, 100, rows),
            "Brake Pressure (bar)": np.linspace(-0.3, 30, rows),
            "Steering Angle (deg)": np.linspace(-30, 30, rows),
            "Yaw Rate (rad/s)": np.sin(theta),
            "Power Level (KW)": np.linspace(-100, 300, rows),
            "State of Charge (%)": np.linspace(80, 75, rows),
            "Front Inverter Temp (%)": np.full(rows, 0.70),
            "Rear Inverter Temp (%)": np.full(rows, 0.72),
            "Battery Temp (%)": np.full(rows, 0.666667),
            "Tire Pressure Front Left (bar)": np.full(rows, 2.6),
            "Tire Pressure Front Right (bar)": np.full(rows, 2.65),
            "Tire Pressure Rear Left (bar)": np.full(rows, 2.7),
            "Tire Pressure Rear Right (bar)": np.full(rows, 2.68),
            "Tire Slip Front Left (% est.)": tire_slips[0],
            "Tire Slip Front Right (% est.)": tire_slips[1],
            "Tire Slip Rear Left (% est.)": tire_slips[2],
            "Tire Slip Rear Right (% est.)": tire_slips[3],
            "Source": "synthetic.csv",
        }
    )
    return frame


def test_discover_pairs_uses_nearest_unique_timestamps(tmp_path: Path) -> None:
    names = (
        "telemetry-v1-2026-07-14-19_28_42.csv",
        "telemetry-v1-2026-07-14-19_28_47.csv",
        "laps-2026-07-14-19_28_44.mp4",
        "laps-2026-07-14-19_28_47.mp4",
        "laps-2026-07-14-19_28_47-thumb.png",
    )
    for name in names:
        (tmp_path / name).touch()

    recordings = discover_recording_pairs(tmp_path)

    assert [recording.filename_offset_s for recording in recordings] == [2.0, 0.0]
    assert recordings[0].video_path.name.endswith("19_28_44.mp4")
    assert recordings[1].thumbnail_path is not None
    assert select_recording(recordings, recordings[1].session_id) == recordings[1]


def test_sync_clock_end_aligns_and_preserves_every_row() -> None:
    telemetry = synthetic_telemetry(101)
    video = VideoInfo(
        path=Path("video.mp4"),
        width=640,
        height=360,
        duration_s=2.0,
        average_rate=30.0,
        frames=60,
        codec="h264",
        time_base=0.001,
        has_audio=False,
    )

    sync = build_sync_info(telemetry, video, video_start_offset_s=0.0)
    prepared = attach_session_clock(telemetry, sync)

    assert sync.sample_rate_hz == pytest.approx(50.0)
    assert prepared["Session Time (s)"].iloc[-1] == pytest.approx(2.0)
    assert len(prepared) == 101


def test_track_projection_and_sampler_scaling() -> None:
    telemetry = synthetic_telemetry()
    sync = SyncInfo(
        telemetry_rows=len(telemetry),
        sample_rate_hz=50.0,
        video_start_offset_s=0.0,
        telemetry_span_s=(len(telemetry) - 1) / 50.0,
        rate_source="test",
        lap_rate_hz=50.0,
    )
    track = build_track_reference(telemetry, lap=1, points=300)
    sampler = TelemetrySampler(telemetry, sync)
    sample = sampler.sample(2.0)
    corner_sample = sampler.sample(3.5)
    launch_sample = sampler.sample(0.0)
    projection = track.project(sample.latitude, sample.longitude)

    assert track.total_distance_m == pytest.approx(2 * np.pi * 500, rel=0.03)
    assert projection.progress == pytest.approx(0.25, abs=0.04)
    assert projection.distance_from_reference_m < 15
    assert sample.speed_kmh == pytest.approx(sample.speed_mph * 1.609344)
    assert sample.battery_thermal_pct == pytest.approx(66.6667)
    assert sample.front_inverter_thermal_pct == pytest.approx(70.0)
    assert sample.brake_bar >= 0
    assert launch_sample.regen_kw == pytest.approx(100.0)
    assert launch_sample.total_g == pytest.approx(0.5)
    assert all(np.isnan(value) for value in launch_sample.tire_slip_est_pct)
    assert sampler.tire_slip_baseline_pct == pytest.approx((2.7778,) * 4)
    assert sampler.tire_slip_deadband_pct == pytest.approx(3.4722)
    assert max(corner_sample.tire_slip_normalized) > 0.5
    assert track.lap_time_ms == pytest.approx(7_980.0)
    assert projection.reference_elapsed_ms == pytest.approx(
        sample.lap_elapsed_ms,
        abs=120.0,
    )
    assert track.delta_to_reference_ms(
        sample.lap_elapsed_ms,
        projection,
    ) == pytest.approx(0.0, abs=120.0)


def test_delta_format_and_panel_geometry_regression() -> None:
    telemetry = synthetic_telemetry()
    renderer = TelemetryOverlayRenderer(build_track_reference(telemetry, lap=1))

    assert format_overlay_delta(-179.4) == "-0.179"
    assert format_overlay_delta(842.6) == "+0.843"
    assert format_overlay_delta(float("nan")) == "--.---"
    assert renderer.panel_boxes((1440, 934)) == {
        "speed": (23, 23, 361, 195),
        "track": (1029, 23, 1417, 312),
        "inputs": (23, 762, 469, 911),
        "dynamics": (479, 762, 831, 911),
        "tire_slip": (841, 762, 1098, 911),
        "thermal": (1108, 762, 1417, 911),
    }

    with pytest.raises(ValueError, match="scale maxima"):
        OverlayStyle(max_regen_kw=0)


def test_overlay_renderer_returns_same_frame_size() -> None:
    telemetry = synthetic_telemetry()
    sync = SyncInfo(
        telemetry_rows=len(telemetry),
        sample_rate_hz=50.0,
        video_start_offset_s=0.0,
        telemetry_span_s=(len(telemetry) - 1) / 50.0,
        rate_source="test",
    )
    track = build_track_reference(telemetry, lap=1)
    renderer = TelemetryOverlayRenderer(
        track,
        OverlayStyle(show_tire_pressures=True),
    )
    sample = TelemetrySampler(telemetry, sync).sample(1.0)
    source = Image.new("RGB", (640, 360), "#18202a")

    rendered = renderer.render(source, sample)

    assert rendered.size == source.size
    assert rendered.mode == "RGB"
    assert np.asarray(rendered).std() > 0


def test_tiny_video_preview_and_render(tmp_path: Path) -> None:
    av = pytest.importorskip("av")
    input_path = tmp_path / "input.mp4"
    output_path = tmp_path / "output.mp4"

    with av.open(str(input_path), "w") as container:
        stream = container.add_stream("mpeg4", rate=10)
        stream.width = 320
        stream.height = 180
        stream.pix_fmt = "yuv420p"
        for index in range(10):
            pixels = np.zeros((180, 320, 3), dtype=np.uint8)
            pixels[:, :, 1] = index * 20
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)

    telemetry = synthetic_telemetry()
    info = probe_video(input_path)
    sync = build_sync_info(telemetry, info, video_start_offset_s=0.0)
    sampler = TelemetrySampler(telemetry, sync)
    renderer = TelemetryOverlayRenderer(build_track_reference(telemetry, lap=1))

    preview = render_overlay_preview(
        input_path,
        sampler,
        renderer,
        video_time_s=0.2,
        output_width=320,
    )
    result = render_overlay_video(
        input_path,
        sampler,
        renderer,
        output_path,
        duration_s=0.5,
        output_width=320,
        preset="ultrafast",
    )

    assert preview.image.size == (320, 180)
    assert result.frames >= 4
    assert output_path.stat().st_size > 0
    assert probe_video(output_path).duration_s > 0


def test_select_recording_rejects_unknown_session() -> None:
    started = datetime(2026, 7, 14, 19, 28, 47)
    recording = RecordingPair(
        telemetry_path=Path("telemetry.csv"),
        video_path=Path("video.mp4"),
        telemetry_started_at=started,
        video_started_at=started,
    )
    with pytest.raises(KeyError, match="Unknown recording"):
        select_recording([recording], "missing")

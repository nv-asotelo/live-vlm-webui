"""
Unit tests for source-URL validation on NetworkVideoTrack.

The allowlist is a security control, not a convenience: the URL arrives from whoever can reach
the WebUI, and ffmpeg will happily open `file:` and friends. These tests pin the boundary.

SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
"""

import pytest

from live_vlm_webui.rtsp_track import NetworkVideoTrack, RTSPVideoTrack, validate_source_url


@pytest.mark.parametrize(
    "url",
    [
        "rtsp://192.168.1.100:554/stream",
        "rtsp://admin:password@192.168.1.100:554/h264Preview_01_main",
        "rtsps://cam.example.com/stream",
        "http://192.168.1.50/mjpg/video.mjpg",
        "https://example.com/live/stream.m3u8",
        "rtmp://media.example.com/live/key",
        "srt://192.168.1.10:9000",
        "udp://127.0.0.1:9000",
    ],
)
def test_network_urls_allowed(url):
    assert validate_source_url(url) == url


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "/etc/shadow",
        "/dev/video0",
        "./local-clip.mp4",
        "v4l2:///dev/video0",
    ],
)
def test_local_sources_blocked_by_default(url):
    """Local reads must be opt-in: the URL is client-supplied."""
    with pytest.raises(ValueError, match="not permitted"):
        validate_source_url(url)

    assert validate_source_url(url, allow_local=True) == url


@pytest.mark.parametrize("url", ["concat:/etc/passwd", "data:text/plain;base64,QQ==", "gopher://x"])
def test_exotic_ffmpeg_schemes_always_blocked(url):
    """These stay blocked even with --allow-local-sources; none of them is a camera."""
    with pytest.raises(ValueError):
        validate_source_url(url)
    with pytest.raises(ValueError):
        validate_source_url(url, allow_local=True)


def test_empty_url_rejected():
    with pytest.raises(ValueError, match="empty"):
        validate_source_url("   ")


def test_scheme_is_case_insensitive():
    assert validate_source_url("RTSP://cam/stream") == "RTSP://cam/stream"


def test_rejected_url_never_opens_a_container():
    """Validation must happen before av.open(), or the block is theatre."""
    with pytest.raises(ValueError):
        NetworkVideoTrack("file:///etc/passwd")


def test_protocol_specific_options():
    """RTSP transport flags should not be sent to non-RTSP demuxers."""
    assert NetworkVideoTrack._default_options("rtsp://cam/s")["rtsp_transport"] == "tcp"
    assert "rtsp_transport" not in NetworkVideoTrack._default_options("http://h/v.mjpg")
    assert NetworkVideoTrack._default_options("http://h/v.mjpg")["reconnect"] == "1"
    assert "fifo_size" in NetworkVideoTrack._default_options("udp://127.0.0.1:9000")


def test_legacy_alias_still_importable():
    """The class shipped as RTSPVideoTrack; external imports must keep working."""
    assert RTSPVideoTrack is NetworkVideoTrack


def test_stop_during_active_decode_does_not_crash(tmp_path):
    """Stopping a live source must not free the container under the reader thread.

    Regression test. Decoding happens in an executor thread while stop() runs on the event loop;
    closing the container without serialising the two segfaults the whole server process (not an
    exception — a core dump), so this runs in a subprocess and asserts on the exit code.
    """
    import subprocess
    import sys
    import textwrap

    # A finite local file is enough: the reader is inside demux() when stop() lands.
    src = tmp_path / "clip.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=320x240:rate=30:duration=10",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-pix_fmt",
            "yuv420p",
            str(src),
        ],
        check=True,
        capture_output=True,
    )

    program = textwrap.dedent(f"""
        import asyncio
        from live_vlm_webui.rtsp_track import NetworkVideoTrack

        async def main():
            track = NetworkVideoTrack({str(src)!r}, allow_local=True)

            async def pump():
                try:
                    while True:
                        await track.recv()
                except Exception:
                    pass

            task = asyncio.create_task(pump())
            await asyncio.sleep(1.0)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            track.stop()
            await asyncio.sleep(1.0)
            print("SURVIVED")

        asyncio.run(main())
    """)

    result = subprocess.run(
        [sys.executable, "-c", program], capture_output=True, text=True, timeout=90
    )
    assert result.returncode == 0, f"process died with {result.returncode} (segfault = -11/139)"
    assert "SURVIVED" in result.stdout

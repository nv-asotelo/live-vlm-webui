"""
Unit tests for PushVideoTrack.

The property that matters most here is the bounded queue: a pusher that outruns the consumer
must drop frames rather than accumulate them, because an unbounded queue would turn a fast
camera into a memory leak on an edge device.

SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
"""

import asyncio

import cv2
import numpy as np
import pytest

from live_vlm_webui.push_track import PushVideoTrack


def make_jpeg(width=320, height=240, label=0) -> bytes:
    """A flat grey frame whose brightness encodes `label`.

    The label goes into all three channels, not one: JPEG subsamples chroma, so a value written
    only to the blue channel does not survive the round trip (1 decodes as 0, 4 as 3), and a test
    asserting on it would be testing JPEG, not this code. Luma is preserved to within a couple of
    levels, so `decoded_label()` can recover it.
    """
    img = np.full((height, width, 3), (label * 5) % 256, dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.tobytes()


def decoded_label(frame) -> int:
    """Recover the label from a decoded frame, undoing make_jpeg()'s encoding."""
    return int(round(frame.to_ndarray(format="bgr24").mean() / 5))


def test_push_and_receive():
    """A pushed frame comes back out of recv() with the right dimensions."""

    async def go():
        track = PushVideoTrack(source_name="test")
        track.push_encoded(make_jpeg(320, 240))
        frame = await track.recv()
        assert frame.width == 320
        assert frame.height == 240
        track.stop()

    asyncio.run(go())


def test_queue_is_bounded_and_drops_stale_frames():
    """Pushing without consuming must drop, never accumulate."""

    async def go():
        track = PushVideoTrack(source_name="test")
        for i in range(50):
            track.push_encoded(make_jpeg(label=i))

        stats = track.get_stats()
        # One frame is held in the single slot; every other push displaced a pending frame.
        assert track._queue.qsize() == 1
        assert stats["frames_received"] == 50
        assert stats["frames_dropped"] == 49

        # The frame still queued must be the NEWEST one, not the oldest.
        frame = await track.recv()
        assert decoded_label(frame) == 49
        track.stop()

    asyncio.run(go())


def test_pts_is_monotonic_and_wall_clock_based():
    """PTS must advance with real time so downstream latency math is meaningful."""

    async def go():
        track = PushVideoTrack(source_name="test")
        track.push_encoded(make_jpeg())
        first = await track.recv()
        await asyncio.sleep(0.15)
        track.push_encoded(make_jpeg())
        second = await track.recv()

        assert second.pts > first.pts
        elapsed = (second.pts - first.pts) * float(second.time_base)
        assert 0.1 < elapsed < 0.5, f"pts delta {elapsed}s does not track wall clock"
        track.stop()

    asyncio.run(go())


def test_invalid_payload_rejected_without_killing_track():
    """Garbage in must raise, but leave the track usable."""

    async def go():
        track = PushVideoTrack(source_name="test")

        with pytest.raises(ValueError):
            track.push_encoded(b"definitely not a jpeg")

        assert track.get_stats()["frames_failed"] == 1

        track.push_encoded(make_jpeg())
        frame = await track.recv()
        assert frame.width == 320
        track.stop()

    asyncio.run(go())


def test_stats_and_connected_state():
    """is_connected reflects whether frames are actually arriving."""

    async def go():
        track = PushVideoTrack(source_name="robot-cam")
        assert track.is_connected is False  # nothing pushed yet

        track.push_encoded(make_jpeg())
        assert track.is_connected is True

        stats = track.get_stats()
        assert stats["source"] == "robot-cam"
        assert stats["width"] == 320 and stats["height"] == 240
        assert stats["stopped"] is False

        track.stop()
        assert track.get_stats()["stopped"] is True
        assert track.is_connected is False

    asyncio.run(go())


def test_recv_raises_after_stop():
    """A stopped track must end the stream rather than hang forever."""
    from aiortc.mediastreams import MediaStreamError

    async def go():
        track = PushVideoTrack(source_name="test")
        track.stop()
        with pytest.raises(MediaStreamError):
            await track.recv()

    asyncio.run(go())

"""
Push Video Track for externally-supplied frames.

Where the other sources *pull* (the browser pushes a WebRTC track, `rtsp_track` opens a URL and
demuxes it), this track is *fed*: a client POSTs encoded frames to the ingestion API and they are
handed to the same VideoProcessorTrack pipeline. That covers every camera that cannot present
itself as either a browser device or a demuxable URL — robots with an SDK-only camera API
(e.g. Reachy Mini), industrial cameras behind vendor SDKs, or any source that can be reached from
a few lines of Python but speaks no streaming protocol.

Two deliberate design choices, both measured (`scripts/measure_source_ram.py`):

* **JPEG is decoded by PyAV, not OpenCV.** `av` is already resident for the WebRTC stack, so
  reusing its MJPEG decoder costs ~4 MB of RSS where routing the same frames through
  `cv2.imdecode` costs ~9 MB.
* **The frame queue holds exactly one frame.** A pusher that outruns the VLM must not be able to
  build a backlog: the newest frame replaces the pending one and the stale frame is dropped. A
  growing queue would trade RAM for staleness, which is the wrong trade for a live view.

SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
"""

import asyncio
import logging
import time
from fractions import Fraction
from typing import Optional

import av
from aiortc import VideoStreamTrack
from aiortc.mediastreams import MediaStreamError

logger = logging.getLogger(__name__)

# 90 kHz is the RTP video clock; VideoProcessorTrack reads pts * time_base as seconds when it
# computes frame latency, so the units have to match what a real WebRTC track would carry.
VIDEO_CLOCK_RATE = 90000


class PushVideoTrack(VideoStreamTrack):
    """
    Video track whose frames are supplied by an external client via the push API.

    Example:
        track = PushVideoTrack()
        track.push_encoded(jpeg_bytes)     # from the HTTP handler
        frame = await track.recv()         # from the processing pipeline
    """

    def __init__(self, source_name: str = "push", max_queue: int = 1):
        super().__init__()
        self.source_name = source_name
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=max_queue)
        self._codec = av.CodecContext.create("mjpeg", "r")
        self._stopped = False
        self._start_time: Optional[float] = None
        self._frames_received = 0
        self._frames_dropped = 0
        self._frames_failed = 0
        self._last_push_time: Optional[float] = None
        self._last_jpeg: Optional[bytes] = None
        self._width: Optional[int] = None
        self._height: Optional[int] = None

    def push_encoded(self, data: bytes) -> None:
        """
        Accept one encoded frame (JPEG) from a pusher.

        Called from the HTTP handler. Never blocks and never raises on a slow consumer: if a frame
        is already waiting, it is discarded in favour of this newer one.

        Raises:
            ValueError: if the payload cannot be decoded as a JPEG frame
        """
        if self._stopped:
            raise ValueError("track is stopped")

        try:
            frames = self._codec.decode(av.Packet(data))
        except Exception as e:
            self._frames_failed += 1
            raise ValueError(f"could not decode pushed frame: {e}") from e

        if not frames:
            self._frames_failed += 1
            raise ValueError("pushed payload contained no decodable frame")

        frame = frames[0]

        if self._start_time is None:
            self._start_time = time.time()
            self._width, self._height = frame.width, frame.height
            logger.info(
                f"Push source '{self.source_name}' received first frame: "
                f"{frame.width}x{frame.height}"
            )

        # Wall-clock PTS. VideoStreamTrack.next_timestamp() cannot be used here: it paces the
        # caller to a fixed 30 fps, but a push source arrives at whatever rate its client manages.
        frame.pts = int((time.time() - self._start_time) * VIDEO_CLOCK_RATE)
        frame.time_base = Fraction(1, VIDEO_CLOCK_RATE)

        if self._queue.full():
            try:
                self._queue.get_nowait()
                self._frames_dropped += 1
            except asyncio.QueueEmpty:
                pass

        try:
            self._queue.put_nowait(frame)
        except asyncio.QueueFull:
            # Consumer raced us to the slot; dropping is the correct outcome either way.
            self._frames_dropped += 1
            return

        self._frames_received += 1
        self._last_push_time = time.time()
        self._last_jpeg = data

    async def recv(self) -> av.VideoFrame:
        """Return the next pushed frame, waiting until one arrives."""
        if self._stopped:
            raise MediaStreamError("push track stopped")

        while True:
            try:
                return await asyncio.wait_for(self._queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                # No frame this second. Loop rather than fail: a push source is allowed to be
                # idle (an event-driven camera may push only when something happens).
                if self._stopped:
                    raise MediaStreamError("push track stopped")

    def stop(self):
        """Stop the track and release the decoder."""
        self._stopped = True
        self._last_jpeg = None
        logger.info(
            f"Push source '{self.source_name}' stopped: {self._frames_received} received, "
            f"{self._frames_dropped} dropped, {self._frames_failed} failed"
        )
        super().stop()

    @property
    def is_connected(self) -> bool:
        """True once a frame has arrived recently enough to call the source live."""
        if self._stopped or self._last_push_time is None:
            return False
        return (time.time() - self._last_push_time) < 10.0

    @property
    def last_jpeg(self) -> Optional[bytes]:
        """Most recently pushed frame, for the UI preview."""
        return self._last_jpeg

    def get_stats(self) -> dict:
        """Statistics about the push source, shaped like RTSPVideoTrack.get_stats()."""
        age = (time.time() - self._last_push_time) if self._last_push_time else None
        fps = None
        if self._start_time and self._frames_received > 1:
            elapsed = time.time() - self._start_time
            if elapsed > 0:
                fps = round(self._frames_received / elapsed, 2)
        return {
            "source": self.source_name,
            "connected": self.is_connected,
            "frames_received": self._frames_received,
            "frames_dropped": self._frames_dropped,
            "frames_failed": self._frames_failed,
            "seconds_since_last_frame": round(age, 2) if age is not None else None,
            "stopped": self._stopped,
            "width": self._width,
            "height": self._height,
            "fps": fps,
        }

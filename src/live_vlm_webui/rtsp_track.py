"""
Network Video Track for URL-addressable video sources

Provides a VideoStreamTrack that reads from any source PyAV can open: RTSP IP cameras, HTTP
MJPEG/HLS streams, RTMP/SRT feeds, local video files, and V4L2 device nodes. This is the "pull"
input — the server opens the source itself, in contrast to `push_track`, where a client feeds
frames in.

Originally RTSP-only; the demux/decode machinery was never RTSP-specific, only the labelling and
the connection options were. `RTSPVideoTrack` remains as an alias.

SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
"""

import av
import asyncio
import logging
import re
import threading
from typing import Optional
from urllib.parse import urlparse
from aiortc import VideoStreamTrack
from av import VideoFrame

# Suppress verbose ffmpeg/libav logging (HEVC decoder errors are normal for IP cameras)
# These POC/slice errors happen due to network packet loss but stream recovers automatically
av.logging.set_level(av.logging.FATAL)  # Only show fatal errors that stop the stream

logger = logging.getLogger(__name__)

# Schemes the server will open when a URL arrives over the API.
#
# This is an allowlist, not a blocklist, because the URL is attacker-controlled in any deployment
# where the WebUI is reachable by someone you do not fully trust: ffmpeg speaks protocols like
# `file:`, `concat:` and `data:` that would turn "point the camera at a stream" into "read a file
# off the server", and an `http://169.254.169.254/...` fetch is a classic cloud-metadata SSRF.
# Local sources (files, /dev/video*) are genuinely useful but are gated behind an explicit opt-in.
NETWORK_SCHEMES = frozenset(
    {"rtsp", "rtsps", "rtmp", "rtmps", "http", "https", "srt", "udp", "rtp", "tcp"}
)
LOCAL_SCHEMES = frozenset({"file", "v4l2", "pipe"})


def validate_source_url(url: str, allow_local: bool = False) -> str:
    """
    Check a source URL against the scheme allowlist.

    Args:
        url: Source URL or local path
        allow_local: Permit local files and device nodes (paths and file:/v4l2: schemes)

    Returns:
        The URL, unchanged, if permitted

    Raises:
        ValueError: If the scheme is not permitted
    """
    if not url or not url.strip():
        raise ValueError("source URL is empty")

    url = url.strip()
    scheme = (urlparse(url).scheme or "").lower()

    # No scheme means a bare filesystem path (/dev/video0, ./clip.mp4). A Windows drive letter
    # ("c:/...") parses as a single-character scheme, so treat that as a path too.
    if not scheme or len(scheme) == 1:
        if allow_local:
            return url
        raise ValueError(
            "local paths are not permitted; start the server with --allow-local-sources to enable"
        )

    if scheme in NETWORK_SCHEMES:
        return url

    if scheme in LOCAL_SCHEMES:
        if allow_local:
            return url
        raise ValueError(
            f"'{scheme}:' sources are not permitted; start the server with "
            f"--allow-local-sources to enable"
        )

    raise ValueError(
        f"unsupported URL scheme '{scheme}'. Permitted: {', '.join(sorted(NETWORK_SCHEMES))}"
    )


class NetworkVideoTrack(VideoStreamTrack):
    """
    Video track that reads from a URL-addressable source and yields aiortc VideoFrames.

    Handles RTSP/RTMP/SRT/HTTP streams, local files, and V4L2 devices through one code path, and
    reconnects automatically when a stream drops.

    Example:
        track = NetworkVideoTrack("rtsp://192.168.1.100:554/stream")
        track = NetworkVideoTrack("http://192.168.1.50/mjpg/video.mjpg")
        frame = await track.recv()
    """

    def __init__(
        self,
        url: str,
        reconnect_attempts: int = 5,
        reconnect_delay: float = 2.0,
        options: Optional[dict] = None,
        allow_local: bool = False,
    ):
        """
        Initialize the video track.

        Args:
            url: Source URL (rtsp://, http://, rtmp://, srt://, file://, or a local path)
            reconnect_attempts: Number of reconnection attempts on failure (default: 5)
            reconnect_delay: Base delay between reconnection attempts in seconds (default: 2.0)
            options: PyAV container options; defaults are chosen per protocol
            allow_local: Permit local files and device nodes
        """
        super().__init__()
        self.url = validate_source_url(url, allow_local=allow_local)
        self.reconnect_attempts = reconnect_attempts
        self.reconnect_delay = reconnect_delay
        self.container: Optional[av.container.InputContainer] = None
        self.stream: Optional[av.video.VideoStream] = None
        self._stopped = False
        self._frame_count = 0
        # Guards every touch of `container`. Decoding runs in an executor thread while stop() and
        # _reconnect() run on the event loop, so without this, closing a container while the
        # reader is inside demux() frees it underneath libav — a hard segfault, not an exception.
        self._io_lock = threading.Lock()

        self.options = options if options is not None else self._default_options(self.url)

        # Connect to stream
        self._connect()

    @property
    def rtsp_url(self) -> str:
        """Backwards-compatible alias for `url`."""
        return self.url

    @staticmethod
    def _default_options(url: str) -> dict:
        """
        Per-protocol container options.

        RTSP's transport flags are meaningless to other demuxers, so they are only set for RTSP.
        Passing them everywhere is harmless in practice but muddies logs and misleads readers.
        """
        scheme = (urlparse(url).scheme or "").lower()

        if scheme in ("rtsp", "rtsps"):
            return {
                "rtsp_transport": "tcp",  # TCP is more reliable than UDP for most networks
                "max_delay": "500000",  # 500ms max delay for low latency
                "rtsp_flags": "prefer_tcp",
            }

        if scheme in ("http", "https"):
            return {
                "reconnect": "1",
                "reconnect_streamed": "1",
                "reconnect_delay_max": "5",
                "max_delay": "500000",
            }

        if scheme in ("udp", "rtp", "srt", "tcp"):
            # Unreliable/bare transports need a buffer big enough to survive a burst.
            return {"fifo_size": "5000000", "max_delay": "500000"}

        return {}

    def _sanitize_url(self, url: str) -> str:
        """
        Remove password from URL for safe logging.

        Args:
            url: Source URL potentially containing credentials

        Returns:
            URL with password replaced by ****
        """
        return re.sub(r"://([^:]+):([^@]+)@", r"://\1:****@", url)

    def _connect(self):
        """
        Connect to RTSP stream.

        Raises:
            Exception: If connection fails after all attempts
        """
        safe_url = self._sanitize_url(self.url)

        try:
            logger.info(f"Connecting to video source: {safe_url}")

            self.container = av.open(self.url, options=self.options)

            # Get video stream
            if not self.container.streams.video:
                raise ValueError("No video stream found in source")

            self.stream = self.container.streams.video[0]

            # Log stream information
            codec = self.stream.codec_context.name
            width = self.stream.width or "unknown"
            height = self.stream.height or "unknown"
            fps = self.stream.average_rate or "unknown"

            logger.info(f"Source connected successfully: {codec} {width}x{height} @{fps}fps")

        except Exception as e:
            logger.error(f"Failed to connect to video source {safe_url}: {e}")
            raise

    async def recv(self) -> VideoFrame:
        """
        Receive next frame from RTSP stream.

        This is called by aiortc framework to get video frames.
        Runs demuxing/decoding in executor to avoid blocking event loop.

        Returns:
            VideoFrame: Next decoded video frame

        Raises:
            StopAsyncIteration: When stream ends or is stopped
        """
        if self._stopped:
            raise StopAsyncIteration

        try:
            # Read frame from container (blocking operation, run in executor)
            loop = asyncio.get_event_loop()
            frame = await loop.run_in_executor(None, self._read_frame)

            if frame is None:
                if not self._stopped:
                    logger.warning("Stream ended unexpectedly, attempting reconnection")
                    await self._reconnect()
                    # Try again after reconnection
                    frame = await loop.run_in_executor(None, self._read_frame)
                    if frame is None:
                        raise StopAsyncIteration
                else:
                    raise StopAsyncIteration

            self._frame_count += 1

            # Log progress periodically
            if self._frame_count % 300 == 0:  # Every ~10 seconds at 30fps
                logger.debug(f"Source: received {self._frame_count} frames")

            return frame

        except StopAsyncIteration:
            raise
        except Exception as e:
            logger.error(f"Error receiving frame: {e}", exc_info=True)
            # Try to reconnect on error
            if not self._stopped:
                await self._reconnect()
            raise

    def _read_frame(self) -> Optional[VideoFrame]:
        """
        Read and decode next frame from the source (blocking).

        This is a blocking operation and should be run in an executor.

        Returns:
            VideoFrame or None if stream ended or error occurred
        """
        with self._io_lock:
            if self._stopped:
                return None

            if not self.container or not self.stream:
                logger.error("Cannot read frame: container or stream not initialized")
                return None

            try:
                # Demux and decode packets until we get a video frame
                for packet in self.container.demux(self.stream):
                    for frame in packet.decode():
                        if isinstance(frame, VideoFrame):
                            return frame
                    # Bail out promptly if stop() landed while we were decoding.
                    if self._stopped:
                        return None

                # No more frames available (stream ended)
                logger.info("Stream reached end of file")
                return None

            except av.error.EOFError:
                logger.warning("Stream EOF")
                return None
            except Exception as e:
                logger.error(f"Error decoding frame: {e}")
                return None

    async def _reconnect(self):
        """
        Attempt to reconnect to the source with exponential backoff.

        Tries multiple times with increasing delay between attempts.
        """
        safe_url = self._sanitize_url(self.url)
        logger.info(f"Attempting reconnection to {safe_url}...")

        # Clean up existing connection. The lock keeps this from racing the reader thread.
        with self._io_lock:
            if self.container:
                try:
                    self.container.close()
                except Exception as e:
                    logger.debug(f"Error closing container during reconnect: {e}")
                self.container = None
                self.stream = None

        # Try to reconnect with exponential backoff
        for attempt in range(self.reconnect_attempts):
            try:
                logger.info(f"Reconnection attempt {attempt + 1}/{self.reconnect_attempts}")

                # Wait with exponential backoff (2, 4, 8, 16, 32 seconds)
                if attempt > 0:
                    delay = self.reconnect_delay * (2 ** (attempt - 1))
                    logger.info(f"Waiting {delay}s before reconnection attempt...")
                    await asyncio.sleep(delay)

                # Attempt connection
                self._connect()
                logger.info(f"Reconnected successfully on attempt {attempt + 1}")
                return

            except Exception as e:
                logger.warning(f"Reconnection attempt {attempt + 1} failed: {e}")
                if attempt == self.reconnect_attempts - 1:
                    logger.error(f"Reconnection failed after {self.reconnect_attempts} attempts")
                    raise

    def stop(self):
        """
        Stop the stream and clean up resources.

        Should be called when stream is no longer needed.
        """
        # Set the flag before taking the lock: a reader already inside demux() checks it between
        # packets and returns promptly, so this does not wait for a full read timeout.
        self._stopped = True

        with self._io_lock:
            if self.container:
                try:
                    self.container.close()
                    logger.info(f"Stream closed: {self._frame_count} frames received")
                except Exception as e:
                    logger.warning(f"Error closing container: {e}")
                finally:
                    self.container = None
                    self.stream = None

        super().stop()

    @property
    def is_connected(self) -> bool:
        """Check if the source is currently connected."""
        return self.container is not None and not self._stopped

    def get_stats(self) -> dict:
        """
        Get statistics about the source.

        Returns:
            Dictionary with stream statistics
        """
        stats = {
            "url": self._sanitize_url(self.url),
            "connected": self.is_connected,
            "frames_received": self._frame_count,
            "stopped": self._stopped,
        }

        if self.stream:
            stats.update(
                {
                    "codec": self.stream.codec_context.name,
                    "width": self.stream.width,
                    "height": self.stream.height,
                    "fps": float(self.stream.average_rate) if self.stream.average_rate else None,
                }
            )

        return stats


# The class was RTSP-only when introduced and is published under that name; keep it importable.
RTSPVideoTrack = NetworkVideoTrack

#!/usr/bin/env python3
"""Push a Reachy Mini's camera into live-vlm-webui.

Reachy Mini's camera is reachable only through its SDK: the daemon owns the hardware and hands
frames to clients over a local IPC endpoint or WebRTC. It is therefore neither a browser camera
device nor a demuxable stream URL, and it exposes no RTSP server — so neither of the WebUI's
original input modes can consume it. The push source exists for exactly this shape of camera.

Usage:
    pip install reachy-mini opencv-python-headless requests
    python push_reachy_mini.py --server http://localhost:8090

    # Robot on the network, WebUI on another host with its default self-signed cert:
    python push_reachy_mini.py --robot-host 192.168.1.42 \
        --server https://192.168.1.50:8090 --insecure \
        --session-id <id from the WebUI> --fps 5

Open the WebUI first, pick the "Push Source" tab and press Start: the tab shows the exact push
URL including the session id this script must target. Frames then appear in the preview and are
analysed by whichever VLM the server is configured against.

The session id matters. Each browser tab gets its own, so leaving this at "default" while the tab
is using a generated id means the frames are accepted and analysed but that tab shows nothing.

SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
"""

import argparse
import sys
import time

import cv2
import requests

try:
    from reachy_mini import ReachyMini
except ImportError:
    sys.exit(
        "reachy-mini is not installed. See https://huggingface.co/docs/reachy_mini/SDK/installation"
    )


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--server",
        default="http://localhost:8090",
        help="live-vlm-webui base URL (default: http://localhost:8090)",
    )
    p.add_argument(
        "--session-id",
        default="default",
        help="must match the session shown in the WebUI's Push Source tab",
    )
    p.add_argument("--fps", type=float, default=5.0, help="frames per second to push (default: 5)")
    p.add_argument(
        "--width",
        type=int,
        default=640,
        help="resize width before encoding; 0 keeps native size (default: 640)",
    )
    p.add_argument("--quality", type=int, default=85, help="JPEG quality (default: 85)")
    p.add_argument(
        "--robot-host",
        help="Reachy Mini address, e.g. 192.168.1.42 or reachy-mini.local. Required for a robot "
        "on the network: without it the SDK only auto-detects a daemon on localhost, and a "
        "wireless robot fails with 'both localhost and remote attempts failed'.",
    )
    p.add_argument(
        "--connection-mode",
        choices=["localhost_only", "network"],
        help="force the SDK connection mode instead of auto-detecting",
    )
    p.add_argument(
        "--insecure",
        action="store_true",
        help="skip TLS verification. live-vlm-webui serves HTTPS with a self-signed certificate "
        "by default, which requests rejects, so this is needed for any https:// server that has "
        "not been given a real certificate.",
    )
    args = p.parse_args()

    url = f"{args.server.rstrip('/')}/api/push/frame?session_id={args.session_id}&source_name=reachy-mini"
    interval = 1.0 / args.fps if args.fps > 0 else 0.0
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, args.quality]

    kwargs = {"media_backend": "default"}
    if args.robot_host:
        kwargs["host"] = args.robot_host
    if args.connection_mode:
        kwargs["connection_mode"] = args.connection_mode

    session = requests.Session()
    session.verify = not args.insecure
    if args.insecure:
        import urllib3

        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    pushed = failed = empty = 0

    print(f"pushing Reachy Mini camera -> {url} at {args.fps} fps")
    with ReachyMini(**kwargs) as mini:
        while True:
            t0 = time.time()
            try:
                # (height, width, 3) uint8 RGB, or None.
                # None is normal, not an error: the WebRTC stream needs a moment to negotiate, and
                # afterwards get_frame() returns whatever the last decoded frame was -- polling
                # faster than the stream delivers simply yields None. Treat it as "not yet".
                frame = mini.media.get_frame()
                if frame is None:
                    empty += 1
                    if empty in (20, 100) or empty % 300 == 0:
                        print(f"  waiting for video... ({empty} empty polls, {pushed} pushed)")
                    time.sleep(0.05)
                    continue
                empty = 0

                if args.width and frame.shape[1] > args.width:
                    h = int(frame.shape[0] * args.width / frame.shape[1])
                    frame = cv2.resize(frame, (args.width, h), interpolation=cv2.INTER_AREA)

                ok, buf = cv2.imencode(
                    ".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), encode_params
                )
                if not ok:
                    raise RuntimeError("JPEG encode failed")

                r = session.post(
                    url, data=buf.tobytes(), headers={"Content-Type": "image/jpeg"}, timeout=10
                )
                r.raise_for_status()
                pushed += 1
                if pushed % 25 == 0:
                    print(f"  pushed {pushed} frames ({failed} failed)")

            except KeyboardInterrupt:
                raise
            except Exception as e:
                failed += 1
                # Keep going: a dropped frame is not a reason to stop a live feed.
                print(f"  frame skipped: {type(e).__name__}: {e}")

            time.sleep(max(0.0, interval - (time.time() - t0)))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nstopped")

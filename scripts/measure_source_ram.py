#!/usr/bin/env python3
"""Measure the resident-memory cost of each video source type, in isolation.

Why this exists
    live-vlm-webui can ingest video from more than one kind of source, and on an 8 GB class
    edge device (Jetson Orin Nano) the RAM each source costs is a deployment constraint, not a
    footnote. This script measures that cost per source type so the cheapest viable source can
    be preferred.

Method
    Each mode runs in its own process (`--mode`), because Python's allocator does not return
    freed arenas to the OS promptly and running two modes in one process would attribute the
    first mode's high-water mark to the second. The driver (`--all`) re-execs this script once
    per mode and collects the results.

    Within a mode:
      1. Import and construct everything, then settle and record `baseline_rss`.
      2. Run a fixed workload: `--frames` frames at `--width`x`--height`, sampling RSS.
      3. Report baseline, mean, and peak RSS, and the delta over baseline.

    The VLM call path is deliberately NOT exercised — `VideoProcessorTrack.process_every_n_frames`
    is set beyond the frame count, so no inference request is made. This isolates the cost of the
    *source*, which is the variable under study; the VLM client cost is identical across modes.

Modes
    idle  — construct nothing; the floor for the interpreter plus imports
    push  — frames arrive pre-encoded as JPEG (the HTTP frame-push ingestion path):
            cv2.imdecode -> VideoFrame.from_ndarray -> single-slot asyncio.Queue
    url   — frames are demuxed and decoded from a URL by PyAV (the RTSP / network-URL path):
            av.open(url) -> container.demux -> packet.decode
"""

import argparse
import asyncio
import gc
import json
import os
import statistics
import subprocess
import sys
import time

import psutil

PROC = psutil.Process()


def rss_mb() -> float:
    return PROC.memory_info().rss / (1024 * 1024)


def settle() -> float:
    """Collect garbage and let the allocator quiesce, then read RSS."""
    gc.collect()
    time.sleep(0.5)
    gc.collect()
    time.sleep(0.5)
    return rss_mb()


# --------------------------------------------------------------------------- workloads
async def run_push(args, samples: list) -> int:
    """HTTP frame-push path: JPEG bytes in, decoded frame out, one-slot queue."""
    import cv2
    import numpy as np
    import av

    # A single-slot queue is the design under test: newest frame wins, no backlog can accumulate.
    queue: asyncio.Queue = asyncio.Queue(maxsize=1)
    payloads = args._payloads
    processed = 0
    for i in range(args.frames):
        jpeg = payloads[i % len(payloads)]
        # --- this is exactly what the ingestion endpoint does per pushed frame ---
        arr = np.frombuffer(jpeg, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        frame = av.VideoFrame.from_ndarray(img, format="bgr24")
        if queue.full():
            queue.get_nowait()  # drop the stale frame rather than block the pusher
        queue.put_nowait(frame)
        # --- consumer side (VideoProcessorTrack.recv would pull here) ---
        got = await queue.get()
        _ = got.to_ndarray(format="bgr24")
        processed += 1
        if i % 10 == 0:
            samples.append(rss_mb())
    return processed


async def run_push_av(args, samples: list) -> int:
    """HTTP frame-push path, JPEG decoded by PyAV instead of OpenCV.

    `av` is already resident (the WebRTC stack needs it), so reusing its MJPEG decoder avoids
    paying to initialise OpenCV's separate JPEG decoder just for ingestion.
    """
    import av

    queue: asyncio.Queue = asyncio.Queue(maxsize=1)
    payloads = args._payloads
    codec = av.CodecContext.create("mjpeg", "r")
    processed = 0
    for i in range(args.frames):
        jpeg = payloads[i % len(payloads)]
        # --- ingestion endpoint, PyAV variant ---
        packet = av.Packet(jpeg)
        frames = codec.decode(packet)
        if not frames:
            continue
        frame = frames[0]
        if queue.full():
            queue.get_nowait()
        queue.put_nowait(frame)
        got = await queue.get()
        _ = got.to_ndarray(format="bgr24")
        processed += 1
        if i % 10 == 0:
            samples.append(rss_mb())
    return processed


async def run_url(args, samples: list) -> int:
    """Network-URL path: PyAV opens the URL and runs a full demux + decode pipeline."""
    import av

    container = av.open(args.url, options={"rtsp_transport": "tcp", "max_delay": "500000"})
    stream = container.streams.video[0]
    processed = 0
    try:
        while processed < args.frames:
            for packet in container.demux(stream):
                for frame in packet.decode():
                    _ = frame.to_ndarray(format="bgr24")
                    processed += 1
                    if processed % 10 == 0:
                        samples.append(rss_mb())
                    if processed >= args.frames:
                        break
                if processed >= args.frames:
                    break
            else:
                # Source exhausted (a finite file); reopen to keep the workload going.
                container.close()
                container = av.open(args.url, options={"rtsp_transport": "tcp"})
                stream = container.streams.video[0]
                continue
            break
    finally:
        container.close()
    return processed


async def run_idle(args, samples: list) -> int:
    for _ in range(10):
        samples.append(rss_mb())
        await asyncio.sleep(0.05)
    return 0


MODES = {"idle": run_idle, "push": run_push, "push_av": run_push_av, "url": run_url}


def measure(args) -> dict:
    # Import the heavy third-party modules up front so they land in the baseline for every mode;
    # otherwise 'push' would be charged for importing cv2 and 'idle' would not.
    import cv2  # noqa: F401
    import numpy  # noqa: F401
    import av  # noqa: F401
    from live_vlm_webui.video_processor import VideoProcessorTrack

    # Ensure no VLM inference fires during the measurement.
    VideoProcessorTrack.process_every_n_frames = args.frames * 100

    # Setup phase, charged to the BASELINE rather than to the mode. The push modes need JPEG
    # payloads to consume, but in a real deployment the encoder runs in the pusher's process --
    # the server only ever decodes. Building them before the baseline keeps the comparison to
    # the thing that actually differs: the decode path each source type uses.
    args._payloads = []
    if args.mode in ("push", "push_av"):
        import numpy as np

        for i in range(min(args.frames, 60)):
            img = np.random.randint(0, 255, (args.height, args.width, 3), dtype=np.uint8)
            img[:, :, 0] = i % 255
            ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
            assert ok
            args._payloads.append(buf.tobytes())

    baseline = settle()
    samples: list = []
    t0 = time.time()
    processed = asyncio.run(MODES[args.mode](args, samples))
    elapsed = time.time() - t0
    peak = max(samples) if samples else baseline
    mean = statistics.mean(samples) if samples else baseline
    after = settle()

    return {
        "mode": args.mode,
        "frames": processed,
        "resolution": f"{args.width}x{args.height}",
        "elapsed_s": round(elapsed, 2),
        "baseline_rss_mb": round(baseline, 1),
        "mean_rss_mb": round(mean, 1),
        "peak_rss_mb": round(peak, 1),
        "settled_rss_mb": round(after, 1),
        "delta_peak_mb": round(peak - baseline, 1),
        "delta_settled_mb": round(after - baseline, 1),
    }


def run_all(args) -> None:
    results = []
    for mode in ("idle", "push", "push_av", "url"):
        if mode == "url" and not args.url:
            print(f"skipping '{mode}': no --url given", file=sys.stderr)
            continue
        cmd = [
            sys.executable,
            os.path.abspath(__file__),
            "--mode",
            mode,
            "--frames",
            str(args.frames),
            "--width",
            str(args.width),
            "--height",
            str(args.height),
            "--json",
        ]
        if args.url:
            cmd += ["--url", args.url]
        out = subprocess.run(cmd, capture_output=True, text=True)
        if out.returncode != 0:
            print(f"mode {mode} failed:\n{out.stderr}", file=sys.stderr)
            continue
        results.append(json.loads(out.stdout.strip().splitlines()[-1]))

    hdr = f"{'mode':8} {'frames':>7} {'baseline':>9} {'mean':>8} {'peak':>8} {'Δpeak':>8} {'Δsettled':>9}"
    print("\n" + hdr)
    print("-" * len(hdr))
    for r in results:
        print(
            f"{r['mode']:8} {r['frames']:>7} {r['baseline_rss_mb']:>8.1f}M "
            f"{r['mean_rss_mb']:>7.1f}M {r['peak_rss_mb']:>7.1f}M "
            f"{r['delta_peak_mb']:>7.1f}M {r['delta_settled_mb']:>8.1f}M"
        )
    print()


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--mode", choices=list(MODES), help="source type to measure")
    p.add_argument("--all", action="store_true", help="run every mode, each in its own process")
    p.add_argument("--frames", type=int, default=300, help="frames of workload (default: 300)")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--url", help="source URL for --mode url (http://, rtsp://, file path, ...)")
    p.add_argument("--json", action="store_true", help="emit one JSON line instead of a table")
    args = p.parse_args()

    if args.all:
        run_all(args)
        return
    if not args.mode:
        p.error("one of --mode or --all is required")
    if args.mode == "url" and not args.url:
        p.error("--mode url requires --url")

    result = measure(args)
    if args.json:
        print(json.dumps(result))
    else:
        for k, v in result.items():
            print(f"{k:20} {v}")


if __name__ == "__main__":
    main()

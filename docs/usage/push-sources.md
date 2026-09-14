# Push sources — feeding frames in from your own code

Live VLM WebUI's other two inputs both work by *pulling*: the browser hands over a camera it can
already see, or the server opens a stream URL and demuxes it. A **push source** inverts that. Your
code POSTs JPEG frames to the server, and they flow through the same VLM pipeline.

That covers the cameras the other two modes cannot reach:

- **Robots and devices with SDK-only cameras.** A [Reachy Mini](https://huggingface.co/docs/reachy_mini)
  hands frames to `mini.media.get_frame()`; its daemon owns the hardware and exposes no RTSP
  server, so it is neither a browser device nor a demuxable URL.
- **Cameras behind a vendor SDK** (industrial / machine-vision, thermal, depth).
- **Frames you produce rather than capture** — a rendered simulation, a decoded proprietary
  format, frames pulled off a message bus, or a pre-processed crop of a larger image.
- **A camera on a different machine from the server**, where you would rather push over HTTP than
  expose a streaming port.

If your camera *is* reachable as an RTSP/HTTP/file URL, prefer
[network URL sources](rtsp-ip-cameras.md) — the server pulls, so there is no client to keep
running.

---

## Quick start

1. Start the server and open the WebUI.
2. In **Video Source**, pick the **Push Source** tab.
3. Press **Start**. The tab shows the exact URL to push to, including the session id.
4. Point a client at that URL.

The smallest possible client:

```python
import cv2, requests

URL = "http://localhost:8090/api/push/frame?session_id=default"
cap = cv2.VideoCapture(0)

while True:
    ok, frame = cap.read()
    if not ok:
        break
    jpeg = cv2.imencode(".jpg", frame)[1].tobytes()
    requests.post(URL, data=jpeg, headers={"Content-Type": "image/jpeg"})
```

Or from the shell:

```bash
curl -X POST --data-binary @frame.jpg \
  -H "Content-Type: image/jpeg" \
  "http://localhost:8090/api/push/frame?session_id=default"
```

A complete Reachy Mini client is in
[`examples/push_reachy_mini.py`](../../examples/push_reachy_mini.py).

> **`session_id` must match the WebUI.** The tab displays the URL with the right one filled in.
> Push to a different id and the frames are processed, but that browser tab will not display them.

---

## API

### `POST /api/push/frame`

Push one frame. This is the only endpoint a simple client needs — the session is created on the
first frame, so no handshake is required.

| | |
|---|---|
| Query | `session_id` (default `default`), `source_name` (label shown in the UI) |
| Body | raw JPEG bytes with `Content-Type: image/jpeg`, **or** JSON `{"image": "<base64>", "session_id": "..."}` |

```json
{"status": "accepted", "session_id": "default", "frames_received": 42, "frames_dropped": 3}
```

| Status | Meaning |
|---|---|
| 200 | frame accepted |
| 400 | empty body, or the payload would not decode as a JPEG |
| 409 | the session was explicitly stopped — call `/api/push/start` before pushing again |

### `POST /api/push/start`

Start (or restart) a session explicitly. The WebUI's Start button calls this. Body:
`{"session_id": "...", "source_name": "..."}`.

### `POST /api/push/stop`

Stop a session and release its decoder. Body: `{"session_id": "..."}`.

After a stop, further pushed frames are refused with **409** until the session is started again.
This is deliberate: without it, a client that had not yet noticed the stop would silently
re-create the session and keep the VLM running — and being billed — behind a UI that shows
nothing.

### `GET /api/push/status`

```json
{"active_streams": 1, "streams": [{
  "session_id": "default", "source": "reachy-mini", "connected": true,
  "frames_received": 512, "frames_dropped": 31, "frames_failed": 0,
  "seconds_since_last_frame": 0.12, "width": 640, "height": 480, "fps": 9.8
}]}
```

`connected` is true when a frame arrived within the last 10 seconds.

### `GET /api/push/latest.jpg`

The most recently pushed frame, for the UI preview. Query: `session_id`.

---

## Behaviour worth knowing

**Frames are dropped, never queued.** The server holds exactly one pending frame per session. If
you push faster than the VLM consumes, the newest frame replaces the pending one and
`frames_dropped` climbs. This is intentional — a growing queue would trade memory for staleness,
and on an 8 GB edge device an unbounded queue turns a fast camera into an OOM. Dropped frames are
not an error; they mean your push rate exceeds the processing rate, which is normal for a live
view.

**Push rate and analysis rate are different things.** Every pushed frame is decoded, but only
every Nth reaches the VLM, per **Frame Processing Interval**. Pushing at 30 fps with an interval
of 30 gets you roughly one inference per second — and 29 of every 30 frames decoded for nothing.
Pushing at the rate you actually want analysed is cheaper.

**Resolution is yours to choose.** The server does not resize. Sending 4K frames to be downscaled
by the VLM anyway wastes encode time, bandwidth, and decode memory — resize client-side first.

**Idle is allowed.** A session with no frames arriving simply waits; an event-driven camera that
pushes only on motion is a valid client.

---

## Cost

Measured with [`scripts/measure_source_ram.py`](../../scripts/measure_source_ram.py) at 640×480:

| Source | RSS over baseline |
|---|---|
| Push (JPEG decoded by PyAV) | ~4 MB |
| Push, were it routed through OpenCV `imdecode` | ~9 MB |
| Network URL (PyAV demux + decode) | ~12 MB |

The cost is a one-time decoder initialisation, not per-frame growth: 300 frames and 900 frames
land within ~1 MB of each other, and a running server showed **no measurable RSS change** across
600 pushed frames. Push sources decode JPEG with PyAV rather than OpenCV precisely because `av` is
already resident for the WebRTC stack, so its decoder costs roughly half as much.

---

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| 409 on every frame | the session was stopped in the UI; press Start again (or call `/api/push/start`) |
| 400 `could not decode pushed frame` | the body is not a JPEG. PNG/raw arrays are not accepted — encode to JPEG first |
| Frames accepted but the UI shows nothing | `session_id` does not match the one in the Push Source tab |
| `frames_dropped` climbing fast | you are pushing faster than the VLM can consume — expected; lower your push rate to save CPU |
| Preview updates but no VLM text | check **Frame Processing Interval**, and that the VLM API base/model are reachable |
| `connected: false` while pushing | frames are failing before they arrive — check `frames_failed` and the client's HTTP status codes |

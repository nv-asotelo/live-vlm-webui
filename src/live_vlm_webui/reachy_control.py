"""
Reachy Mini control proxy.

The robot's daemon already exposes a REST API, so this is a thin, deliberate pass-through rather
than an SDK dependency: the WebUI process stays free of the SDK's GStreamer/WebRTC stack, which is
heavy and exists only to carry media the WebUI receives as pushed frames anyway.

Proxying rather than calling the daemon from the browser buys three things:
  * the robot's address stays server-side, so the page works unchanged from any client;
  * no CORS negotiation with a daemon we do not control;
  * no mixed-content block, since the WebUI is usually HTTPS and the daemon is plain HTTP.

Angles are degrees at this boundary and radians on the wire. Degrees are what the robot's own
documentation states its limits in, and what a person reasoning about "look up 20" is thinking in;
the daemon wants radians. Converting in one place keeps the UI and the limits legible.

SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
"""

import asyncio
import logging
import math
import time
from typing import Optional

import aiohttp

logger = logging.getLogger(__name__)

# Published safety limits (Reachy Mini "Core Concepts"). The SDK clamps to the nearest valid
# position rather than refusing, so clamping here too is about telling the user what happened
# instead of silently moving somewhere else.
LIMITS_DEG = {
    "pitch": (-40.0, 40.0),
    "roll": (-40.0, 40.0),
    "yaw": (-180.0, 180.0),
    "body_yaw": (-160.0, 160.0),
    "antenna": (-150.0, 150.0),
}
# Head translation limits, in millimetres, measured on the robot by commanding each axis and
# reading back what it achieved:
#   z: 10 -> 7.7, 20 -> 18.2, 30 -> 21.3 (saturated)   so useful travel ends near 20
#   x: 10 -> 9.7, 20 -> 16.6 (saturating)              so useful travel ends near 15
#   y: 10 -> 12.7, 20 -> 23.8 (tracking)               so 20 is comfortably inside
# This is a Stewart platform, so the axes are coupled: commanding z alone also shifts x and y by a
# few mm. The limits below keep commands inside the range where the platform actually follows.
LIMITS_MM = {
    "x": (-15.0, 15.0),
    "y": (-20.0, 20.0),
    "z": (-10.0, 20.0),
}

# The head may not be twisted more than this away from the body.
MAX_YAW_DELTA_DEG = 65.0

# How long set_target() trusts its own last command when holding an unspecified axis. Long enough
# that a dragged control never falls back to the measured pose mid-drag, short enough that moving
# the head by hand (or another client moving it) is adopted rather than fought.
CMD_MEMORY_S = 5.0

# Where "centre" parks the antennas, and the UI's default.
#
# Not 0: at rest an antenna servo hunts and visibly twitches, and holding a few degrees off the
# null position settles it. That is an observation of the physical robot, not of the telemetry -
# polling /api/state/full tops out around 2.5 Hz against a 50 Hz control loop, which is far too
# slow to characterise a twitch and will alias it. An earlier attempt to verify this from sampled
# positions "disproved" it and the change was reverted; the robot says otherwise, so the robot
# wins. Visually 10 deg is barely off level.
ANTENNA_PARK_DEG = 10.0


MOTOR_MODES = ("enabled", "disabled", "gravity_compensation")

# Per-motor torque is not in the daemon's REST API - only a global mode switch is. It is reachable
# over the SDK's WebSocket channel, which takes plain JSON, so this speaks that directly rather
# than pulling in the SDK (whose install drags the GStreamer/WebRTC media stack along with it).
# Names come from the SDK's hardware_config.yaml: body_rotation, stewart_1..6, left_antenna,
# right_antenna.
ANTENNA_MOTORS = {"left": ["left_antenna"], "right": ["right_antenna"],
                  "both": ["left_antenna", "right_antenna"]}
INTERPOLATIONS = ("linear", "minjerk", "ease_in_out", "cartoon")


class ReachyControl:
    """Async client for one Reachy Mini daemon."""

    def __init__(self, host: str, port: int = 8000, timeout: float = 10.0):
        host = (host or "").strip()
        if host.startswith("http://") or host.startswith("https://"):
            self.base = host.rstrip("/")
        else:
            self.base = f"http://{host}:{port}"
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        # Last pose this client commanded, so set_target() can hold an unspecified axis at
        # what it was ASKED to be rather than at what the platform settled on. See set_target.
        self._last_cmd: dict = {}
        self._last_cmd_at = 0.0

    # ------------------------------------------------------------------ helpers
    async def _get(self, path: str) -> dict:
        async with aiohttp.ClientSession(timeout=self.timeout) as s:
            async with s.get(f"{self.base}{path}") as r:
                r.raise_for_status()
                return await r.json()

    async def _post(self, path: str, payload: Optional[dict] = None) -> dict:
        async with aiohttp.ClientSession(timeout=self.timeout) as s:
            async with s.post(f"{self.base}{path}", json=payload) as r:
                if r.status >= 400:
                    raise RuntimeError(f"{path} -> HTTP {r.status}: {(await r.text())[:200]}")
                try:
                    return await r.json()
                except Exception:
                    return {}

    @staticmethod
    def _clamp(value: float, key: str) -> tuple[float, bool]:
        lo, hi = (LIMITS_MM if key in LIMITS_MM else LIMITS_DEG)[key]
        clamped = max(lo, min(hi, value))
        return clamped, clamped != value

    # ------------------------------------------------------------------ state
    async def state(self) -> dict:
        """Pose, motor mode, and whether the robot is actually ready to move."""
        status = await self._get("/api/daemon/status")
        backend = status.get("backend_status") or {}
        out = {
            "reachable": True,
            "robot_name": status.get("robot_name"),
            "version": status.get("version"),
            "state": status.get("state"),
            # `ready` is False until the robot has been woken; it is not an error condition.
            "ready": bool(backend.get("ready")),
            "motor_mode": backend.get("motor_control_mode"),
            "wireless": status.get("wireless_version"),
            # Handed to the UI so it can link out to the daemon's own admin pages (/docs, /logs).
            # The browser reaches the robot directly for these; they are not proxied, because the
            # point is full unmediated access to the daemon.
            "admin_base": self.base,
        }
        try:
            full = await self._get("/api/state/full")
            pose = full.get("head_pose") or {}
            out["pose_deg"] = {
                k: round(math.degrees(float(pose.get(k, 0.0))), 1)
                for k in ("roll", "pitch", "yaw")
            }
            out["pos_mm"] = {k: round(float(pose.get(k, 0.0)) * 1000.0, 1) for k in ("x", "y", "z")}
            out["body_yaw_deg"] = round(math.degrees(float(full.get("body_yaw") or 0.0)), 1)
            out["antennas_deg"] = [
                round(math.degrees(float(a)), 1) for a in (full.get("antennas_position") or [])
            ]
            out["control_mode"] = full.get("control_mode")
        except Exception as e:  # pose is a nice-to-have; status alone is still useful
            logger.debug(f"Reachy pose unavailable: {e}")
        return out

    # ------------------------------------------------------------------ actions
    async def wake_up(self) -> dict:
        await self._post("/api/move/play/wake_up")
        return {"message": "waking up"}

    async def go_to_sleep(self) -> dict:
        await self._post("/api/move/play/goto_sleep")
        return {"message": "going to sleep"}

    async def set_motor_mode(self, mode: str) -> dict:
        if mode not in MOTOR_MODES:
            raise ValueError(f"mode must be one of {', '.join(MOTOR_MODES)}")
        await self._post(f"/api/motors/set_mode/{mode}")
        return {"message": f"motors {mode}"}

    async def set_antenna_power(self, side: str, on: bool) -> dict:
        """Cut or restore torque to one antenna (or both) without touching the head.

        A powered servo that is hunting will twitch; with torque off it simply goes slack, and the
        head keeps holding position because only these motor ids are addressed.
        """
        side = (side or "").lower()
        if side not in ANTENNA_MOTORS:
            raise ValueError(f"side must be one of {', '.join(ANTENNA_MOTORS)}")

        ids = ANTENNA_MOTORS[side]
        ws_url = self.base.replace("https://", "wss://").replace("http://", "ws://") + "/ws/sdk"
        payload = {"type": "set_torque", "on": bool(on), "ids": ids}

        async with aiohttp.ClientSession(timeout=self.timeout) as s:
            async with s.ws_connect(ws_url) as ws:
                await ws.send_json(payload)
                # The daemon does not acknowledge commands; give it a moment on the wire before
                # the connection closes underneath it.
                await asyncio.sleep(0.3)

        return {"message": f"{side} antenna power {'restored' if on else 'cut'}",
                "motors": ids, "torque": bool(on)}

    async def goto(
        self,
        pitch: Optional[float] = None,
        yaw: Optional[float] = None,
        roll: Optional[float] = None,
        body_yaw: Optional[float] = None,
        antennas: Optional[list] = None,
        x: Optional[float] = None,
        y: Optional[float] = None,
        z: Optional[float] = None,
        duration: float = 1.0,
        interpolation: str = "minjerk",
    ) -> dict:
        """Move the head. Angles in DEGREES, translation in MILLIMETRES.

        Sign conventions are the robot's own, not flattened for convenience:
        positive pitch tilts the head DOWN (verified from the camera: +25 deg pitch moved the
        scene up 80 px in frame). Callers wanting "look up" send negative pitch. Translation is
        converted to metres for the daemon, which is what XYZRPYPose expects.
        """
        if interpolation not in INTERPOLATIONS:
            raise ValueError(f"interpolation must be one of {', '.join(INTERPOLATIONS)}")

        duration = max(0.1, min(10.0, float(duration)))
        notes = []

        # Start from where the robot is, so a control that only sets pitch does not silently
        # reset yaw to zero.
        current = await self.state()
        cur_pose = current.get("pose_deg") or {}
        cur_body = current.get("body_yaw_deg") or 0.0
        cur_pos = current.get("pos_mm") or {}

        # With motors disabled the daemon accepts a move and returns success, but nothing turns.
        # Reporting "moving" for a command that cannot move is worse than refusing it: the robot
        # sits still while the UI claims it worked, and there is nothing to debug from.
        if (current.get("motor_mode") or "").lower() == "disabled":
            raise ValueError(
                "motors are disabled, so the robot cannot move - press Stiff (or Wake) first"
            )
        if (current.get("motor_mode") or "").lower() == "gravity_compensation":
            notes.append("motors are in Soft mode; the head is compliant and may not hold position")

        pitch = cur_pose.get("pitch", 0.0) if pitch is None else float(pitch)
        yaw = cur_pose.get("yaw", 0.0) if yaw is None else float(yaw)
        roll = cur_pose.get("roll", 0.0) if roll is None else float(roll)
        body = cur_body if body_yaw is None else float(body_yaw)
        tx = cur_pos.get("x", 0.0) if x is None else float(x)
        ty = cur_pos.get("y", 0.0) if y is None else float(y)
        tz = cur_pos.get("z", 0.0) if z is None else float(z)
        for axis, val in (("x", tx), ("y", ty), ("z", tz)):
            clamped, hit = self._clamp(val, axis)
            if hit:
                notes.append(f"{axis} clamped to {clamped:g} mm")
            if axis == "x":
                tx = clamped
            elif axis == "y":
                ty = clamped
            else:
                tz = clamped

        for name, val in (("pitch", pitch), ("roll", roll), ("yaw", yaw), ("body_yaw", body)):
            clamped, hit = self._clamp(val, name)
            if hit:
                notes.append(f"{name} clamped to {clamped:g}°")
            if name == "pitch":
                pitch = clamped
            elif name == "roll":
                roll = clamped
            elif name == "yaw":
                yaw = clamped
            else:
                body = clamped

        # The head may not be twisted more than MAX_YAW_DELTA_DEG from the body. Rather than
        # refuse, bring the body along - which is what a person means by "look further right".
        if abs(yaw - body) > MAX_YAW_DELTA_DEG:
            body = self._clamp(
                yaw - math.copysign(MAX_YAW_DELTA_DEG, yaw - body), "body_yaw"
            )[0]
            notes.append(f"body turned to {body:g}° to stay within the {MAX_YAW_DELTA_DEG:g}° limit")

        payload: dict = {
            "head_pose": {
                "x": tx / 1000.0, "y": ty / 1000.0, "z": tz / 1000.0,
                "roll": math.radians(roll),
                "pitch": math.radians(pitch),
                "yaw": math.radians(yaw),
            },
            "body_yaw": math.radians(body),
            "duration": duration,
            "interpolation": interpolation,
        }

        if antennas is not None:
            vals = []
            for a in list(antennas)[:2]:
                clamped, hit = self._clamp(float(a), "antenna")
                if hit:
                    notes.append(f"antenna clamped to {clamped:g}°")
                vals.append(math.radians(clamped))
            if len(vals) == 2:
                payload["antennas"] = vals

        await self._post("/api/move/goto", payload)
        return {
            "message": "moving",
            "applied_deg": {"pitch": pitch, "yaw": yaw, "roll": roll, "body_yaw": body},
            "applied_mm": {"x": tx, "y": ty, "z": tz},
            "notes": notes,
        }

    async def set_target(
        self,
        pitch: Optional[float] = None,
        yaw: Optional[float] = None,
        roll: Optional[float] = None,
        body_yaw: Optional[float] = None,
        antennas: Optional[list] = None,
        x: Optional[float] = None,
        y: Optional[float] = None,
        z: Optional[float] = None,
    ) -> dict:
        """Set the pose the robot should be tracking, and return at once. Degrees and millimetres.

        This is goto()'s sibling for *live* controls - sliders and drag pads. goto() plans an
        interpolated trajectory of a given duration, which is right for "centre yourself" and wrong
        for a control being dragged: each new position queues another overlapping trajectory, so
        the head lags the finger and then catches up in lurches. set_target() only updates the
        setpoint that the robot's own 50 Hz loop is already chasing, so the newest value always
        wins and the motion is as smooth as that loop.

        Same clamping, same motor-mode refusal and same head-vs-body twist limit as goto(); the
        only difference is the endpoint and the absence of a duration.
        """
        notes = []
        current = await self.state()
        cur_pose = current.get("pose_deg") or {}
        cur_body = current.get("body_yaw_deg") or 0.0
        cur_pos = current.get("pos_mm") or {}

        if (current.get("motor_mode") or "").lower() == "disabled":
            raise ValueError(
                "motors are disabled, so the robot cannot move - press Stiff (or Wake) first"
            )

        # An unspecified axis holds its LAST COMMANDED value, not its measured one.
        #
        # Measured looks more obviously correct and is wrong: this is a Stewart platform that
        # settles a degree or two off target, so feeding the measurement back as the next command
        # amplifies that offset every call. Measured live, roll walked 3.7 -> 5.1 -> 8.6 -> 10.0
        # over four requests that never mentioned roll - a slow head tilt with no cause the user
        # could see. The cache is dropped after CMD_MEMORY_S so that moving the robot by hand, or
        # any other client, is still picked up rather than being fought.
        held = self._last_cmd if (time.monotonic() - self._last_cmd_at) < CMD_MEMORY_S else {}

        def hold(key, given, measured):
            if given is not None:
                return float(given)
            return float(held.get(key, measured))

        pitch = hold("pitch", pitch, cur_pose.get("pitch", 0.0))
        yaw = hold("yaw", yaw, cur_pose.get("yaw", 0.0))
        roll = hold("roll", roll, cur_pose.get("roll", 0.0))
        body = hold("body_yaw", body_yaw, cur_body)
        tx = hold("x", x, cur_pos.get("x", 0.0))
        ty = hold("y", y, cur_pos.get("y", 0.0))
        tz = hold("z", z, cur_pos.get("z", 0.0))

        vals = {"x": tx, "y": ty, "z": tz}
        for axis in ("x", "y", "z"):
            vals[axis], hit = self._clamp(vals[axis], axis)
            if hit:
                notes.append(f"{axis} clamped to {vals[axis]:g} mm")
        tx, ty, tz = vals["x"], vals["y"], vals["z"]

        rot = {"pitch": pitch, "roll": roll, "yaw": yaw, "body_yaw": body}
        for name in ("pitch", "roll", "yaw", "body_yaw"):
            rot[name], hit = self._clamp(rot[name], name)
            if hit:
                notes.append(f"{name} clamped to {rot[name]:g}°")
        pitch, roll, yaw, body = rot["pitch"], rot["roll"], rot["yaw"], rot["body_yaw"]

        if abs(yaw - body) > MAX_YAW_DELTA_DEG:
            body = self._clamp(
                yaw - math.copysign(MAX_YAW_DELTA_DEG, yaw - body), "body_yaw"
            )[0]
            notes.append(f"body turned to {body:g}° to stay within the {MAX_YAW_DELTA_DEG:g}° limit")

        payload: dict = {
            "target_head_pose": {
                "x": tx / 1000.0, "y": ty / 1000.0, "z": tz / 1000.0,
                "roll": math.radians(roll),
                "pitch": math.radians(pitch),
                "yaw": math.radians(yaw),
            },
            "target_body_yaw": math.radians(body),
        }
        if antennas is not None:
            av = []
            for a in list(antennas)[:2]:
                clamped, hit = self._clamp(float(a), "antenna")
                if hit:
                    notes.append(f"antenna clamped to {clamped:g}°")
                av.append(math.radians(clamped))
            if len(av) == 2:
                payload["target_antennas"] = av

        await self._post("/api/move/set_target", payload)
        self._last_cmd = {"pitch": pitch, "yaw": yaw, "roll": roll, "body_yaw": body,
                          "x": tx, "y": ty, "z": tz}
        self._last_cmd_at = time.monotonic()
        return {
            "message": "tracking",
            "applied_deg": {"pitch": pitch, "yaw": yaw, "roll": roll, "body_yaw": body},
            "applied_mm": {"x": tx, "y": ty, "z": tz},
            "notes": notes,
        }

    async def center(self, duration: float = 1.0) -> dict:
        """Return to the neutral pose, antennas down.

        Antennas park at ANTENNA_PARK_DEG rather than 0 to keep them out of the position where
        they twitch. If the twitch still appears, cut antenna torque with set_antenna_power();
        the daemon exposes no servo gain or deadband setting to tune.
        """
        res = await self.goto(
            pitch=0, yaw=0, roll=0, body_yaw=0, x=0, y=0, z=0,
            antennas=[ANTENNA_PARK_DEG, ANTENNA_PARK_DEG],
            duration=duration,
        )
        res["message"] = "centering"
        return res

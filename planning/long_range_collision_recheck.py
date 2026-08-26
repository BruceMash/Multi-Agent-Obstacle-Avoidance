"""Independent trajectory collision audit for long-range benchmark records.

The online environment terminates on the first discrete post-transition
collision.  This module reproduces that contract from a frozen manifest entry
and a stored position trajectory while additionally decomposing obstacle
collisions into static and dynamic primitives.  It is evaluation-only and is
never called by a controller.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from Entity.dynamic_obstacles import MovingSphereObstacle
from Entity.static_obstacles import (
    AxisAlignedBoxObstacle,
    StaticCylinderObstacle,
)


def _obstacle_from_spec(spec: Mapping[str, Any]) -> Any:
    kind = str(spec["type"])
    if kind == "box":
        return AxisAlignedBoxObstacle(
            center=np.asarray(spec["center"], dtype=float),
            half_extents=np.asarray(spec["half_extents"], dtype=float),
            safety_margin=float(spec.get("safety_margin", 0.0)),
        )
    if kind == "cylinder":
        return StaticCylinderObstacle(
            center=np.asarray(spec["center"], dtype=float),
            radius=float(spec["radius"]),
            half_height=float(spec["half_height"]),
            safety_margin=float(spec.get("safety_margin", 0.0)),
        )
    if kind == "moving_sphere_constant_translation":
        return MovingSphereObstacle(
            center=np.asarray(spec["center"], dtype=float),
            radius=float(spec["radius"]),
            velocity=np.asarray(spec["velocity"], dtype=float),
            safety_margin=float(spec.get("safety_margin", 0.0)),
            bounds=None,
        )
    raise ValueError(f"unsupported long-range obstacle type: {kind}")


def audit_trajectory_collisions(
    positions: np.ndarray,
    entry: Mapping[str, Any],
    *,
    collision_margin_m: float = 0.0,
    peer_threshold_m: float = 0.6,
) -> dict[str, Any]:
    """Recompute discrete collision labels and minimum signed clearances.

    Frame zero is the initial condition.  Only frames 1..T are executed
    post-transition states, matching ``MultiAgentDMPEnv._check_collision``.
    Dynamic obstacle track index ``t`` is paired with trajectory frame ``t``.
    """

    frames = np.asarray(positions, dtype=float)
    if frames.ndim != 3 or frames.shape[1:] != (3, 3):
        raise ValueError("positions must have shape [time, 3 agents, 3 axes]")
    if len(frames) < 1:
        raise ValueError("positions must retain at least the initial frame")

    static_obstacles = [
        _obstacle_from_spec(spec) for spec in entry.get("static_obstacles", [])
    ]
    dynamic_obstacles = [
        _obstacle_from_spec(spec) for spec in entry.get("dynamic_obstacles", [])
    ]
    dynamic_tracks = entry.get("dynamic_obstacle_trajectories", [])
    if len(dynamic_obstacles) != len(dynamic_tracks):
        raise ValueError("dynamic obstacle/track count mismatch")

    lower, upper = np.asarray(entry["workspace_bounds"], dtype=float)
    flags = {"static": False, "dynamic": False, "peer": False, "boundary": False}
    first: dict[str, int | None] = {key: None for key in flags}
    agent_flags = {
        key: np.zeros(frames.shape[1], dtype=bool) for key in flags
    }
    minimum_static = float("inf")
    minimum_dynamic = float("inf")
    minimum_peer = float("inf")
    minimum_boundary = float("inf")
    agent_minimum_static = np.full(frames.shape[1], float("inf"), dtype=float)
    agent_minimum_dynamic = np.full(frames.shape[1], float("inf"), dtype=float)
    agent_minimum_peer = np.full(frames.shape[1], float("inf"), dtype=float)
    agent_minimum_boundary = np.full(frames.shape[1], float("inf"), dtype=float)

    for step in range(1, len(frames)):
        frame = frames[step]
        for obstacle_id, obstacle in enumerate(dynamic_obstacles):
            track = np.asarray(dynamic_tracks[obstacle_id], dtype=float)
            obstacle.center = track[min(step, len(track) - 1)].copy()

        for agent_id, point in enumerate(frame):
            boundary_clearance = float(
                np.min(np.concatenate((point - lower, upper - point)))
            )
            minimum_boundary = min(minimum_boundary, boundary_clearance)
            agent_minimum_boundary[agent_id] = min(
                agent_minimum_boundary[agent_id], boundary_clearance
            )
            if boundary_clearance < 0.0:
                flags["boundary"] = True
                agent_flags["boundary"][agent_id] = True
                if first["boundary"] is None:
                    first["boundary"] = step

            for obstacle in static_obstacles:
                signed = float(obstacle.signed_distance(point))
                minimum_static = min(minimum_static, signed)
                agent_minimum_static[agent_id] = min(
                    agent_minimum_static[agent_id], signed
                )
                if signed <= float(collision_margin_m):
                    flags["static"] = True
                    agent_flags["static"][agent_id] = True
                    if first["static"] is None:
                        first["static"] = step

            for obstacle in dynamic_obstacles:
                signed = float(obstacle.signed_distance(point))
                minimum_dynamic = min(minimum_dynamic, signed)
                agent_minimum_dynamic[agent_id] = min(
                    agent_minimum_dynamic[agent_id], signed
                )
                if signed <= float(collision_margin_m):
                    flags["dynamic"] = True
                    agent_flags["dynamic"][agent_id] = True
                    if first["dynamic"] is None:
                        first["dynamic"] = step

        for left in range(frame.shape[0]):
            for right in range(left + 1, frame.shape[0]):
                distance = float(np.linalg.norm(frame[left] - frame[right]))
                minimum_peer = min(minimum_peer, distance)
                agent_minimum_peer[[left, right]] = np.minimum(
                    agent_minimum_peer[[left, right]], distance
                )
                if distance <= float(peer_threshold_m):
                    flags["peer"] = True
                    agent_flags["peer"][[left, right]] = True
                    if first["peer"] is None:
                        first["peer"] = step

    obstacle_collision = flags["static"] or flags["dynamic"]
    any_collision = obstacle_collision or flags["peer"] or flags["boundary"]
    minimum_obstacle = min(minimum_static, minimum_dynamic)
    return {
        "static_obstacle_collision": flags["static"],
        "dynamic_obstacle_collision": flags["dynamic"],
        "obstacle_collision": obstacle_collision,
        "inter_agent_collision": flags["peer"],
        "boundary_collision": flags["boundary"],
        "any_collision": any_collision,
        "minimum_static_obstacle_signed_clearance_m": minimum_static,
        "minimum_dynamic_obstacle_signed_clearance_m": minimum_dynamic,
        "minimum_obstacle_signed_clearance_m": minimum_obstacle,
        "minimum_inter_agent_distance_m": minimum_peer,
        "minimum_boundary_clearance_m": minimum_boundary,
        "first_static_obstacle_collision_step": first["static"],
        "first_dynamic_obstacle_collision_step": first["dynamic"],
        "first_inter_agent_collision_step": first["peer"],
        "first_boundary_collision_step": first["boundary"],
        "agent_static_obstacle_collision": agent_flags["static"].tolist(),
        "agent_dynamic_obstacle_collision": agent_flags["dynamic"].tolist(),
        "agent_obstacle_collision": np.logical_or(
            agent_flags["static"], agent_flags["dynamic"]
        ).tolist(),
        "agent_inter_agent_collision": agent_flags["peer"].tolist(),
        "agent_boundary_collision": agent_flags["boundary"].tolist(),
        "agent_any_collision": np.logical_or.reduce(
            tuple(agent_flags.values())
        ).tolist(),
        "agent_minimum_static_obstacle_signed_clearance_m": agent_minimum_static.tolist(),
        "agent_minimum_dynamic_obstacle_signed_clearance_m": agent_minimum_dynamic.tolist(),
        "agent_minimum_obstacle_signed_clearance_m": np.minimum(
            agent_minimum_static, agent_minimum_dynamic
        ).tolist(),
        "agent_minimum_inter_agent_distance_m": agent_minimum_peer.tolist(),
        "agent_minimum_boundary_clearance_m": agent_minimum_boundary.tolist(),
    }

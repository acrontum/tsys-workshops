#!/usr/bin/env python3
"""Print the installed Isaac Lab gear-assembly task ID."""

from __future__ import annotations

import gymnasium as gym
import isaaclab_tasks  # noqa: F401  # registers the task catalog


TASK_CANDIDATES = ("IsaacContrib-Factory-GearMesh-Direct",)


def resolve_task_id() -> str:
    for task_id in TASK_CANDIDATES:
        try:
            gym.spec(task_id)
        except gym.error.Error:
            continue
        return task_id
    raise RuntimeError(
        "The Franka Factory gear-insertion RL task is not registered. Tried: "
        + ", ".join(TASK_CANDIDATES)
    )


if __name__ == "__main__":
    print(resolve_task_id())


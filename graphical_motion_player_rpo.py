"""Reproduce graphical stick-figure keyframes with the RoboParty RPO in PyBullet.

The script transfers changes in the saved stick pose to matching landmarks on
the RPO, solves one approximate whole-body configuration per keyframe, and
smoothly interpolates the robot joint positions. This is kinematic rough-motion
generation; gravity, contact dynamics, ZMP, and balance are not evaluated.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import time

import numpy as np

try:
    import pybullet as p
    import pybullet_data
except ImportError as error:
    raise SystemExit("Install PyBullet first: python -m pip install pybullet") from error


SCRIPT_DIRECTORY = Path(__file__).resolve().parent
KEYFRAME_FILE = SCRIPT_DIRECTORY / "keyframes.json"
ROBOT_URDF = SCRIPT_DIRECTORY / "rpo_description" / "urdf" / "rpo.urdf"
ROBOT_NAME = "RPO"

PELVIS_HEIGHT = 0.758
HIP_PITCH_JOINT = "{side}_thigh_pitch_joint"
SECONDS_PER_SEGMENT = 1.5
PLAYBACK_HZ = 60
FOOT_ORIENTATION_WEIGHT = 1.00
SOLE_CLEARANCE = 0.003

POINT_LINK_JOINTS = {
    "torso": "torso_joint",
    "left hand": "left_elbow_yaw_joint",
    "right hand": "right_elbow_yaw_joint",
    "left elbow": "left_elbow_pitch_joint",
    "right elbow": "right_elbow_pitch_joint",
    "left knee": "left_knee_joint",
    "right knee": "right_knee_joint",
    "left foot": "left_ankle_roll_joint",
    "right foot": "right_ankle_roll_joint",
}
LANDMARK_LOCAL_OFFSETS = {
    "torso": np.array((0.0, 0.0, 0.20)),
    "left hand": np.array((0.14, 0.0, 0.0)),
    "right hand": np.array((0.14, 0.0, 0.0)),
}
POINT_NAMES = ("pelvis", *POINT_LINK_JOINTS)
FOOT_NAMES = ("left foot", "right foot")
POINT_WEIGHTS = {
    "pelvis": 1.8,
    "torso": 1.5,
    "left hand": 1.2,
    "right hand": 1.2,
    "left foot": 1.4,
    "right foot": 1.4,
    "left elbow": 0.55,
    "right elbow": 0.55,
    "left knee": 0.65,
    "right knee": 0.65,
}


@dataclass(frozen=True)
class StickKeyframe:
    pose: dict[str, np.ndarray]


def load_stick_keyframes(path: Path = KEYFRAME_FILE) -> list[StickKeyframe]:
    """Read and validate postures saved by graphical_motion_designer.py."""
    if not path.exists():
        raise SystemExit(
            f"No keyframes found at {path}. Open graphical_motion_designer.py "
            "and save at least two poses."
        )
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"Could not read {path.name}: {error}") from error

    if document.get("format") != "humanoid-motion-stick-keyframes":
        raise SystemExit(f"{path.name} is not a graphical motion keyframe file.")

    result = []
    for number, keyframe in enumerate(document.get("keyframes", []), start=1):
        source_pose = keyframe.get("pose", {})
        missing = [name for name in POINT_NAMES if name not in source_pose]
        if missing:
            raise SystemExit(
                f"Keyframe {number} is missing: {', '.join(missing)}"
            )
        pose = {
                name: np.asarray(source_pose[name], dtype=float)
                for name in POINT_NAMES
            }
        result.append(StickKeyframe(pose=pose))
    if len(result) < 2:
        raise SystemExit(
            "Save at least two keyframes in graphical_motion_designer.py before "
            "running graphical_motion_player_rpo.py."
        )
    return result


def joint_indices(robot: int) -> dict[str, int]:
    return {
        p.getJointInfo(robot, joint)[1].decode("utf-8"): joint
        for joint in range(p.getNumJoints(robot))
    }


def movable_joints(robot: int) -> list[int]:
    return [
        joint
        for joint in range(p.getNumJoints(robot))
        if p.getJointInfo(robot, joint)[2] != p.JOINT_FIXED
    ]


def initialize_robot(robot: int, indices: dict[str, int]) -> None:
    """Use a nonsingular bent-knee stance and place both soles on the floor."""
    for side in ("left", "right"):
        p.resetJointState(
            robot, indices[HIP_PITCH_JOINT.format(side=side)], -0.12
        )
        p.resetJointState(robot, indices[f"{side}_knee_joint"], 0.24)
        p.resetJointState(robot, indices[f"{side}_ankle_pitch_joint"], -0.12)

    foot_links = [
        indices[POINT_LINK_JOINTS["left foot"]],
        indices[POINT_LINK_JOINTS["right foot"]],
    ]
    sole_height = min(p.getAABB(robot, link)[0][2] for link in foot_links)
    base_position, base_orientation = p.getBasePositionAndOrientation(robot)
    corrected = np.asarray(base_position, dtype=float)
    corrected[2] -= sole_height
    p.resetBasePositionAndOrientation(robot, corrected, base_orientation)


def read_configuration(robot: int, joints: list[int]) -> np.ndarray:
    base_position, _ = p.getBasePositionAndOrientation(robot)
    return np.concatenate(
        (
            np.asarray(base_position, dtype=float),
            np.asarray([p.getJointState(robot, joint)[0] for joint in joints]),
        )
    )


def apply_configuration(
    robot: int,
    values: np.ndarray,
    base_orientation: tuple[float, float, float, float],
    joints: list[int],
) -> None:
    p.resetBasePositionAndOrientation(robot, values[:3], base_orientation)
    for column, joint in enumerate(joints, start=3):
        info = p.getJointInfo(robot, joint)
        value = float(values[column])
        lower, upper = float(info[8]), float(info[9])
        if lower < upper:
            value = float(np.clip(value, lower, upper))
        p.resetJointState(robot, joint, value)


def prevent_floor_penetration(
    robot: int,
    foot_links: tuple[int, int],
    base_orientation: tuple[float, float, float, float],
) -> None:
    """Lift the complete pose when either foot collision shape enters the floor."""
    lowest_point = min(
        p.getAABB(robot, link)[0][2]
        for link in foot_links
    )
    if lowest_point >= SOLE_CLEARANCE:
        return
    base_position, _ = p.getBasePositionAndOrientation(robot)
    corrected_position = np.asarray(base_position, dtype=float)
    corrected_position[2] += SOLE_CLEARANCE - lowest_point
    p.resetBasePositionAndOrientation(
        robot, corrected_position, base_orientation
    )


def robot_points(
    robot: int, point_links: dict[str, int]
) -> dict[str, np.ndarray]:
    base_position, _ = p.getBasePositionAndOrientation(robot)
    points = {"pelvis": np.asarray(base_position, dtype=float)}
    for name, link in point_links.items():
        state = p.getLinkState(robot, link, computeForwardKinematics=True)
        position = np.asarray(state[4], dtype=float)
        local_offset = LANDMARK_LOCAL_OFFSETS.get(name)
        if local_offset is not None:
            position = np.asarray(
                p.multiplyTransforms(
                    position,
                    state[5],
                    local_offset,
                    (0.0, 0.0, 0.0, 1.0),
                )[0],
                dtype=float,
            )
        points[name] = position
    return points


def foot_orientations(
    robot: int, point_links: dict[str, int]
) -> dict[str, tuple[float, float, float, float]]:
    return {
        name: p.getLinkState(
            robot, point_links[name], computeForwardKinematics=True
        )[5]
        for name in FOOT_NAMES
    }


def orientation_features(
    orientations: dict[str, tuple[float, float, float, float]]
) -> np.ndarray:
    """Represent both foot rotations with scaled rotation-matrix entries."""
    return FOOT_ORIENTATION_WEIGHT * np.concatenate(
        [
            np.asarray(p.getMatrixFromQuaternion(orientations[name]), dtype=float)
            for name in FOOT_NAMES
        ]
    )


def solver_features(
    robot: int,
    point_links: dict[str, int],
) -> np.ndarray:
    return np.concatenate(
        (
            stack_points(robot_points(robot, point_links)),
            orientation_features(foot_orientations(robot, point_links)),
        )
    )


def stack_points(points: dict[str, np.ndarray]) -> np.ndarray:
    return np.concatenate(
        [
            POINT_WEIGHTS[name] * points[name]
            for name in POINT_NAMES
        ]
    )


def map_stick_motion_to_robot(
    keyframes: list[StickKeyframe],
    initial_robot_points: dict[str, np.ndarray],
) -> list[dict[str, np.ndarray]]:
    """Transfer relative stick motion to RPO landmarks with one body scale."""
    reference = keyframes[0].pose
    stick_height = reference["torso"][2] - 0.5 * (
        reference["left foot"][2] + reference["right foot"][2]
    )
    robot_height = initial_robot_points["torso"][2] - 0.5 * (
        initial_robot_points["left foot"][2]
        + initial_robot_points["right foot"][2]
    )
    scale = float(robot_height / max(stick_height, 1e-6))

    def map_point(name: str, point: np.ndarray) -> np.ndarray:
        displacement = point - reference[name]
        world_displacement = np.array(
            [displacement[1], -displacement[0], displacement[2]]
        )
        return initial_robot_points[name] + scale * world_displacement

    targets = []
    for keyframe in keyframes:
        frame = {}
        for name in POINT_NAMES:
            frame[name] = map_point(name, keyframe.pose[name])
        targets.append(frame)
    return targets


def solve_keyframe(
    robot: int,
    joints: list[int],
    base_orientation: tuple[float, float, float, float],
    point_links: dict[str, int],
    target_points: dict[str, np.ndarray],
    target_foot_orientations: dict[str, tuple[float, float, float, float]],
    max_iterations: int = 35,
) -> tuple[np.ndarray, float]:
    """Solve one approximate G1 whole-body pose using numerical differential IK."""
    target = np.concatenate(
        (
            stack_points(target_points),
            orientation_features(target_foot_orientations),
        )
    )
    values = read_configuration(robot, joints)
    regularization = np.ones(len(values))
    regularization[:3] = 0.35

    for _ in range(max_iterations):
        current = solver_features(robot, point_links)
        error = target - current
        if np.linalg.norm(error) < 8e-3:
            break

        epsilon = 1e-4
        jacobian = np.empty((len(current), len(values)))
        for column in range(len(values)):
            perturbed = values.copy()
            perturbed[column] += epsilon
            apply_configuration(robot, perturbed, base_orientation, joints)
            sample = solver_features(robot, point_links)
            jacobian[:, column] = (sample - current) / epsilon
        apply_configuration(robot, values, base_orientation, joints)

        weighted_transpose = (1.0 / regularization)[:, None] * jacobian.T
        damping = 2e-3
        increment = weighted_transpose @ np.linalg.solve(
            jacobian @ weighted_transpose + damping * np.eye(len(error)),
            error,
        )
        values += np.clip(increment, -0.06, 0.06)
        apply_configuration(robot, values, base_orientation, joints)
        values = read_configuration(robot, joints)

    final_error = float(
        np.linalg.norm(
            target - solver_features(robot, point_links)
        )
    )
    return values.copy(), final_error


def solve_motion(
    robot: int,
    keyframes: list[StickKeyframe],
    indices: dict[str, int],
) -> tuple[
    list[np.ndarray],
    list[int],
    tuple[float, float, float, float],
    list[float],
]:
    joints = movable_joints(robot)
    _, base_orientation = p.getBasePositionAndOrientation(robot)
    point_links = {
        name: indices[joint_name]
        for name, joint_name in POINT_LINK_JOINTS.items()
    }
    initial_points = robot_points(robot, point_links)
    fixed_foot_orientations = foot_orientations(robot, point_links)
    targets = map_stick_motion_to_robot(keyframes, initial_points)

    configurations = []
    errors = []
    for number, target in enumerate(targets, start=1):
        configuration, error = solve_keyframe(
            robot,
            joints,
            base_orientation,
            point_links,
            target,
            fixed_foot_orientations,
        )
        foot_links = (
            point_links["left foot"],
            point_links["right foot"],
        )
        prevent_floor_penetration(
            robot, foot_links, base_orientation
        )
        configuration = read_configuration(robot, joints)
        configurations.append(configuration)
        errors.append(error)
        print(f"Solved keyframe {number}/{len(targets)}: weighted error {error:.4f}")
    return configurations, joints, base_orientation, errors


def smoothstep(value: float) -> float:
    return value * value * (3.0 - 2.0 * value)


@dataclass(frozen=True)
class PlaybackFrame:
    configuration: np.ndarray


def interpolated_frames(
    configurations: list[np.ndarray],
) -> list[PlaybackFrame]:
    """Interpolate joint poses between consecutive solved keyframes."""
    samples = max(2, int(SECONDS_PER_SEGMENT * PLAYBACK_HZ))
    frames = []
    for start, end in zip(configurations[:-1], configurations[1:]):
        for sample in range(samples):
            fraction = sample / samples
            blend = smoothstep(fraction)
            frames.append(
                PlaybackFrame(
                    configuration=(1.0 - blend) * start + blend * end,
                )
            )
    frames.append(
        PlaybackFrame(
            configuration=configurations[-1].copy(),
        )
    )
    return frames


def add_status_text(keyframe_count: int) -> None:
    p.addUserDebugText(
        (
            f"{keyframe_count} keyframes | E: play/pause | R: restart"
        ),
        (-0.65, 0.0, 1.75),
        textColorRGB=(0.08, 0.08, 0.08),
        textSize=1.25,
    )


def run(headless: bool = False) -> None:
    keyframes = load_stick_keyframes()
    connection = p.connect(p.DIRECT) if headless else p.connect(
        p.GUI, options="--width=900 --height=700"
    )
    if connection < 0:
        raise SystemExit("Could not start PyBullet.")

    try:
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.resetSimulation()
        if not headless:
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        p.setGravity(0.0, 0.0, 0.0)
        p.loadURDF("plane.urdf")
        robot = p.loadURDF(
            str(ROBOT_URDF),
            basePosition=(0.0, 0.0, PELVIS_HEIGHT),
            useFixedBase=True,
        )
        indices = joint_indices(robot)
        missing = [name for name in POINT_LINK_JOINTS.values() if name not in indices]
        if missing:
            raise SystemExit(
                f"The {ROBOT_NAME} model is missing joints: {', '.join(missing)}"
            )

        initialize_robot(robot, indices)
        configurations, joints, base_orientation, errors = solve_motion(
            robot, keyframes, indices
        )
        point_links = {
            name: indices[joint_name]
            for name, joint_name in POINT_LINK_JOINTS.items()
        }
        foot_links = (
            point_links["left foot"],
            point_links["right foot"],
        )
        frames = interpolated_frames(configurations)
        apply_configuration(
            robot, frames[0].configuration, base_orientation, joints
        )
        prevent_floor_penetration(
            robot, foot_links, base_orientation
        )

        if headless:
            print(f"Prepared {len(frames)} interpolated frames.")
            print(f"Maximum weighted keyframe error: {max(errors):.4f}")
            return

        p.resetDebugVisualizerCamera(
            cameraDistance=2.3,
            cameraYaw=40.0,
            cameraPitch=-12.0,
            cameraTargetPosition=(0.0, 0.0, 0.85),
        )
        add_status_text(len(configurations))
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
        frame_index = 0
        playing = True
        while p.isConnected():
            keys = p.getKeyboardEvents()
            if ord("e") in keys and keys[ord("e")] & p.KEY_WAS_TRIGGERED:
                playing = not playing
            if ord("r") in keys and keys[ord("r")] & p.KEY_WAS_TRIGGERED:
                frame_index = 0
                playing = True

            if playing:
                apply_configuration(
                    robot,
                    frames[frame_index].configuration,
                    base_orientation,
                    joints,
                )
                prevent_floor_penetration(
                    robot, foot_links, base_orientation
                )
                frame_index += 1
                if frame_index >= len(frames):
                    frame_index = len(frames) - 1
                    playing = False
            time.sleep(1.0 / PLAYBACK_HZ)
    finally:
        if p.isConnected():
            p.disconnect()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--headless", action="store_true", help="Solve and validate without a GUI."
    )
    args = parser.parse_args()
    run(headless=args.headless)


if __name__ == "__main__":
    main()

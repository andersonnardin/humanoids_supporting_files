"""Plan and simulate an offline cart-table walk with the Unitree G1.

The complete desired ZMP sequence is defined from a footstep plan. A
tridiagonal cart-table solve produces the constant-height CoM reference before
the simulation starts. A collisionless fixed-base G1 is used only as an inverse
kinematics reference; the visible G1 remains free-floating and is driven by
joint motors while PyBullet resolves gravity, inertia, collision, friction, and
ground contact.

The green path is the desired ZMP, the blue path is the planned CoM, the red
trace is the measured multibody CoM, and the yellow marker is the contact-wrench
ZMP measured from the physical simulation.

Controls: E pauses/resumes; T stops the plan and holds the current joints;
R restarts the complete trial.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np

try:
    import pybullet as p
    import pybullet_data
except ImportError as error:  # pragma: no cover - learner environment check
    raise SystemExit("Install PyBullet first: python3 -m pip install pybullet") from error


SCRIPT_DIRECTORY = Path(__file__).resolve().parent
DESCRIPTION_DIRECTORY = SCRIPT_DIRECTORY / "g1_description"
DEFAULT_URDF = DESCRIPTION_DIRECTORY / "g1_29dof.urdf"
G = 9.81

LEFT = "left"
RIGHT = "right"
LEG_JOINT_NAMES = {
    LEFT: (
        "left_hip_pitch_joint",
        "left_hip_roll_joint",
        "left_hip_yaw_joint",
        "left_knee_joint",
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
    ),
    RIGHT: (
        "right_hip_pitch_joint",
        "right_hip_roll_joint",
        "right_hip_yaw_joint",
        "right_knee_joint",
        "right_ankle_pitch_joint",
        "right_ankle_roll_joint",
    ),
}
ANKLE_LINK_JOINT = {
    LEFT: "left_ankle_roll_joint",
    RIGHT: "right_ankle_roll_joint",
}


@dataclass(frozen=True)
class WalkingSettings:
    """Offline cart-table, G1 gait, and simulation settings."""

    com_height: float = 0.72
    nominal_pelvis_height: float = 0.79
    support_time: float = 0.90
    step_lengths: tuple[float, ...] = (0.0, 0.12, 0.12, 0.12, 0.0)
    step_widths: tuple[float, ...] = (0.237, 0.237, 0.237, 0.237, 0.237)
    foot_length: float = 0.18
    foot_width: float = 0.07
    foot_center_from_ankle_x: float = 0.035
    ankle_height: float = 0.036
    swing_height: float = 0.06
    samples_per_phase: int = 90
    simulation_time_step: float = 1.0 / 240.0


@dataclass(frozen=True)
class MotorSettings:
    """Position-controller gains and effort scaling."""

    leg_position_gain: float = 0.34
    upper_body_position_gain: float = 0.22
    effort_scale: float = 0.90
    minimum_force: float = 8.0


def nominal_foot_placements(settings: WalkingSettings) -> np.ndarray:
    """Return alternating right/left support-foot centers."""
    lengths = np.asarray(settings.step_lengths, dtype=float)
    widths = np.asarray(settings.step_widths, dtype=float)
    if len(lengths) != len(widths):
        raise ValueError("step_lengths and step_widths must have equal length.")

    feet = np.zeros((len(lengths) + 1, 2))
    feet[0] = (0.0, -widths[0] / 2.0)
    for step, (length, width) in enumerate(zip(lengths, widths), start=1):
        feet[step] = feet[step - 1] + (
            length,
            width if step % 2 else -width,
        )
    return feet


def desired_zmp_sequence(
    feet: np.ndarray, settings: WalkingSettings
) -> np.ndarray:
    """Hold the desired ZMP at the active support-foot center."""
    return np.repeat(feet, settings.samples_per_phase, axis=0)


def solve_offline_cart_table(
    desired_zmp: np.ndarray,
    settings: WalkingSettings,
    initial_velocity: np.ndarray | None = None,
    terminal_velocity: np.ndarray | None = None,
) -> np.ndarray:
    """Solve the discretized cart-table equation for x and y together."""
    sample_count = len(desired_zmp)
    dt = settings.support_time / settings.samples_per_phase
    alpha = settings.com_height / (G * dt**2)

    matrix = np.diag(np.full(sample_count, 1.0 + 2.0 * alpha))
    matrix += np.diag(np.full(sample_count - 1, -alpha), 1)
    matrix += np.diag(np.full(sample_count - 1, -alpha), -1)

    matrix[0, 0] = 1.0 + alpha
    matrix[-1, -1] = 1.0 + alpha
    initial_velocity = (
        np.zeros(2) if initial_velocity is None else np.asarray(initial_velocity)
    )
    terminal_velocity = (
        np.zeros(2) if terminal_velocity is None else np.asarray(terminal_velocity)
    )
    right_hand_side = desired_zmp.copy()
    right_hand_side[0] -= alpha * initial_velocity * dt
    right_hand_side[-1] += alpha * terminal_velocity * dt
    return np.linalg.solve(matrix, right_hand_side)


def plan_pattern(
    settings: WalkingSettings,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """Return feet, desired ZMP samples, and CoM support-phase segments."""
    feet = nominal_foot_placements(settings)
    desired_zmp = desired_zmp_sequence(feet, settings)
    planned_com = solve_offline_cart_table(desired_zmp, settings)
    trajectories = [
        planned_com[start : start + settings.samples_per_phase]
        for start in range(0, len(planned_com), settings.samples_per_phase)
    ]
    return feet, desired_zmp, trajectories


def joint_indices(robot: int) -> dict[str, int]:
    """Map URDF joint names to PyBullet joint indices."""
    return {
        p.getJointInfo(robot, joint)[1].decode("utf-8"): joint
        for joint in range(p.getNumJoints(robot))
    }


def movable_joint_indices(robot: int) -> list[int]:
    """Return joints represented in PyBullet's IK result vector."""
    return [
        joint
        for joint in range(p.getNumJoints(robot))
        if p.getJointInfo(robot, joint)[2] != p.JOINT_FIXED
    ]


def leg_joint_indices(indices: dict[str, int], side: str) -> list[int]:
    return [indices[name] for name in LEG_JOINT_NAMES[side]]


def initial_joint_targets(
    robot: int, indices: dict[str, int]
) -> dict[int, float]:
    """Return a slightly bent, symmetric G1 reference posture."""
    targets = {joint: 0.0 for joint in movable_joint_indices(robot)}
    for side in (LEFT, RIGHT):
        targets[indices[f"{side}_hip_pitch_joint"]] = -0.28
        targets[indices[f"{side}_knee_joint"]] = 0.56
        targets[indices[f"{side}_ankle_pitch_joint"]] = -0.28
    targets[indices["left_shoulder_roll_joint"]] = 0.18
    targets[indices["right_shoulder_roll_joint"]] = -0.18
    targets[indices["left_elbow_joint"]] = 0.35
    targets[indices["right_elbow_joint"]] = 0.35
    return targets


def center_of_mass(robot: int) -> np.ndarray:
    """Calculate the mass-weighted multibody CoM in world coordinates."""
    weighted_position = np.zeros(3)
    total_mass = 0.0
    for link in range(-1, p.getNumJoints(robot)):
        mass = p.getDynamicsInfo(robot, link)[0]
        if mass <= 0.0:
            continue
        position = (
            np.asarray(p.getBasePositionAndOrientation(robot)[0])
            if link == -1
            else np.asarray(
                p.getLinkState(robot, link, computeForwardKinematics=True)[0]
            )
        )
        weighted_position += mass * position
        total_mass += mass
    return weighted_position / total_mass


def configure_reference_model(robot: int) -> None:
    """Make the fixed-base IK reference collisionless and invisible."""
    for link in range(-1, p.getNumJoints(robot)):
        p.changeVisualShape(robot, link, rgbaColor=(1.0, 1.0, 1.0, 0.0))
        p.setCollisionFilterGroupMask(robot, link, 0, 0)


def disable_default_motors(robot: int) -> None:
    """Remove PyBullet's default velocity motors."""
    joints = movable_joint_indices(robot)
    p.setJointMotorControlArray(
        robot,
        joints,
        p.VELOCITY_CONTROL,
        targetVelocities=[0.0] * len(joints),
        forces=[0.0] * len(joints),
    )


def reset_reference_posture(
    robot: int, targets: dict[int, float]
) -> None:
    for joint, value in targets.items():
        p.resetJointState(robot, joint, value, 0.0)


def solve_leg_ik(
    reference_model: int,
    indices: dict[str, int],
    side: str,
    ankle_target: np.ndarray,
) -> np.ndarray:
    """Solve one six-joint G1 leg for a level ankle pose."""
    movable = movable_joint_indices(reference_model)
    lower = [p.getJointInfo(reference_model, joint)[8] for joint in movable]
    upper = [p.getJointInfo(reference_model, joint)[9] for joint in movable]
    ranges = [high - low for low, high in zip(lower, upper)]
    rest = [p.getJointState(reference_model, joint)[0] for joint in movable]
    end_effector = indices[ANKLE_LINK_JOINT[side]]

    solution = p.calculateInverseKinematics(
        reference_model,
        end_effector,
        ankle_target.tolist(),
        targetOrientation=p.getQuaternionFromEuler((0.0, 0.0, 0.0)),
        lowerLimits=lower,
        upperLimits=upper,
        jointRanges=ranges,
        restPoses=rest,
        maxNumIterations=300,
        residualThreshold=1.0e-5,
    )
    solution_by_joint = dict(zip(movable, solution))
    leg = leg_joint_indices(indices, side)
    targets = np.array([solution_by_joint[joint] for joint in leg])
    for joint, target in zip(leg, targets):
        p.resetJointState(reference_model, joint, float(target), 0.0)
    return targets


def desired_leg_postures(
    reference_model: int,
    indices: dict[str, int],
    pelvis_target: np.ndarray,
    right_ankle_target: np.ndarray,
    left_ankle_target: np.ndarray,
) -> dict[str, np.ndarray]:
    """Map desired pelvis and ankle poses into both G1 leg postures."""
    p.resetBasePositionAndOrientation(
        reference_model,
        pelvis_target.tolist(),
        (0.0, 0.0, 0.0, 1.0),
    )
    return {
        RIGHT: solve_leg_ik(
            reference_model, indices, RIGHT, right_ankle_target
        ),
        LEFT: solve_leg_ik(reference_model, indices, LEFT, left_ankle_target),
    }


def ankle_target(
    foot_center: np.ndarray, settings: WalkingSettings
) -> np.ndarray:
    """Convert a planned sole center into a G1 ankle-roll-link target."""
    return np.array(
        (
            foot_center[0] - settings.foot_center_from_ankle_x,
            foot_center[1],
            settings.ankle_height,
        )
    )


def swing_position(
    start: np.ndarray,
    finish: np.ndarray,
    fraction: float,
    settings: WalkingSettings,
) -> np.ndarray:
    """Return a straight horizontal transfer with sinusoidal foot clearance."""
    position = (1.0 - fraction) * start + fraction * finish
    position[2] = (
        settings.ankle_height
        + settings.swing_height * np.sin(np.pi * fraction)
    )
    return position


def reset_dynamic_robot(
    robot: int,
    indices: dict[str, int],
    base_position: np.ndarray,
    nominal_targets: dict[int, float],
    leg_targets: dict[str, np.ndarray],
) -> None:
    """Place the physical G1 at the start of the walking trial."""
    p.resetBasePositionAndOrientation(
        robot, base_position.tolist(), (0.0, 0.0, 0.0, 1.0)
    )
    p.resetBaseVelocity(robot, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    for joint, value in nominal_targets.items():
        p.resetJointState(robot, joint, value, 0.0)
    for side in (RIGHT, LEFT):
        for joint, value in zip(leg_joint_indices(indices, side), leg_targets[side]):
            p.resetJointState(robot, joint, float(value), 0.0)
    disable_default_motors(robot)


def command_robot(
    robot: int,
    indices: dict[str, int],
    nominal_targets: dict[int, float],
    desired_legs: dict[str, np.ndarray],
    motor_settings: MotorSettings,
) -> None:
    """Track the reference posture with effort-limited joint motors."""
    targets = nominal_targets.copy()
    leg_set: set[int] = set()
    for side in (RIGHT, LEFT):
        joints = leg_joint_indices(indices, side)
        leg_set.update(joints)
        targets.update(
            {joint: float(value) for joint, value in zip(joints, desired_legs[side])}
        )

    joints = movable_joint_indices(robot)
    positions = [targets[joint] for joint in joints]
    forces = [
        max(
            motor_settings.minimum_force,
            motor_settings.effort_scale * p.getJointInfo(robot, joint)[10],
        )
        for joint in joints
    ]
    gains = [
        motor_settings.leg_position_gain
        if joint in leg_set
        else motor_settings.upper_body_position_gain
        for joint in joints
    ]
    p.setJointMotorControlArray(
        robot,
        joints,
        p.POSITION_CONTROL,
        targetPositions=positions,
        forces=forces,
        positionGains=gains,
    )


def hold_current_joint_state(robot: int, targets: list[float]) -> None:
    """Use the motors as brakes after T is pressed."""
    joints = movable_joint_indices(robot)
    forces = [max(8.0, 0.9 * p.getJointInfo(robot, joint)[10]) for joint in joints]
    p.setJointMotorControlArray(
        robot,
        joints,
        p.POSITION_CONTROL,
        targetPositions=targets,
        forces=forces,
        positionGains=[0.38] * len(joints),
    )


def contact_wrench_zmp(
    robot: int, plane: int
) -> tuple[np.ndarray | None, float]:
    """Estimate the ground-plane ZMP from PyBullet contact forces."""
    total_force = np.zeros(3)
    total_moment = np.zeros(3)
    for contact in p.getContactPoints(robot, plane):
        position = np.asarray(contact[6])
        normal_force = contact[9] * np.asarray(contact[7])
        friction_one = contact[10] * np.asarray(contact[11])
        friction_two = contact[12] * np.asarray(contact[13])
        force = normal_force + friction_one + friction_two
        total_force += force
        total_moment += np.cross(position, force)

    if total_force[2] <= 1.0e-6:
        return None, 0.0
    zmp = np.array(
        (
            -total_moment[1] / total_force[2],
            total_moment[0] / total_force[2],
        )
    )
    return zmp, float(total_force[2])


def draw_rectangle(center: np.ndarray, settings: WalkingSettings) -> None:
    half_length = settings.foot_length / 2.0
    half_width = settings.foot_width / 2.0
    corners = [
        (center[0] - half_length, center[1] - half_width, 0.003),
        (center[0] + half_length, center[1] - half_width, 0.003),
        (center[0] + half_length, center[1] + half_width, 0.003),
        (center[0] - half_length, center[1] + half_width, 0.003),
    ]
    for first, second in zip(corners, corners[1:] + corners[:1]):
        p.addUserDebugLine(first, second, (0.85, 0.15, 0.15), lineWidth=1.4)


def add_plan_debug_geometry(
    feet: np.ndarray,
    desired_zmp: np.ndarray,
    trajectories: list[np.ndarray],
    settings: WalkingSettings,
) -> None:
    """Draw requested feet, desired ZMP, and planned CoM."""
    for foot in feet:
        draw_rectangle(foot, settings)

    support_centers = desired_zmp[:: settings.samples_per_phase]
    for first, second in zip(support_centers[:-1], support_centers[1:]):
        p.addUserDebugLine(
            (first[0], first[1], 0.008),
            (second[0], second[1], 0.008),
            (0.10, 0.60, 0.20),
            lineWidth=2.2,
        )
    for point in support_centers:
        p.addUserDebugText(
            "x", (point[0], point[1], 0.015), (0.10, 0.60, 0.20), textSize=1.0
        )

    planned_com = np.vstack(trajectories)
    for first, second in zip(planned_com[:-1], planned_com[1:]):
        p.addUserDebugLine(
            (first[0], first[1], settings.com_height),
            (second[0], second[1], settings.com_height),
            (0.05, 0.35, 0.90),
            lineWidth=1.8,
        )


def create_zmp_marker() -> int:
    visual = p.createVisualShape(
        p.GEOM_SPHERE,
        radius=0.018,
        rgbaColor=(1.0, 0.78, 0.05, 1.0),
    )
    return p.createMultiBody(
        baseMass=0.0,
        baseVisualShapeIndex=visual,
        basePosition=(0.0, 0.0, -1.0),
    )


def key_was_pressed(keys: dict[int, int], key: str) -> bool:
    return bool(keys.get(ord(key), 0) & p.KEY_WAS_TRIGGERED)


def run(
    gui: bool = True,
    urdf_path: Path = DEFAULT_URDF,
    diagnostics: bool = False,
) -> None:
    """Run the cart-table-to-G1 physical walking experiment."""
    urdf_path = urdf_path.resolve()
    if not urdf_path.is_file():
        raise FileNotFoundError(
            f"Missing G1 URDF: {urdf_path}\n"
            "Place g1_description beside this script before running it."
        )

    settings = WalkingSettings()
    motor_settings = MotorSettings()
    feet, desired_zmp, trajectories = plan_pattern(settings)

    client = (
        p.connect(p.GUI, options="--width=900 --height=700")
        if gui
        else p.connect(p.DIRECT)
    )
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0.0, 0.0, -G)
    p.setTimeStep(settings.simulation_time_step)
    p.setRealTimeSimulation(0)
    p.setPhysicsEngineParameter(numSolverIterations=180, enableConeFriction=1)
    if gui:
        p.resetDebugVisualizerCamera(2.2, 48.0, -22.0, (0.25, 0.0, 0.65))

    plane = p.loadURDF("plane.urdf")
    flags = p.URDF_USE_INERTIA_FROM_FILE | p.URDF_MAINTAIN_LINK_ORDER
    robot = p.loadURDF(
        str(urdf_path),
        basePosition=(0.0, 0.0, settings.nominal_pelvis_height),
        useFixedBase=False,
        flags=flags,
    )
    reference_model = p.loadURDF(
        str(urdf_path),
        basePosition=(0.0, 0.0, settings.nominal_pelvis_height),
        useFixedBase=True,
        flags=flags,
    )
    configure_reference_model(reference_model)
    indices = joint_indices(robot)
    reference_indices = joint_indices(reference_model)
    nominal_targets = initial_joint_targets(robot, indices)
    reference_targets = initial_joint_targets(reference_model, reference_indices)
    reset_reference_posture(reference_model, reference_targets)

    for link in range(-1, p.getNumJoints(robot)):
        p.changeDynamics(
            robot,
            link,
            lateralFriction=1.2,
            spinningFriction=0.05,
            rollingFriction=0.01,
            linearDamping=0.02,
            angularDamping=0.02,
        )
    p.changeDynamics(plane, -1, lateralFriction=1.2)

    initial_com_target = np.array(
        (trajectories[0][0, 0], trajectories[0][0, 1], settings.com_height)
    )
    initial_right = ankle_target(feet[0], settings)
    initial_left = ankle_target(feet[1], settings)
    initial_pelvis_guess = np.array(
        (
            initial_com_target[0],
            initial_com_target[1],
            settings.nominal_pelvis_height,
        )
    )
    initial_postures = desired_leg_postures(
        reference_model,
        reference_indices,
        initial_pelvis_guess,
        initial_right,
        initial_left,
    )
    nominal_com_offset = center_of_mass(reference_model) - initial_pelvis_guess
    initial_pelvis = initial_com_target - nominal_com_offset
    initial_postures = desired_leg_postures(
        reference_model,
        reference_indices,
        initial_pelvis,
        initial_right,
        initial_left,
    )

    try:
        while p.isConnected(client):
            reset_dynamic_robot(
                robot,
                indices,
                initial_pelvis,
                nominal_targets,
                initial_postures,
            )
            right_ankle = initial_right.copy()
            left_ankle = initial_left.copy()
            elapsed_time = 0.0
            paused = False
            joints_locked = False
            locked_targets: list[float] | None = None
            restart_requested = False
            previous_com = center_of_mass(robot)

            if gui:
                p.removeAllUserDebugItems()
                add_plan_debug_geometry(feet, desired_zmp, trajectories, settings)
                zmp_marker = create_zmp_marker()
                zmp_error_line = p.addUserDebugLine(
                    (0.0, 0.0, -1.0),
                    (0.0, 0.0, -1.0),
                    (1.0, 0.65, 0.0),
                    lineWidth=2.0,
                )
                status_id = p.addUserDebugText(
                    "", (0.0, 0.0, 1.35), (0.05, 0.05, 0.05), textSize=1.15
                )
            else:
                zmp_marker = -1
                zmp_error_line = -1
                status_id = -1

            for phase, segment in enumerate(trajectories):
                support_side = RIGHT if phase % 2 == 0 else LEFT
                swing_side = LEFT if support_side == RIGHT else RIGHT
                support_ankle = ankle_target(feet[phase], settings)
                landing_ankle = ankle_target(
                    feet[min(phase + 1, len(feet) - 1)], settings
                )
                swing_start = (
                    left_ankle.copy() if swing_side == LEFT else right_ankle.copy()
                )

                for frame, planned_xy in enumerate(segment):
                    fraction = frame / (len(segment) - 1)
                    swing_ankle = swing_position(
                        swing_start, landing_ankle, fraction, settings
                    )
                    planned_com = np.array(
                        (planned_xy[0], planned_xy[1], settings.com_height)
                    )
                    pelvis_target = planned_com - nominal_com_offset
                    right_target = (
                        support_ankle if support_side == RIGHT else swing_ankle
                    )
                    left_target = (
                        support_ankle if support_side == LEFT else swing_ankle
                    )
                    desired_legs = desired_leg_postures(
                        reference_model,
                        reference_indices,
                        pelvis_target,
                        right_target,
                        left_target,
                    )

                    frame_duration = settings.support_time / len(segment)
                    substeps = max(
                        1, round(frame_duration / settings.simulation_time_step)
                    )
                    completed_substeps = 0
                    desired_point = desired_zmp[
                        phase * settings.samples_per_phase + frame
                    ]

                    while completed_substeps < substeps:
                        keys = p.getKeyboardEvents() if gui else {}
                        if gui and key_was_pressed(keys, "r"):
                            restart_requested = True
                            break
                        if gui and key_was_pressed(keys, "e"):
                            paused = not paused
                        if gui and key_was_pressed(keys, "t") and not joints_locked:
                            joints_locked = True
                            locked_targets = [
                                p.getJointState(robot, joint)[0]
                                for joint in movable_joint_indices(robot)
                            ]
                        if paused:
                            time.sleep(settings.simulation_time_step)
                            continue

                        if joints_locked:
                            hold_current_joint_state(robot, locked_targets)
                        else:
                            command_robot(
                                robot,
                                indices,
                                nominal_targets,
                                desired_legs,
                                motor_settings,
                            )
                        p.stepSimulation()
                        if not joints_locked:
                            completed_substeps += 1
                        elapsed_time += settings.simulation_time_step

                        measured_com = center_of_mass(robot)
                        realized_zmp, vertical_force = contact_wrench_zmp(robot, plane)
                        if diagnostics and completed_substeps == 1 and frame % 15 == 0:
                            print(
                                f"t={elapsed_time:.2f} "
                                f"CoM={np.round(measured_com, 3)} "
                                f"ZMP={None if realized_zmp is None else np.round(realized_zmp, 3)} "
                                f"Fz={vertical_force:.1f}"
                            )

                        if gui:
                            p.addUserDebugLine(
                                previous_com,
                                measured_com,
                                (0.95, 0.12, 0.08),
                                lineWidth=2.4,
                            )
                            if realized_zmp is not None:
                                p.resetBasePositionAndOrientation(
                                    zmp_marker,
                                    (realized_zmp[0], realized_zmp[1], 0.022),
                                    (0.0, 0.0, 0.0, 1.0),
                                )
                                p.addUserDebugLine(
                                    (desired_point[0], desired_point[1], 0.014),
                                    (realized_zmp[0], realized_zmp[1], 0.014),
                                    (1.0, 0.65, 0.0),
                                    lineWidth=2.0,
                                    replaceItemUniqueId=zmp_error_line,
                                )
                                zmp_error = float(
                                    np.linalg.norm(realized_zmp - desired_point)
                                )
                            else:
                                zmp_error = float("nan")
                            state = "JOINTS HELD" if joints_locked else "RUNNING"
                            p.addUserDebugText(
                                (
                                    f"{elapsed_time:.2f} s | {state} | "
                                    f"Fz={vertical_force:.1f} N | "
                                    f"ZMP error={zmp_error:.3f} m"
                                ),
                                (measured_com + np.array((-0.35, 0.0, 0.48))).tolist(),
                                (0.05, 0.05, 0.05),
                                textSize=1.05,
                                replaceItemUniqueId=status_id,
                            )
                            time.sleep(settings.simulation_time_step)
                        previous_com = measured_com

                    if restart_requested:
                        break
                if restart_requested:
                    break

                if swing_side == RIGHT:
                    right_ankle = landing_ankle.copy()
                else:
                    left_ankle = landing_ankle.copy()

            if not gui:
                break
            if restart_requested:
                continue
            while p.isConnected(client):
                keys = p.getKeyboardEvents()
                if key_was_pressed(keys, "r"):
                    break
                time.sleep(settings.simulation_time_step)
    finally:
        if p.isConnected(client):
            p.disconnect(client)


def main() -> None:
    run(gui=True)


if __name__ == "__main__":
    main()

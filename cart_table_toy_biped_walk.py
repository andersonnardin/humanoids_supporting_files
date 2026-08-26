"""Plan and simulate an offline cart-table walk with a physical toy biped.

The complete desired ZMP sequence is defined from the support-foot plan.
The cart-table equation is then solved offline as one tridiagonal system for
the horizontal CoM trajectory. PyBullet tracks the resulting pelvis and foot
targets using inverse kinematics, joint motors, gravity, and contact dynamics.

Controls: E pauses/resumes; T turns off walking and locks the current joint
state; R restarts the full walking trial.
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
G = 9.81
FOOT_CENTER_FROM_ANKLE = np.array((0.05, 0.0, -0.04))


@dataclass(frozen=True)
class WalkingSettings:
    """Offline cart-table, gait, foot, and PyBullet settings."""

    com_height: float = 0.88
    body_com_offset: float = 0.30
    support_time: float = 0.75
    step_lengths: tuple[float, ...] = (0.0, 0.3, 0.3, 0.3, 0.0)
    step_widths: tuple[float, ...] = (0.2, 0.2, 0.2, 0.2, 0.2)
    foot_length: float = 0.20
    foot_width: float = 0.10
    foot_height: float = 0.05
    swing_height: float = 0.08
    samples_per_phase: int = 90
    simulation_time_step: float = 1.0 / 240.0

    @property
    def pelvis_height(self) -> float:
        return self.com_height - self.body_com_offset


def nominal_foot_placements(settings: WalkingSettings) -> np.ndarray:
    lengths, widths = np.asarray(settings.step_lengths), np.asarray(settings.step_widths)
    if len(lengths) != len(widths):
        raise ValueError("step_lengths and step_widths must have equal length.")
    feet = np.zeros((len(lengths) + 1, 2))
    feet[0] = (0.0, -widths[0] / 2.0)
    for step, (length, width) in enumerate(zip(lengths, widths), start=1):
        feet[step] = feet[step - 1] + (length, width if step % 2 else -width)
    return feet


def desired_zmp_sequence(
    feet: np.ndarray, settings: WalkingSettings
) -> np.ndarray:
    """Hold the desired ZMP at the active support-foot center in each phase."""
    return np.repeat(feet, settings.samples_per_phase, axis=0)


def solve_offline_cart_table(
    desired_zmp: np.ndarray,
    settings: WalkingSettings,
    initial_velocity: np.ndarray | None = None,
    terminal_velocity: np.ndarray | None = None,
) -> np.ndarray:
    """Solve the discretized cart-table equation for the complete CoM path."""
    sample_count = len(desired_zmp)
    dt = settings.support_time / settings.samples_per_phase
    alpha = settings.com_height / (G * dt**2)

    # p_i = -alpha*x_(i-1) + (1+2*alpha)*x_i - alpha*x_(i+1)
    matrix = np.diag(np.full(sample_count, 1.0 + 2.0 * alpha))
    matrix += np.diag(np.full(sample_count - 1, -alpha), 1)
    matrix += np.diag(np.full(sample_count - 1, -alpha), -1)

    # Replace the unavailable samples outside the plan with specified endpoint
    # velocities. Both are zero by default, producing a rest-to-rest plan.
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

    # One factorization solves x and y together because both use the same A.
    return np.linalg.solve(matrix, right_hand_side)


def plan_pattern(
    settings: WalkingSettings,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[np.ndarray]]:
    """Return feet, support centers, desired ZMP, and offline CoM segments."""
    feet = nominal_foot_placements(settings)
    desired_zmp = desired_zmp_sequence(feet, settings)
    planned_com = solve_offline_cart_table(desired_zmp, settings)
    trajectories = [
        planned_com[start : start + settings.samples_per_phase]
        for start in range(0, len(planned_com), settings.samples_per_phase)
    ]
    return feet, feet.copy(), desired_zmp, trajectories


def joint_indices(robot: int) -> dict[str, int]:
    return {p.getJointInfo(robot, index)[1].decode("utf-8"): index for index in range(p.getNumJoints(robot))}


def leg_indices(indices: dict[str, int], side: str) -> list[int]:
    return [indices[f"{side}LEG_J{joint}"] for joint in range(6)]


def swing_position(start: np.ndarray, finish: np.ndarray, fraction: float, settings: WalkingSettings) -> np.ndarray:
    position = (1.0 - fraction) * start + fraction * finish
    position[2] = settings.foot_height + settings.swing_height * np.sin(np.pi * fraction)
    return position


def add_plan_debug_geometry(
    desired_feet: np.ndarray,
    desired_zmp: np.ndarray,
    trajectories: list[np.ndarray],
    settings: WalkingSettings,
) -> None:
    """Draw foot rectangles, desired ZMP, and the planned offline CoM path."""
    half_length, half_width = settings.foot_length / 2.0, settings.foot_width / 2.0
    for foot in desired_feet:
        corners = [(foot[0] - half_length, foot[1] - half_width, .002), (foot[0] + half_length, foot[1] - half_width, .002), (foot[0] + half_length, foot[1] + half_width, .002), (foot[0] - half_length, foot[1] + half_width, .002)]
        for first, second in zip(corners, corners[1:] + corners[:1]):
            p.addUserDebugLine(first, second, (0.85, 0.15, 0.15), lineWidth=1.4)

    zmp_height = 0.006
    support_centers = desired_zmp[:: settings.samples_per_phase]
    for first, second in zip(support_centers[:-1], support_centers[1:]):
        p.addUserDebugLine(
            (first[0], first[1], zmp_height),
            (second[0], second[1], zmp_height),
            (0.10, 0.55, 0.18),
            lineWidth=2.0,
        )
    for point in support_centers:
        p.addUserDebugText(
            "x",
            (point[0], point[1], 0.01),
            (0.10, 0.55, 0.18),
            textSize=1.0,
        )

    planned_com = np.vstack(trajectories)
    for first, second in zip(planned_com[:-1], planned_com[1:]):
        p.addUserDebugLine(
            (first[0], first[1], settings.com_height),
            (second[0], second[1], settings.com_height),
            (0.05, 0.35, 0.90),
            lineWidth=1.8,
        )


def total_mass(robot: int) -> float:
    return sum(p.getDynamicsInfo(robot, link)[0] for link in range(-1, p.getNumJoints(robot)))


def center_of_mass(robot: int) -> np.ndarray:
    weighted_position, mass_sum = np.zeros(3), 0.0
    for link in range(-1, p.getNumJoints(robot)):
        mass = p.getDynamicsInfo(robot, link)[0]
        if mass <= 0.0:
            continue
        position = np.asarray(p.getBasePositionAndOrientation(robot)[0]) if link == -1 else np.asarray(p.getLinkState(robot, link, computeForwardKinematics=True)[0])
        weighted_position += mass * position
        mass_sum += mass
    return weighted_position / mass_sum


@dataclass(frozen=True)
class RobotModel:
    """Robot-specific names and geometry used by the controller."""

    urdf_path: Path
    right_side: str = "R"
    left_side: str = "L"


@dataclass(frozen=True)
class MotorSettings:
    """Joint-motor limits used to track the cart-table-derived IK posture."""

    support_force_limit: float = 120.0
    swing_force_limit: float = 70.0
    support_position_gain: float = 0.28
    swing_position_gain: float = 0.32


TOY_BIPED = RobotModel(SCRIPT_DIRECTORY / "toy_biped_physics.urdf")


def key_was_pressed(keys: dict[int, int], key: str) -> bool:
    """Return true once for a lowercase key press."""
    return bool(keys.get(ord(key), 0) & p.KEY_WAS_TRIGGERED)


def ankle_target(foot_center: np.ndarray, settings: WalkingSettings) -> np.ndarray:
    """Convert a planned sole-center position into the URDF ankle target."""
    contact_preload = 0.001
    return np.array(
        (
            foot_center[0] - FOOT_CENTER_FROM_ANKLE[0],
            foot_center[1],
            settings.foot_height - contact_preload,
        )
    )


def configure_reference_model(robot: int) -> None:
    """Keep the fixed-base kinematic reference out of the physical scene."""
    for link in range(-1, p.getNumJoints(robot)):
        p.changeVisualShape(robot, link, rgbaColor=(1.0, 1.0, 1.0, 0.0))
        p.setCollisionFilterGroupMask(robot, link, 0, 0)


def disable_default_motors(robot: int) -> None:
    """Disable URDF velocity motors so every actuator uses torque control."""
    joints = list(range(p.getNumJoints(robot)))
    p.setJointMotorControlArray(
        robot,
        joints,
        p.VELOCITY_CONTROL,
        targetVelocities=[0.0] * len(joints),
        forces=[0.0] * len(joints),
    )


def solve_ik_leg(
    reference_model: int,
    reference_indices: dict[str, int],
    side: str,
    target: np.ndarray,
) -> np.ndarray:
    """Solve one leg of the reference model for a level-foot pose."""
    end_effector = reference_indices[f"{side}LEG_J5"]
    joint_count = p.getNumJoints(reference_model)
    lower_limits = [p.getJointInfo(reference_model, joint)[8] for joint in range(joint_count)]
    upper_limits = [p.getJointInfo(reference_model, joint)[9] for joint in range(joint_count)]
    joint_ranges = [upper - lower for lower, upper in zip(lower_limits, upper_limits)]
    rest_poses = [p.getJointState(reference_model, joint)[0] for joint in range(joint_count)]

    # A perfectly straight leg is an IK singularity. Seed a human-like bent
    # knee only when this leg has not yet acquired a valid previous solution.
    knee = reference_indices[f"{side}LEG_J3"]
    if abs(rest_poses[knee]) < 0.10:
        rest_poses[reference_indices[f"{side}LEG_J2"]] = -0.45
        rest_poses[knee] = 0.90
        rest_poses[reference_indices[f"{side}LEG_J4"]] = -0.45
        for joint in leg_indices(reference_indices, side):
            p.resetJointState(reference_model, joint, rest_poses[joint])

    solution = p.calculateInverseKinematics(
        reference_model,
        end_effector,
        target.tolist(),
        targetOrientation=p.getQuaternionFromEuler((0.0, 0.0, 0.0)),
        lowerLimits=lower_limits,
        upperLimits=upper_limits,
        jointRanges=joint_ranges,
        restPoses=rest_poses,
        maxNumIterations=300,
        residualThreshold=1.0e-6,
    )
    joints = leg_indices(reference_indices, side)
    targets = np.array([solution[joint] for joint in joints])
    for joint, target_position in zip(joints, targets):
        p.resetJointState(reference_model, joint, float(target_position))
    return targets


def desired_leg_postures(
    reference_model: int,
    reference_indices: dict[str, int],
    pelvis_target: np.ndarray,
    right_ankle_target: np.ndarray,
    left_ankle_target: np.ndarray,
) -> dict[str, np.ndarray]:
    """Map desired pelvis and feet into right/left leg joint targets."""
    p.resetBasePositionAndOrientation(
        reference_model, pelvis_target.tolist(), (0.0, 0.0, 0.0, 1.0)
    )
    return {
        "R": solve_ik_leg(reference_model, reference_indices, "R", right_ankle_target),
        "L": solve_ik_leg(reference_model, reference_indices, "L", left_ankle_target),
    }


def reset_dynamic_robot(
    robot: int,
    indices: dict[str, int],
    base_position: np.ndarray,
    leg_targets: dict[str, np.ndarray],
) -> None:
    """Place the dynamic robot in the initial multibody configuration."""
    p.resetBasePositionAndOrientation(
        robot, base_position.tolist(), (0.0, 0.0, 0.0, 1.0)
    )
    p.resetBaseVelocity(robot, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    for side in ("R", "L"):
        for joint, target in zip(leg_indices(indices, side), leg_targets[side]):
            p.resetJointState(robot, joint, float(target), 0.0)
    disable_default_motors(robot)


def command_leg_postures(
    robot: int,
    indices: dict[str, int],
    support_side: str,
    desired_postures: dict[str, np.ndarray],
    motor_settings: MotorSettings,
) -> None:
    """Track the planned support and swing leg postures with joint motors."""
    swing_side = "L" if support_side == "R" else "R"
    support_joints = leg_indices(indices, support_side)
    swing_joints = leg_indices(indices, swing_side)
    p.setJointMotorControlArray(
        robot,
        support_joints,
        p.POSITION_CONTROL,
        targetPositions=desired_postures[support_side].tolist(),
        forces=[motor_settings.support_force_limit] * len(support_joints),
        positionGains=[motor_settings.support_position_gain] * len(support_joints),
    )
    p.setJointMotorControlArray(
        robot,
        swing_joints,
        p.POSITION_CONTROL,
        targetPositions=desired_postures[swing_side].tolist(),
        forces=[motor_settings.swing_force_limit] * len(swing_joints),
        positionGains=[motor_settings.swing_position_gain] * len(swing_joints),
    )


def captured_joint_positions(robot: int) -> list[float]:
    """Return the current position of every actuated joint."""
    return [
        p.getJointState(robot, joint)[0]
        for joint in range(p.getNumJoints(robot))
    ]


def hold_current_joint_state(robot: int, targets: list[float]) -> None:
    """Use joint motors as brakes at the pose captured when T was pressed."""
    joints = list(range(p.getNumJoints(robot)))
    p.setJointMotorControlArray(
        robot,
        joints,
        p.POSITION_CONTROL,
        targetPositions=targets,
        forces=[140.0] * len(joints),
        positionGains=[0.45] * len(joints),
    )


def measured_vertical_ground_force(robot: int, plane: int) -> float:
    """Sum PyBullet's normal contact forces under the physical robot."""
    return sum(contact[9] for contact in p.getContactPoints(robot, plane))


def active_sole_position(
    robot: int, indices: dict[str, int], side: str
) -> np.ndarray:
    """Return the active sole's measured world position, without constraining it."""
    link_state = p.getLinkState(
        robot,
        indices[f"{side}LEG_J5"],
        computeForwardKinematics=True,
    )
    sole_local = FOOT_CENTER_FROM_ANKLE + np.array((0.0, 0.0, -0.01))
    sole_world, _ = p.multiplyTransforms(
        link_state[4],
        link_state[5],
        sole_local.tolist(),
        (0.0, 0.0, 0.0, 1.0),
    )
    return np.asarray(sole_world, dtype=float)


def add_status(
    timer_id: int,
    elapsed_time: float,
    measured_reaction_z: float,
    paused: bool,
    joints_locked: bool = False,
) -> int:
    """Update the simulation time and measured vertical contact force."""
    state = " (paused)" if paused else ""
    if joints_locked:
        state += " (walking off, joints locked)"
    return p.addUserDebugText(
        f"time: {elapsed_time:.2f} s{state} | ground reaction z: {measured_reaction_z:.1f} N",
        (-0.08, -0.48, 0.84),
        (0.08, 0.08, 0.08),
        textSize=1.05,
        replaceItemUniqueId=timer_id,
    )


def run_walk(
    *,
    gui: bool = True,
    model: RobotModel = TOY_BIPED,
    diagnostics: bool = False,
) -> np.ndarray:
    """Run the offline cart-table-to-multibody walking experiment."""
    if not model.urdf_path.is_file():
        raise FileNotFoundError(f"Missing robot model: {model.urdf_path}")

    settings = WalkingSettings()
    motor_settings = MotorSettings()
    desired_feet, supports, desired_zmp, trajectories = plan_pattern(settings)
    client = (
        p.connect(p.GUI, options="--width=900 --height=700")
        if gui
        else p.connect(p.DIRECT)
    )
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0.0, 0.0, -G)
    p.setTimeStep(settings.simulation_time_step)
    p.setRealTimeSimulation(0)
    p.setPhysicsEngineParameter(numSolverIterations=160, enableConeFriction=1)
    if gui:
        p.resetDebugVisualizerCamera(1.70, 45.0, -24.0, (0.42, 0.0, 0.42))

    plane = p.loadURDF("plane.urdf")
    robot = p.loadURDF(
        str(model.urdf_path),
        basePosition=(supports[0, 0], supports[0, 1], settings.pelvis_height),
        useFixedBase=False,
    )
    reference_model = p.loadURDF(str(model.urdf_path), useFixedBase=True)
    configure_reference_model(reference_model)
    indices = joint_indices(robot)
    reference_indices = joint_indices(reference_model)

    for link in range(-1, p.getNumJoints(robot)):
        p.changeDynamics(
            robot,
            link,
            lateralFriction=1.3,
            spinningFriction=0.08,
            rollingFriction=0.02,
            linearDamping=0.0,
            angularDamping=0.0,
        )
    p.changeDynamics(plane, -1, lateralFriction=1.3)

    initial_com_target = np.array(
        (trajectories[0][0, 0], trajectories[0][0, 1], settings.com_height)
    )
    initial_right = ankle_target(supports[0], settings)
    initial_left = ankle_target(desired_feet[1], settings)

    # Establish the nominal multibody CoM-to-pelvis offset once, as assumed by
    # the simplified-model-to-multibody conversion used by this walking method.
    initial_pelvis_guess = initial_com_target.copy()
    initial_postures = desired_leg_postures(
        reference_model,
        reference_indices,
        initial_pelvis_guess,
        initial_right,
        initial_left,
    )
    reset_dynamic_robot(robot, indices, initial_pelvis_guess, initial_postures)
    nominal_com_offset = center_of_mass(robot) - initial_pelvis_guess
    initial_pelvis = initial_com_target - nominal_com_offset
    initial_postures = desired_leg_postures(
        reference_model,
        reference_indices,
        initial_pelvis,
        initial_right,
        initial_left,
    )

    while p.isConnected(client):
        reset_dynamic_robot(robot, indices, initial_pelvis, initial_postures)
        right_ankle = initial_right.copy()
        left_ankle = initial_left.copy()
        elapsed_time = 0.0
        paused = False
        joints_locked = False
        locked_joint_targets: list[float] | None = None
        restart_requested = False
        previous_com = center_of_mass(robot)

        if gui:
            p.removeAllUserDebugItems()
            add_plan_debug_geometry(
                desired_feet,
                desired_zmp,
                trajectories,
                settings,
            )
            status_id = add_status(-1, 0.0, 0.0, False)
        else:
            status_id = -1

        for phase, segment in enumerate(trajectories):
            support_side = "R" if phase % 2 == 0 else "L"
            swing_side = "L" if support_side == "R" else "R"
            support_ankle = ankle_target(supports[phase], settings)
            landing_ankle = ankle_target(
                supports[min(phase + 1, len(supports) - 1)], settings
            )
            swing_start = left_ankle.copy() if swing_side == "L" else right_ankle.copy()

            for frame, planned_xy in enumerate(segment):
                fraction = frame / (len(segment) - 1)
                swing_ankle = swing_position(
                    swing_start, landing_ankle, fraction, settings
                )
                planned_com = np.array(
                    (planned_xy[0], planned_xy[1], settings.com_height)
                )
                pelvis_target = planned_com - nominal_com_offset
                right_target = support_ankle if support_side == "R" else swing_ankle
                left_target = support_ankle if support_side == "L" else swing_ankle
                desired_postures = desired_leg_postures(
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
                while completed_substeps < substeps:
                    keys = p.getKeyboardEvents() if gui else {}
                    if gui and key_was_pressed(keys, "r"):
                        restart_requested = True
                        break
                    if gui and key_was_pressed(keys, "e"):
                        paused = not paused
                    if gui and key_was_pressed(keys, "t") and not joints_locked:
                        joints_locked = True
                        locked_joint_targets = captured_joint_positions(robot)
                    if paused:
                        status_id = add_status(
                            status_id,
                            elapsed_time,
                            measured_vertical_ground_force(robot, plane),
                            True,
                        )
                        time.sleep(settings.simulation_time_step)
                        continue

                    if joints_locked:
                        hold_current_joint_state(robot, locked_joint_targets)
                        p.stepSimulation()
                        elapsed_time += settings.simulation_time_step
                        measured_com = center_of_mass(robot)
                        if gui:
                            p.addUserDebugLine(
                                previous_com,
                                measured_com,
                                (0.95, 0.12, 0.08),
                                lineWidth=2.4,
                            )
                            status_id = add_status(
                                status_id,
                                elapsed_time,
                                measured_vertical_ground_force(robot, plane),
                                False,
                                True,
                            )
                            time.sleep(settings.simulation_time_step)
                        previous_com = measured_com
                        continue

                    command_leg_postures(
                        robot,
                        indices,
                        support_side,
                        desired_postures,
                        motor_settings,
                    )
                    p.stepSimulation()
                    completed_substeps += 1
                    elapsed_time += settings.simulation_time_step

                    measured_com = center_of_mass(robot)
                    if diagnostics and (frame < 8 or frame % 15 == 0) and completed_substeps == 1:
                        roll, pitch, yaw = p.getEulerFromQuaternion(
                            p.getBasePositionAndOrientation(robot)[1]
                        )
                        print(
                            f"t={elapsed_time:.3f} "
                            f"com={np.round(measured_com, 3)} "
                            f"GRFz={measured_vertical_ground_force(robot, plane):.1f} "
                            f"rpy={np.round((roll, pitch, yaw), 2)}"
                        )
                    if gui:
                        p.addUserDebugLine(
                            previous_com,
                            measured_com,
                            (0.95, 0.12, 0.08),
                            lineWidth=2.4,
                        )
                        status_id = add_status(
                            status_id,
                            elapsed_time,
                            measured_vertical_ground_force(robot, plane),
                            False,
                        )
                        time.sleep(settings.simulation_time_step)
                    previous_com = measured_com

                if restart_requested:
                    break
            if restart_requested:
                break

            if swing_side == "R":
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
        if not p.isConnected(client):
            break

    final_com = center_of_mass(robot)
    if p.isConnected(client):
        p.disconnect(client)
    return final_com


if __name__ == "__main__":
    run_walk()

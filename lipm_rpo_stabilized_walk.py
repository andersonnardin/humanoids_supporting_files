"""Test RPO LIPM walking with a simple support-ankle stabilizer.
Controls: E pauses/resumes; T turns off LIPM walking and locks the current
joint state; R restarts the full walking trial.
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
RPO_LEG_JOINTS = {
    "R": ("right_thigh_yaw_joint", "right_thigh_roll_joint", "right_thigh_pitch_joint", "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint"),
    "L": ("left_thigh_yaw_joint", "left_thigh_roll_joint", "left_thigh_pitch_joint", "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint"),
}


@dataclass(frozen=True)
class WalkingSettings:
    """LIPM, gait, foot, and PyBullet settings used by this standalone file."""

    com_height: float = 0.62
    body_com_offset: float = -0.075
    support_time: float = 0.75
    position_weight: float = 10.0
    velocity_weight: float = 1.0
    step_lengths: tuple[float, ...] = (0.0, 0.3, 0.3, 0.3, 0.0)
    step_widths: tuple[float, ...] = (0.2, 0.2, 0.2, 0.2, 0.2)
    foot_length: float = 0.20
    foot_width: float = 0.10
    foot_height: float = 0.05
    swing_height: float = 0.08
    samples_per_phase: int = 90
    simulation_time_step: float = 1.0 / 240.0

    @property
    def time_constant(self) -> float:
        return np.sqrt(self.com_height / G)

    @property
    def pelvis_height(self) -> float:
        return self.com_height - self.body_com_offset

    @property
    def transition_terms(self) -> tuple[float, float]:
        u = self.support_time / self.time_constant
        return np.cosh(u), np.sinh(u)


def nominal_foot_placements(settings: WalkingSettings) -> np.ndarray:
    lengths, widths = np.asarray(settings.step_lengths), np.asarray(settings.step_widths)
    if len(lengths) != len(widths):
        raise ValueError("step_lengths and step_widths must have equal length.")
    feet = np.zeros((len(lengths) + 1, 2))
    feet[0] = (0.0, -widths[0] / 2.0)
    for step, (length, width) in enumerate(zip(lengths, widths), start=1):
        feet[step] = feet[step - 1] + (length, width if step % 2 else -width)
    return feet


def primitive_target(feet: np.ndarray, phase: int, settings: WalkingSettings) -> tuple[np.ndarray, np.ndarray]:
    c, s = settings.transition_terms
    step = feet[min(phase + 1, len(feet) - 1)] - feet[phase]
    midpoint = step / 2.0
    tc = settings.time_constant
    velocity = midpoint * np.array(((c + 1.0) / (tc * s), (c - 1.0) / (tc * s)))
    return feet[phase] + midpoint, velocity


def optimized_support_coordinate(xi: float, vi: float, xd: float, vd: float, settings: WalkingSettings) -> float:
    c, s = settings.transition_terms
    tc = settings.time_constant
    denominator = settings.position_weight * (c - 1.0) ** 2 + settings.velocity_weight * (s / tc) ** 2
    position_error = xd - c * xi - tc * s * vi
    velocity_error = vd - (s / tc) * xi - c * vi
    return (-settings.position_weight * (c - 1.0) * position_error - settings.velocity_weight * s * velocity_error / tc) / denominator


def propagate_lipm(position: np.ndarray, velocity: np.ndarray, support: np.ndarray, settings: WalkingSettings, times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    tc = settings.time_constant
    c, s = np.cosh(times / tc)[:, None], np.sinh(times / tc)[:, None]
    relative = position - support
    return relative * c + tc * velocity * s + support, relative * s / tc + velocity * c


def plan_pattern(settings: WalkingSettings) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    feet = nominal_foot_placements(settings)
    position, velocity = feet[0].copy(), np.zeros(2)
    supports, trajectories = [], []
    times = np.linspace(0.0, settings.support_time, settings.samples_per_phase)
    for phase in range(len(feet)):
        target_position, target_velocity = primitive_target(feet, phase, settings)
        support = np.array([optimized_support_coordinate(position[axis], velocity[axis], target_position[axis], target_velocity[axis], settings) for axis in range(2)])
        positions, velocities = propagate_lipm(position, velocity, support, settings, times)
        supports.append(support)
        trajectories.append(positions)
        position, velocity = positions[-1], velocities[-1]
    return feet, np.asarray(supports), trajectories


def joint_indices(robot: int) -> dict[str, int]:
    return {p.getJointInfo(robot, index)[1].decode("utf-8"): index for index in range(p.getNumJoints(robot))}


def leg_indices(indices: dict[str, int], side: str) -> list[int]:
    return [indices[name] for name in RPO_LEG_JOINTS[side]]


def swing_position(start: np.ndarray, finish: np.ndarray, fraction: float, settings: WalkingSettings) -> np.ndarray:
    position = (1.0 - fraction) * start + fraction * finish
    position[2] = settings.foot_height + settings.swing_height * np.sin(np.pi * fraction)
    return position


def add_plan_debug_geometry(desired_feet: np.ndarray, supports: np.ndarray, settings: WalkingSettings) -> None:
    half_length, half_width = settings.foot_length / 2.0, settings.foot_width / 2.0
    for feet, color in ((desired_feet, (0.85, 0.15, 0.15)), (supports, (0.1, 0.1, 0.1))):
        for foot in feet:
            corners = [(foot[0] - half_length, foot[1] - half_width, .002), (foot[0] + half_length, foot[1] - half_width, .002), (foot[0] + half_length, foot[1] + half_width, .002), (foot[0] - half_length, foot[1] + half_width, .002)]
            for first, second in zip(corners, corners[1:] + corners[:1]):
                p.addUserDebugLine(first, second, color, lineWidth=1.4)


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


def center_of_mass_velocity(robot: int) -> np.ndarray:
    """Return the mass-weighted linear velocity of the robot."""
    weighted_velocity, mass_sum = np.zeros(3), 0.0
    for link in range(-1, p.getNumJoints(robot)):
        mass = p.getDynamicsInfo(robot, link)[0]
        if mass <= 0.0:
            continue
        velocity = (
            p.getBaseVelocity(robot)[0]
            if link == -1
            else p.getLinkState(robot, link, computeLinkVelocity=True)[6]
        )
        weighted_velocity += mass * np.asarray(velocity)
        mass_sum += mass
    return weighted_velocity / mass_sum


@dataclass(frozen=True)
class RobotModel:
    """Robot-specific names and geometry used by the controller."""

    urdf_path: Path
    right_side: str = "R"
    left_side: str = "L"


@dataclass(frozen=True)
class MotorSettings:
    """Joint-motor limits used to track the LIPM-derived IK posture."""

    support_force_limit: float = 120.0
    swing_force_limit: float = 70.0
    support_position_gain: float = 0.28
    swing_position_gain: float = 0.32


@dataclass(frozen=True)
class AnkleStabilizerSettings:
    natural_frequency: float = 2.5
    damping_ratio: float = 0.7
    torque_limit: float = 20.0


RPO = RobotModel(SCRIPT_DIRECTORY / "rpo_description" / "urdf" / "rpo.urdf")


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

def lock_non_walking_joints(robot: int, indices: dict[str, int]) -> None:
    """Hold the torso and arm joints at their neutral positions."""
    leg_names = set(RPO_LEG_JOINTS["R"] + RPO_LEG_JOINTS["L"])
    locked_joints = [index for name, index in indices.items() if name not in leg_names]
    for joint in locked_joints:
        p.resetJointState(robot, joint, 0.0, 0.0)
    p.setJointMotorControlArray(
        robot,
        locked_joints,
        p.POSITION_CONTROL,
        targetPositions=[0.0] * len(locked_joints),
        forces=[80.0] * len(locked_joints),
        positionGains=[0.4] * len(locked_joints),
    )

def solve_ik_leg(
    reference_model: int,
    reference_indices: dict[str, int],
    side: str,
    target: np.ndarray,
) -> np.ndarray:
    """Solve one leg of the reference model for a level-foot pose."""
    joints = leg_indices(reference_indices, side)
    end_effector = joints[-1]
    joint_count = p.getNumJoints(reference_model)
    lower_limits = [p.getJointInfo(reference_model, joint)[8] for joint in range(joint_count)]
    upper_limits = [p.getJointInfo(reference_model, joint)[9] for joint in range(joint_count)]
    joint_ranges = [upper - lower for lower, upper in zip(lower_limits, upper_limits)]
    rest_poses = [p.getJointState(reference_model, joint)[0] for joint in range(joint_count)]

    # A perfectly straight leg is an IK singularity. Seed a human-like bent
    # knee only when this leg has not yet acquired a valid previous solution.
    knee = joints[3]
    if abs(rest_poses[knee]) < 0.10:
        rest_poses[joints[2]] = -0.45
        rest_poses[knee] = 0.90
        rest_poses[joints[4]] = -0.45
        for joint in joints:
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
    lock_non_walking_joints(robot, indices)


def command_leg_postures(
    robot: int,
    indices: dict[str, int],
    support_side: str,
    desired_postures: dict[str, np.ndarray],
    motor_settings: MotorSettings,
) -> None:
    """Track the LIPM-derived support and swing leg postures with joint motors."""
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


def apply_ankle_torque_stabilizer(
    robot: int,
    indices: dict[str, int],
    support_side: str,
    support_point: np.ndarray,
    robot_mass: float,
    walking_settings: WalkingSettings,
    stabilizer_settings: AnkleStabilizerSettings,
) -> None:
    """Apply tau = -kp*x - kd*xdot at the support ankle pitch joint."""
    displacement = center_of_mass(robot)[0] - support_point[0]
    velocity = center_of_mass_velocity(robot)[0]
    omega = stabilizer_settings.natural_frequency
    damping = stabilizer_settings.damping_ratio
    height = walking_settings.com_height
    kp = robot_mass * (height * omega**2 + G)
    kd = 2.0 * robot_mass * height * damping * omega
    torque = -kp * displacement - kd * velocity
    ankle_pitch = indices[RPO_LEG_JOINTS[support_side][4]]
    limit = stabilizer_settings.torque_limit
    p.setJointMotorControl2(
        robot,
        ankle_pitch,
        p.TORQUE_CONTROL,
        force=float(np.clip(torque, -limit, limit)),
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
        leg_indices(indices, side)[-1],
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
        state += " (LIPM off, joints locked)"
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
    model: RobotModel = RPO,
    diagnostics: bool = False,
) -> np.ndarray:
    """Run the torque-controlled LIPM-to-multibody walking experiment."""
    if not model.urdf_path.is_file():
        raise FileNotFoundError(f"Missing robot model: {model.urdf_path}")

    settings = WalkingSettings()
    motor_settings = MotorSettings()
    stabilizer_settings = AnkleStabilizerSettings()
    desired_feet, supports, trajectories = plan_pattern(settings)
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
    robot_mass = total_mass(robot)

    for link in range(-1, p.getNumJoints(robot)):
        p.changeDynamics(
            robot,
            link,
            lateralFriction=1.3,
            spinningFriction=0.08,
            rollingFriction=0.1,
            linearDamping=0.0,
            angularDamping=0.0,
        )
    p.changeDynamics(plane, -1, lateralFriction=1.3, spinningFriction=0.08, rollingFriction=0.1)

    initial_com_target = np.array(
        (trajectories[0][0, 0], trajectories[0][0, 1], settings.com_height)
    )
    initial_right = ankle_target(supports[0], settings)
    initial_left = ankle_target(desired_feet[1], settings)

    # Establish the nominal multibody CoM-to-pelvis offset once, as assumed by
    # the LIPM-to-multibody conversion described in the walking method.
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
            add_plan_debug_geometry(desired_feet, supports, settings)
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
                    apply_ankle_torque_stabilizer(
                        robot,
                        indices,
                        support_side,
                        supports[phase],
                        robot_mass,
                        settings,
                        stabilizer_settings,
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

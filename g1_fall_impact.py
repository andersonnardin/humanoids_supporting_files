"""Demonstrate backward-fall impact reduction for G1.

The sequence squats to lower the center of mass, lets gravity continue the
backward fall, then extends the legs after touchdown begins.
It is a PyBullet teaching demonstration, not a hardware-safe fall controller.
The G1 URDF has no hip or back padding, so do not interpret its contacts as
real impact loads.

Run:
    python3 g1_fall_impact.py

Controls:
    E: pause or resume.
    T: stop the fall controller and hold the current posture.
    R: restart the sequence.
    Q: quit.
"""

from __future__ import annotations

from enum import Enum, auto
from pathlib import Path
import time

import numpy as np

try:
    import pybullet as p
    import pybullet_data
except ImportError as error:  # pragma: no cover
    raise SystemExit("Install PyBullet first: python3 -m pip install pybullet") from error


ROOT = Path(__file__).resolve().parent
G1_URDF = ROOT / "g1_description" / "g1_29dof.urdf"
TIME_STEP = 1.0 / 240.0
GROUND_CLEARANCE = 0.002
FALL_DETECTION_ANGLE_DEGREES = 85.0
SQUAT_SECONDS = 2.0
FALL_PREPARATION_SECONDS = 1.0
BACKWARD_COM_MARGIN = 0.015
EXTEND1_ANGLE_DEGREES = 30.0
TOUCHDOWN_ANGLE_DEGREES = 25.0
EXTEND1_SECONDS = 0.55
TOUCHDOWN_SECONDS = 0.25
EXTEND2_SECONDS = 2.0

STANCE = (-0.28, 0.62, -0.30)
SQUAT = (-1.35, 2.22, -0.87)
EXTENDED = (-0.18, 0.30, -0.12)


class FallState(Enum):
    MONITORING = auto()
    SQUATTING = auto()
    EXTEND1 = auto()
    TOUCHDOWN = auto()
    EXTEND2 = auto()
    FINISH = auto()


def smoothstep(value: float) -> float:
    value = float(np.clip(value, 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)


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


def posture_targets(
    robot: int,
    indices: dict[str, int],
    leg_pose: tuple[float, float, float],
    protective_upper_body: bool = False,
    rear_shift: float = 0.0,
) -> dict[int, float]:
    """Set symmetric legs and the waist-and-arm pose used during a back fall."""
    targets = {joint: 0.0 for joint in movable_joints(robot)}
    hip, knee, ankle = leg_pose
    for side in ("left", "right"):
        targets[indices[f"{side}_hip_pitch_joint"]] = hip
        targets[indices[f"{side}_knee_joint"]] = knee
        targets[indices[f"{side}_ankle_pitch_joint"]] = ankle
    if protective_upper_body:
        # G1 has no actuated neck in this model. The waist and arms aim the
        # pelvis toward the ground.
        targets[indices["waist_pitch_joint"]] = 0.42 + 0.10 * rear_shift
        for side in ("left", "right"):
            targets[indices[f"{side}_shoulder_pitch_joint"]] = -0.80
            targets[indices[f"{side}_elbow_joint"]] = 0.80
    return targets


def current_leg_pose(robot: int, indices: dict[str, int]) -> tuple[float, float, float]:
    """Return the average current pitch posture of the two physical legs."""
    values = []
    for joint_part in ("hip_pitch_joint", "knee_joint", "ankle_pitch_joint"):
        left = p.getJointState(robot, indices[f"left_{joint_part}"])[0]
        right = p.getJointState(robot, indices[f"right_{joint_part}"])[0]
        values.append(0.5 * (left + right))
    return tuple(values)


def command_posture(
    robot: int,
    targets: dict[int, float],
    gain: float = 0.34,
    release_ankles: bool = False,
) -> None:
    joints = list(targets)
    positions = []
    forces = []
    for joint in joints:
        lower, upper = p.getJointInfo(robot, joint)[8:10]
        value = targets[joint]
        positions.append(float(np.clip(value, lower, upper)) if lower < upper else value)
        forces.append(max(15.0, 1.1 * float(p.getJointInfo(robot, joint)[10])))
    p.setJointMotorControlArray(
        robot,
        joints,
        p.POSITION_CONTROL,
        targetPositions=positions,
        targetVelocities=[0.0] * len(joints),
        forces=forces,
        positionGains=[gain] * len(joints),
        velocityGains=[1.0] * len(joints),
    )
    if release_ankles:
        ankle_joints = [
            joint
            for joint in joints
            if p.getJointInfo(robot, joint)[1].decode("utf-8")
            in {"left_ankle_pitch_joint", "right_ankle_pitch_joint"}
        ]
        p.setJointMotorControlArray(
            robot,
            ankle_joints,
            p.VELOCITY_CONTROL,
            targetVelocities=[0.0] * len(ankle_joints),
            forces=[0.0] * len(ankle_joints),
        )


def disable_servos(robot: int) -> None:
    """Release all joints during the touchdown preparation state."""
    joints = movable_joints(robot)
    p.setJointMotorControlArray(
        robot,
        joints,
        p.VELOCITY_CONTROL,
        targetVelocities=[0.0] * len(joints),
        forces=[0.0] * len(joints),
    )


def captured_joint_positions(robot: int) -> list[float]:
    """Capture every joint angle when T stops the fall controller."""
    return [p.getJointState(robot, joint)[0] for joint in range(p.getNumJoints(robot))]


def hold_current_joint_state(robot: int, targets: list[float]) -> None:
    """Use joint motors as brakes after the learner presses T."""
    joints = list(range(p.getNumJoints(robot)))
    p.setJointMotorControlArray(
        robot,
        joints,
        p.POSITION_CONTROL,
        targetPositions=targets,
        forces=[140.0] * len(joints),
        positionGains=[0.45] * len(joints),
    )


def center_of_mass(robot: int) -> np.ndarray:
    """Return the current mass-weighted center of mass."""
    weighted_position = np.zeros(3)
    total_mass = 0.0
    base_position, _ = p.getBasePositionAndOrientation(robot)
    base_mass = float(p.getDynamicsInfo(robot, -1)[0])
    weighted_position += base_mass * np.asarray(base_position)
    total_mass += base_mass
    for link in range(p.getNumJoints(robot)):
        mass = float(p.getDynamicsInfo(robot, link)[0])
        if mass <= 0.0:
            continue
        position = np.asarray(p.getLinkState(robot, link)[0])
        weighted_position += mass * position
        total_mass += mass
    return weighted_position / total_mass


def support_center(robot: int, indices: dict[str, int]) -> np.ndarray:
    """Return the midpoint of the two ankle links in the stance support area."""
    left = np.asarray(p.getLinkState(robot, indices["left_ankle_roll_joint"])[0])
    right = np.asarray(p.getLinkState(robot, indices["right_ankle_roll_joint"])[0])
    return 0.5 * (left + right)


def heel_to_hip_angle(robot: int, indices: dict[str, int]) -> float:
    """Approximate the book's ground-to-heel-to-hip angle theta in degrees."""
    heel_center = 0.5 * sum(
        (
            np.asarray(p.getLinkState(robot, indices[f"{side}_ankle_roll_joint"])[0])
            for side in ("left", "right")
        ),
        np.zeros(3),
    )
    hip_position = np.asarray(p.getBasePositionAndOrientation(robot)[0])
    horizontal_distance = abs(hip_position[0] - heel_center[0])
    return float(np.degrees(np.arctan2(max(0.0, hip_position[2]), horizontal_distance)))


def ground_reaction_force(robot: int, plane: int) -> float:
    """Return the magnitude of the resultant force from the floor on G1."""
    resultant = np.zeros(3)
    for contact in p.getContactPoints(bodyA=robot, bodyB=plane):
        resultant += float(contact[9]) * np.asarray(contact[7])
        resultant += float(contact[10]) * np.asarray(contact[11])
        resultant += float(contact[12]) * np.asarray(contact[13])
    return float(np.linalg.norm(resultant))


def load_scene() -> tuple[int, int, dict[str, int]]:
    if not G1_URDF.exists():
        raise FileNotFoundError(f"G1 URDF not found: {G1_URDF}")
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    plane = p.loadURDF("plane.urdf")
    p.changeDynamics(
        plane,
        -1,
        lateralFriction=1.3,
        spinningFriction=0.08,
        rollingFriction=0.1,
    )
    p.setAdditionalSearchPath(str(G1_URDF.parent))
    robot = p.loadURDF(
        str(G1_URDF),
        basePosition=(0.0, 0.0, 0.78),
        useFixedBase=False,
        flags=p.URDF_USE_INERTIA_FROM_FILE,
    )
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
    return plane, robot, joint_indices(robot)


def place_feet_on_floor(robot: int, indices: dict[str, int]) -> None:
    """Place the lowest foot collision point just above the ground plane."""
    p.performCollisionDetection()
    lowest_foot_point = min(
        p.getAABB(robot, indices[f"{side}_ankle_roll_joint"])[0][2]
        for side in ("left", "right")
    )
    position, orientation = p.getBasePositionAndOrientation(robot)
    p.resetBasePositionAndOrientation(
        robot,
        (position[0], position[1], position[2] + GROUND_CLEARANCE - lowest_foot_point),
        orientation,
    )


def reset_sequence(robot: int, indices: dict[str, int]) -> None:
    targets = posture_targets(robot, indices, STANCE)
    for joint, value in targets.items():
        p.resetJointState(robot, joint, value)
    p.resetBasePositionAndOrientation(robot, (0.0, 0.0, 0.78), (0.0, 0.0, 0.0, 1.0))
    p.resetBaseVelocity(robot, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    place_feet_on_floor(robot, indices)
    command_posture(robot, targets)


def main() -> None:
    client = p.connect(p.GUI, options="--width=900 --height=700")
    if client < 0:
        raise RuntimeError("PyBullet could not open the presentation window.")
    try:
        p.setGravity(0.0, 0.0, -9.81)
        p.setTimeStep(TIME_STEP)
        p.setPhysicsEngineParameter(numSolverIterations=150, enableConeFriction=1)
        p.setRealTimeSimulation(0)
        plane, robot, indices = load_scene()
        p.resetDebugVisualizerCamera(
            cameraDistance=2.0,
            cameraYaw=108.0,
            cameraPitch=-14.0,
            cameraTargetPosition=(0.0, 0.0, 0.42),
        )

        reset_sequence(robot, indices)
        state = FallState.MONITORING
        state_time = 0.0
        extend2_start_pose = SQUAT
        paused = False
        control_stopped = False
        stopped_joint_targets: list[float] = []
        status_id = -1
        force_id = -1
        maximum_impact_force = 0.0

        while p.isConnected(client):
            keys = p.getKeyboardEvents()
            if keys.get(ord("q"), 0) & p.KEY_WAS_TRIGGERED:
                break
            if keys.get(ord("e"), 0) & p.KEY_WAS_TRIGGERED:
                paused = not paused
            if keys.get(ord("t"), 0) & p.KEY_WAS_TRIGGERED and not control_stopped:
                control_stopped = True
                stopped_joint_targets = captured_joint_positions(robot)
            if keys.get(ord("r"), 0) & p.KEY_WAS_TRIGGERED:
                reset_sequence(robot, indices)
                state = FallState.MONITORING
                state_time = 0.0
                extend2_start_pose = SQUAT
                paused = False
                control_stopped = False
                stopped_joint_targets = []
                maximum_impact_force = 0.0

            if paused:
                time.sleep(TIME_STEP)
                continue

            if control_stopped:
                hold_current_joint_state(robot, stopped_joint_targets)
                status_id = p.addUserDebugText(
                    "State: Control stopped",
                    (-0.45, 0.0, 1.30),
                    textColorRGB=(0.05, 0.05, 0.05),
                    textSize=1.2,
                    replaceItemUniqueId=status_id,
                )
                p.stepSimulation()
                force_magnitude = ground_reaction_force(robot, plane)
                maximum_impact_force = max(maximum_impact_force, force_magnitude)
                force_id = p.addUserDebugText(
                    f"Max floor impact force: {maximum_impact_force:.0f} N",
                    (-0.45, 0.0, 1.18),
                    textColorRGB=(0.85, 0.05, 0.05),
                    textSize=1.2,
                    replaceItemUniqueId=force_id,
                )
                time.sleep(TIME_STEP)
                continue

            theta = heel_to_hip_angle(robot, indices)
            com = center_of_mass(robot)
            support = support_center(robot, indices)
            if state is FallState.MONITORING:
                command_posture(robot, posture_targets(robot, indices, STANCE))
                if theta <= FALL_DETECTION_ANGLE_DEGREES:
                    state = FallState.SQUATTING
                    state_time = 0.0
            elif state is FallState.SQUATTING:
                blend = smoothstep(state_time / SQUAT_SECONDS)
                rear_shift = smoothstep(
                    (state_time - SQUAT_SECONDS) / FALL_PREPARATION_SECONDS
                )
                pose = tuple(
                    (1.0 - blend) * start + blend * finish
                    for start, finish in zip(STANCE, SQUAT)
                )
                command_posture(
                    robot,
                    posture_targets(
                        robot,
                        indices,
                        pose,
                        protective_upper_body=True,
                        rear_shift=rear_shift,
                    ),
                    release_ankles=state_time >= SQUAT_SECONDS,
                )
                # The waist and arms shift the CoM behind the heels. Gravity,
                # rather than an imposed force or base velocity, starts the fall.
                if (
                    state_time >= SQUAT_SECONDS
                    and com[0] < support[0] - BACKWARD_COM_MARGIN
                    and theta <= EXTEND1_ANGLE_DEGREES
                ):
                    state = FallState.EXTEND1
                    state_time = 0.0
            elif state is FallState.EXTEND1:
                # Extending the legs while theta is still moderate lowers the
                # hip landing speed and helps keep the hips as the first target.
                blend = smoothstep(state_time / EXTEND1_SECONDS)
                pose = tuple(
                    (1.0 - blend) * start + blend * finish
                    for start, finish in zip(SQUAT, EXTENDED)
                )
                command_posture(
                    robot,
                    posture_targets(robot, indices, pose, protective_upper_body=True),
                    gain=0.26,
                )
                if theta <= TOUCHDOWN_ANGLE_DEGREES:
                    state = FallState.TOUCHDOWN
                    state_time = 0.0
                    disable_servos(robot)
            elif state is FallState.TOUCHDOWN:
                # Servo control remains off while the hip/back impact occurs.
                if state_time >= TOUCHDOWN_SECONDS:
                    state = FallState.EXTEND2
                    state_time = 0.0
                    extend2_start_pose = current_leg_pose(robot, indices)
            elif state is FallState.EXTEND2:
                # A later extension counters continued backward rotation toward
                # the head after the initial hip/back landing.
                blend = smoothstep(state_time / EXTEND2_SECONDS)
                pose = tuple(
                    (1.0 - blend) * start + blend * finish
                    for start, finish in zip(extend2_start_pose, EXTENDED)
                )
                command_posture(
                    robot,
                    posture_targets(robot, indices, pose, protective_upper_body=True),
                    gain=0.18,
                )
                if state_time >= EXTEND2_SECONDS:
                    state = FallState.FINISH
            else:
                command_posture(
                    robot,
                    posture_targets(robot, indices, EXTENDED, protective_upper_body=True),
                    gain=0.16,
                )

            status_id = p.addUserDebugText(
                f"State: {state.name.title()} | theta: {theta:.1f} deg",
                (-0.45, 0.0, 1.30),
                textColorRGB=(0.05, 0.05, 0.05),
                textSize=1.2,
                replaceItemUniqueId=status_id,
            )
            p.stepSimulation()
            force_magnitude = ground_reaction_force(robot, plane)
            maximum_impact_force = max(maximum_impact_force, force_magnitude)
            force_id = p.addUserDebugText(
                f"Max floor impact force: {maximum_impact_force:.0f} N",
                (-0.45, 0.0, 1.18),
                textColorRGB=(0.85, 0.05, 0.05),
                textSize=1.2,
                replaceItemUniqueId=force_id,
            )
            state_time += TIME_STEP
            time.sleep(TIME_STEP)
    finally:
        if p.isConnected(client):
            p.disconnect(client)


if __name__ == "__main__":
    main()

"""Run the G1 RRT reach with a CoM-Reference Controller.

The G1 reaches beneath a table while a single rapidly exploring random tree
searches a 15-dimensional posture vector: one coordinated squat variable,
three waist joints, seven right-arm joints, and four left-arm joints. Every node and
interpolated edge must respect joint limits and avoid self-collision and table
collision, including the visible terminal hand mesh omitted by the G1 URDF's
collision model.

The complete RRT stage from the supplied G1 planner is retained: it produces
a collision-free rough posture trajectory beneath the table. During playback,
the G1 has a free base and gravity is enabled. Its support comes only from
physical foot-floor contacts and friction. The robot first holds a quiet
double-support posture, then a compact CoM-Reference Controller uses a filtered CoM error
relative to that measured support geometry to adjust ankle, hip, and waist
targets at every physics step.

This simplified controller is inspired by the Auto-Balancer concept. It is not
a hardware-ready second-order whole-body optimizer or a proof of dynamic feasibility.

Controls:
    E: pause or resume.
    T: stop the CoM-Reference Controller and hold the current joint angles.
    R: restart the motion.
    Q: quit.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
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

STARTING_BASE_HEIGHT = 0.774
GROUND_CLEARANCE = 0.002
STANCE_X = -0.15
TIME_STEP = 1.0 / 240.0
SETTLE_SECONDS = 1.5
BALANCE_RAMP_SECONDS = 0.8
BALANCE_FILTER = 0.90
MAXIMUM_SUPPORT_ERROR = 0.10
TABLE_SAFETY_DISTANCE = 0.06
# Gravity-tested lower-body postures used during dynamic playback.  The RRT's
# original leg angles remain unchanged during geometric planning.
STANCE_HIP = -0.28
STANCE_KNEE = 0.62
STANCE_ANKLE = -0.30
SQUAT_HIP = -1.35
SQUAT_KNEE = 2.22
SQUAT_ANKLE = -0.87

TABLETOP_HEIGHT = 0.70
TARGET_POSITION = np.array((0.33, -0.22, 0.60))

ACTIVE_JOINT_NAMES = (
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
)
RIGHT_HAND_NAME = "right_hand_palm_joint"
LEFT_FOOT_NAME = "left_ankle_roll_joint"
RIGHT_FOOT_NAME = "right_ankle_roll_joint"

# The G1 hand link has a visual mesh but no active collision element.
HAND_LOCAL_MIN = np.array((0.0, -0.0187, -0.0431))
HAND_LOCAL_MAX = np.array((0.1319, 0.0480, 0.0635))
HAND_TARGET_LOCAL = np.array((0.075, 0.015, 0.010))
HAND_TABLE_CLEARANCE = 0.004
HAND_SAMPLE_POINTS = np.asarray(
    [
        (x, y, z)
        for x in np.linspace(HAND_LOCAL_MIN[0], HAND_LOCAL_MAX[0], 9)
        for y in np.linspace(HAND_LOCAL_MIN[1], HAND_LOCAL_MAX[1], 5)
        for z in np.linspace(HAND_LOCAL_MIN[2], HAND_LOCAL_MAX[2], 5)
    ]
)


@dataclass(frozen=True)
class Settings:
    step_size: float = 0.24
    edge_resolution: float = 0.025
    max_iterations: int = 14000
    goal_bias: float = 0.18
    goal_tolerance: float = 0.055
    animation_seconds: float = 7.0


@dataclass
class Plan:
    trajectory: list[np.ndarray]
    hand_targets: list[np.ndarray]
    explored_nodes: int
    path_nodes: int
    planning_seconds: float
    final_error: float


@dataclass
class BalanceState:
    sagittal: float = 0.0
    lateral: float = 0.0


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


def reset_neutral(robot: int) -> None:
    for joint in movable_joints(robot):
        p.resetJointState(robot, joint, 0.0)


def apply_squat(robot: int, indices: dict[str, int], amount: float) -> None:
    """Bend both legs and compensate the base so both soles remain down."""
    blend = amount * amount * (3.0 - 2.0 * amount)
    for side in ("left", "right"):
        p.resetJointState(robot, indices[f"{side}_hip_pitch_joint"], SQUAT_HIP * blend)
        p.resetJointState(robot, indices[f"{side}_knee_joint"], SQUAT_KNEE * blend)
        p.resetJointState(robot, indices[f"{side}_ankle_pitch_joint"], SQUAT_ANKLE * blend)

    p.performCollisionDetection()
    left_ankle = np.asarray(
        p.getLinkState(robot, indices[LEFT_FOOT_NAME], computeForwardKinematics=True)[4]
    )
    right_ankle = np.asarray(
        p.getLinkState(robot, indices[RIGHT_FOOT_NAME], computeForwardKinematics=True)[4]
    )
    ankle_center = 0.5 * (left_ankle + right_ankle)
    sole_z = min(
        p.getAABB(robot, indices[LEFT_FOOT_NAME])[0][2],
        p.getAABB(robot, indices[RIGHT_FOOT_NAME])[0][2],
    )
    base_position = p.getBasePositionAndOrientation(robot)[0]
    corrected = (
        base_position[0] - ankle_center[0] + STANCE_X,
        base_position[1],
        base_position[2] + GROUND_CLEARANCE - sole_z,
    )
    p.resetBasePositionAndOrientation(robot, corrected, (0.0, 0.0, 0.0, 1.0))


def load_g1(fixed_base: bool = True) -> int:
    if not G1_URDF.exists():
        raise FileNotFoundError(f"G1 URDF not found: {G1_URDF}")
    p.setAdditionalSearchPath(str(G1_URDF.parent))
    return p.loadURDF(
        str(G1_URDF),
        basePosition=(0.0, 0.0, STARTING_BASE_HEIGHT),
        useFixedBase=fixed_base,
        flags=p.URDF_USE_INERTIA_FROM_FILE | p.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT,
    )


def create_table() -> list[int]:
    parts = []
    top_half = (0.32, 0.42, 0.025)
    top_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=top_half)
    top_visual = p.createVisualShape(
        p.GEOM_BOX, halfExtents=top_half, rgbaColor=(0.78, 0.38, 0.08, 1.0)
    )
    parts.append(
        p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=top_collision,
            baseVisualShapeIndex=top_visual,
            basePosition=(0.62, 0.0, TABLETOP_HEIGHT),
        )
    )

    leg_half = (0.03, 0.03, (TABLETOP_HEIGHT - top_half[2]) / 2.0)
    leg_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=leg_half)
    leg_visual = p.createVisualShape(
        p.GEOM_BOX, halfExtents=leg_half, rgbaColor=(0.34, 0.16, 0.05, 1.0)
    )
    for x, y in ((0.36, -0.35), (0.36, 0.35), (0.88, -0.35), (0.88, 0.35)):
        parts.append(
            p.createMultiBody(
                baseMass=0.0,
                baseCollisionShapeIndex=leg_collision,
                baseVisualShapeIndex=leg_visual,
                basePosition=(x, y, leg_half[2]),
            )
        )
    return parts


def create_target() -> None:
    visual = p.createVisualShape(
        p.GEOM_SPHERE, radius=0.03, rgbaColor=(0.92, 0.08, 0.05, 1.0)
    )
    p.createMultiBody(baseMass=0.0, baseVisualShapeIndex=visual, basePosition=TARGET_POSITION)
    p.addUserDebugText(
        "Hand target",
        TARGET_POSITION + np.array((0.0, 0.0, 0.05)),
        textColorRGB=(0.80, 0.05, 0.03),
        textSize=1.0,
    )


def state_bounds(robot: int, active_joints: list[int]) -> tuple[np.ndarray, np.ndarray]:
    lower = [0.0]
    upper = [1.0]
    for joint in active_joints:
        low, high = p.getJointInfo(robot, joint)[8:10]
        if low >= high:
            low, high = -np.pi, np.pi
        lower.append(low)
        upper.append(high)
    return np.asarray(lower), np.asarray(upper)


def apply_state(
    robot: int,
    indices: dict[str, int],
    active_joints: list[int],
    state: np.ndarray,
) -> None:
    apply_squat(robot, indices, float(state[0]))
    for joint, value in zip(active_joints, state[1:]):
        p.resetJointState(robot, joint, float(value))


def hand_frame(robot: int, hand: int) -> tuple[np.ndarray, np.ndarray]:
    state = p.getLinkState(robot, hand, computeForwardKinematics=True)
    origin = np.asarray(state[4])
    rotation = np.asarray(p.getMatrixFromQuaternion(state[5])).reshape(3, 3)
    return origin, rotation


def hand_position(robot: int, hand: int) -> np.ndarray:
    origin, rotation = hand_frame(robot, hand)
    return origin + rotation @ HAND_TARGET_LOCAL


def visible_hand_hits_table(robot: int, hand: int, table_parts: list[int]) -> bool:
    origin, rotation = hand_frame(robot, hand)
    world_points = origin + HAND_SAMPLE_POINTS @ rotation.T
    for table in table_parts:
        aabb_min, aabb_max = p.getAABB(table)
        lower = np.asarray(aabb_min) - HAND_TABLE_CLEARANCE
        upper = np.asarray(aabb_max) + HAND_TABLE_CLEARANCE
        if np.any(np.all((world_points >= lower) & (world_points <= upper), axis=1)):
            return True
    return False


def state_is_valid(
    robot: int,
    indices: dict[str, int],
    active_joints: list[int],
    state: np.ndarray,
    table_parts: list[int],
) -> bool:
    apply_state(robot, indices, active_joints, state)
    p.performCollisionDetection()
    if any(p.getContactPoints(robot, table) for table in table_parts):
        return False
    if visible_hand_hits_table(robot, indices[RIGHT_HAND_NAME], table_parts):
        return False
    return not p.getContactPoints(robot, robot)


def edge_is_valid(
    robot: int,
    indices: dict[str, int],
    active_joints: list[int],
    start: np.ndarray,
    finish: np.ndarray,
    table_parts: list[int],
    settings: Settings,
) -> bool:
    distance = np.linalg.norm(finish - start)
    samples = max(2, int(np.ceil(distance / settings.edge_resolution)) + 1)
    for fraction in np.linspace(0.0, 1.0, samples):
        candidate = start + fraction * (finish - start)
        if not state_is_valid(robot, indices, active_joints, candidate, table_parts):
            return False
    return True


def goal_state(
    robot: int,
    indices: dict[str, int],
    active_joints: list[int],
    hand: int,
    table_parts: list[int],
    lower: np.ndarray,
    upper: np.ndarray,
    settings: Settings,
    rng: np.random.Generator,
) -> np.ndarray:
    seed = np.zeros(len(active_joints) + 1)
    seed[0] = 1.0
    nominal = seed.copy()
    for name, value in (
        ("left_shoulder_pitch_joint", -0.35),
        ("left_shoulder_roll_joint", 0.20),
        ("left_elbow_joint", 0.65),
    ):
        nominal[1 + active_joints.index(indices[name])] = value
    nominal = np.clip(nominal, lower, upper)

    # Solve over the three waist coordinates and seven right-arm coordinates.
    controlled_columns = range(1, 11)
    epsilon = 1e-4
    damping = 2e-3
    for _ in range(120):
        apply_state(robot, indices, active_joints, nominal)
        current = hand_position(robot, hand)
        error = TARGET_POSITION - current
        if np.linalg.norm(error) < 0.004:
            break
        jacobian = np.zeros((3, len(controlled_columns)))
        for jacobian_column, state_column in enumerate(controlled_columns):
            perturbed = nominal.copy()
            perturbed[state_column] = np.clip(
                perturbed[state_column] + epsilon,
                lower[state_column],
                upper[state_column],
            )
            apply_state(robot, indices, active_joints, perturbed)
            jacobian[:, jacobian_column] = (
                hand_position(robot, hand) - current
            ) / epsilon
        system = jacobian @ jacobian.T + damping * np.eye(3)
        update = jacobian.T @ np.linalg.solve(system, error)
        update = np.clip(update, -0.14, 0.14)
        for jacobian_column, state_column in enumerate(controlled_columns):
            nominal[state_column] = np.clip(
                nominal[state_column] + update[jacobian_column],
                lower[state_column],
                upper[state_column],
            )

    candidates = [nominal]
    for _ in range(1100):
        candidate = nominal + rng.normal(0.0, 0.16, len(nominal))
        candidate[0] = np.clip(rng.normal(0.94, 0.08), 0.65, 1.0)
        candidates.append(np.clip(candidate, lower, upper))

    best = None
    best_error = float("inf")
    for candidate in candidates:
        if not state_is_valid(robot, indices, active_joints, candidate, table_parts):
            continue
        error = np.linalg.norm(hand_position(robot, hand) - TARGET_POSITION)
        if error < best_error:
            best = candidate.copy()
            best_error = error
    if best is None or best_error > settings.goal_tolerance:
        raise RuntimeError(
            f"No collision-free G1 reaching posture was found; best error: {best_error:.3f} m."
        )
    return best


def steer(start: np.ndarray, finish: np.ndarray, step_size: float) -> np.ndarray:
    direction = finish - start
    distance = np.linalg.norm(direction)
    return finish.copy() if distance <= step_size else start + step_size * direction / distance


def nearest(nodes: list[np.ndarray], sample: np.ndarray) -> int:
    return int(np.argmin([np.linalg.norm(node - sample) for node in nodes]))


def trace(nodes: list[np.ndarray], parents: list[int], index: int) -> list[np.ndarray]:
    branch = []
    while index >= 0:
        branch.append(nodes[index])
        index = parents[index]
    return list(reversed(branch))


def basic_rrt(
    robot: int,
    indices: dict[str, int],
    active_joints: list[int],
    start: np.ndarray,
    goal: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    table_parts: list[int],
    settings: Settings,
    rng: np.random.Generator,
) -> tuple[list[np.ndarray], int]:
    nodes = [start]
    parents = [-1]
    for _ in range(settings.max_iterations):
        draw = rng.random()
        if draw < settings.goal_bias:
            sample = goal.copy()
        elif draw < 0.82:
            fraction = rng.uniform(0.0, 1.0)
            sample = (1.0 - fraction) * start + fraction * goal
            sample += rng.normal(0.0, 0.22, len(sample))
            sample[0] = np.clip(fraction + rng.normal(0.0, 0.12), 0.0, 1.0)
            sample = np.clip(sample, lower, upper)
        else:
            sample = rng.uniform(lower, upper)
            sample[0] = rng.uniform(0.25, 1.0)

        parent = nearest(nodes, sample)
        candidate = np.clip(steer(nodes[parent], sample, settings.step_size), lower, upper)
        if not edge_is_valid(
            robot, indices, active_joints, nodes[parent], candidate, table_parts, settings
        ):
            continue
        nodes.append(candidate)
        parents.append(parent)
        candidate_index = len(nodes) - 1
        if np.linalg.norm(candidate - goal) > settings.step_size:
            continue
        if not edge_is_valid(
            robot, indices, active_joints, candidate, goal, table_parts, settings
        ):
            continue
        nodes.append(goal.copy())
        parents.append(candidate_index)
        return trace(nodes, parents, len(nodes) - 1), len(nodes)
    raise RuntimeError(f"The G1 RRT failed after {settings.max_iterations} iterations.")


def shorten_path(
    robot: int,
    indices: dict[str, int],
    active_joints: list[int],
    path: list[np.ndarray],
    table_parts: list[int],
    settings: Settings,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    result = path.copy()
    for _ in range(240):
        if len(result) <= 3:
            break
        first, second = sorted(rng.choice(len(result), 2, replace=False))
        if second - first <= 1:
            continue
        shortened = result[: first + 1] + result[second:]
        if len(shortened) >= 3 and edge_is_valid(
            robot,
            indices,
            active_joints,
            result[first],
            result[second],
            table_parts,
            settings,
        ):
            result = shortened
    return result


def interpolate(path: list[np.ndarray], samples_per_edge: int = 40) -> list[np.ndarray]:
    trajectory = []
    for start, finish in zip(path[:-1], path[1:]):
        for fraction in np.linspace(0.0, 1.0, samples_per_edge, endpoint=False):
            blend = fraction * fraction * (3.0 - 2.0 * fraction)
            trajectory.append((1.0 - blend) * start + blend * finish)
    return trajectory + [path[-1]]


def state_joint_targets(
    robot: int,
    indices: dict[str, int],
    active_joints: list[int],
    state: np.ndarray,
) -> dict[int, float]:
    """Convert one RRT state into joint targets without moving the free base."""
    amount = float(state[0])
    blend = amount * amount * (3.0 - 2.0 * amount)
    # Keep every unplanned joint at its neutral angle so the free-base model
    # does not acquire uncontrolled leg-yaw, finger, or head motion.
    targets = {joint: 0.0 for joint in movable_joints(robot)}
    targets.update({
        indices[f"{side}_hip_pitch_joint"]: (
            (1.0 - blend) * STANCE_HIP + blend * SQUAT_HIP
        )
        for side in ("left", "right")
    })
    for side in ("left", "right"):
        targets[indices[f"{side}_knee_joint"]] = (
            (1.0 - blend) * STANCE_KNEE + blend * SQUAT_KNEE
        )
        targets[indices[f"{side}_ankle_pitch_joint"]] = (
            (1.0 - blend) * STANCE_ANKLE + blend * SQUAT_ANKLE
        )
    targets.update(
        {joint: float(value) for joint, value in zip(active_joints, state[1:])}
    )
    return targets


def reset_dynamic_state(
    robot: int,
    indices: dict[str, int],
    active_joints: list[int],
    state: np.ndarray,
) -> None:
    """Reset the free robot to the balanced version of one RRT state."""
    for joint, value in state_joint_targets(
        robot, indices, active_joints, state
    ).items():
        p.resetJointState(robot, joint, value)
    position, _ = p.getBasePositionAndOrientation(robot)
    p.resetBasePositionAndOrientation(
        robot,
        (STANCE_X, 0.0, position[2]),
        (0.0, 0.0, 0.0, 1.0),
    )
    place_feet_on_floor(robot, indices)
    p.resetBaseVelocity(robot, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))


def place_feet_on_floor(robot: int, indices: dict[str, int]) -> None:
    """Place the lowest foot collision point just above the floor at startup."""
    p.performCollisionDetection()
    lowest = min(
        p.getAABB(robot, indices[name])[0][2]
        for name in (LEFT_FOOT_NAME, RIGHT_FOOT_NAME)
    )
    position, orientation = p.getBasePositionAndOrientation(robot)
    corrected = (position[0], position[1], position[2] + GROUND_CLEARANCE - lowest)
    p.resetBasePositionAndOrientation(robot, corrected, orientation)


def center_of_mass(robot: int) -> tuple[np.ndarray, np.ndarray]:
    """Return the mass-weighted whole-body CoM position and velocity."""
    position_sum = np.zeros(3)
    velocity_sum = np.zeros(3)
    total_mass = 0.0

    base_position, _ = p.getBasePositionAndOrientation(robot)
    base_velocity, _ = p.getBaseVelocity(robot)
    base_mass = float(p.getDynamicsInfo(robot, -1)[0])
    position_sum += base_mass * np.asarray(base_position)
    velocity_sum += base_mass * np.asarray(base_velocity)
    total_mass += base_mass

    for link in range(p.getNumJoints(robot)):
        mass = float(p.getDynamicsInfo(robot, link)[0])
        if mass <= 0.0:
            continue
        state = p.getLinkState(robot, link, computeLinkVelocity=True)
        position_sum += mass * np.asarray(state[0])
        velocity_sum += mass * np.asarray(state[6])
        total_mass += mass
    return position_sum / total_mass, velocity_sum / total_mass


def support_center(robot: int, indices: dict[str, int]) -> np.ndarray:
    """Return the geometric midpoint of the two physical stance feet."""
    feet = [
        np.asarray(
            p.getLinkState(
                robot, indices[name], computeForwardKinematics=True
            )[0]
        )
        for name in (LEFT_FOOT_NAME, RIGHT_FOOT_NAME)
    ]
    return 0.5 * (feet[0] + feet[1])


def table_clearance(robot: int, table_parts: list[int]) -> float:
    """Return the nearest robot-to-table distance within the safety margin."""
    nearest = TABLE_SAFETY_DISTANCE
    for table in table_parts:
        points = p.getClosestPoints(robot, table, TABLE_SAFETY_DISTANCE)
        if points:
            nearest = min(nearest, min(float(point[8]) for point in points))
    return nearest


def auto_balance_targets(
    robot: int,
    indices: dict[str, int],
    rough_targets: dict[int, float],
    support_reference: np.ndarray,
    balance_state: BalanceState,
    balance_weight: float,
) -> tuple[dict[int, float], np.ndarray, BalanceState]:
    """Add filtered, bounded CoM feedback while remaining near the RRT posture."""
    com, com_velocity = center_of_mass(robot)
    error = com - support_reference

    requested_sagittal = float(
        np.clip(10.0 * error[0] + 1.50 * com_velocity[0], -0.60, 0.60)
    )
    requested_lateral = float(
        np.clip(-5.0 * error[1] - 0.80 * com_velocity[1], -0.25, 0.25)
    )
    balance_state.sagittal = (
        BALANCE_FILTER * balance_state.sagittal
        + (1.0 - BALANCE_FILTER) * requested_sagittal
    )
    balance_state.lateral = (
        BALANCE_FILTER * balance_state.lateral
        + (1.0 - BALANCE_FILTER) * requested_lateral
    )
    sagittal = balance_weight * balance_state.sagittal
    lateral = balance_weight * balance_state.lateral
    corrected = rough_targets.copy()
    for side, sign in (("left", 1.0), ("right", -1.0)):
        ankle_pitch = indices[f"{side}_ankle_pitch_joint"]
        hip_pitch = indices[f"{side}_hip_pitch_joint"]
        ankle_roll = indices[f"{side}_ankle_roll_joint"]
        hip_roll = indices[f"{side}_hip_roll_joint"]
        corrected[ankle_pitch] = corrected.get(ankle_pitch, 0.0) + sagittal
        corrected[hip_pitch] = corrected.get(hip_pitch, 0.0) - 0.55 * sagittal
        corrected[ankle_roll] = corrected.get(ankle_roll, 0.0) + sign * lateral
        corrected[hip_roll] = corrected.get(hip_roll, 0.0) - sign * 0.45 * lateral
    waist_pitch = indices["waist_pitch_joint"]
    waist_roll = indices["waist_roll_joint"]
    rough_waist_pitch = corrected.get(waist_pitch, 0.0)
    rough_waist_roll = corrected.get(waist_roll, 0.0)
    safe_waist_pitch = float(np.clip(rough_waist_pitch, -0.14, 0.14))
    safe_waist_roll = float(np.clip(rough_waist_roll, -0.10, 0.10))
    corrected[waist_pitch] = (
        (1.0 - balance_weight) * rough_waist_pitch
        + balance_weight * safe_waist_pitch
        + 0.55 * sagittal
    )
    corrected[waist_roll] = (
        (1.0 - balance_weight) * rough_waist_roll
        + balance_weight * safe_waist_roll
        + 0.45 * lateral
    )
    return corrected, com, balance_state


def command_joint_targets(robot: int, targets: dict[int, float]) -> None:
    joints = list(targets)
    positions = []
    forces = []
    for joint in joints:
        info = p.getJointInfo(robot, joint)
        value = targets[joint]
        lower, upper = float(info[8]), float(info[9])
        positions.append(float(np.clip(value, lower, upper)) if lower < upper else value)
        forces.append(max(12.0, 1.25 * float(info[10])))
    p.setJointMotorControlArray(
        robot,
        joints,
        p.POSITION_CONTROL,
        targetPositions=positions,
        targetVelocities=[0.0] * len(joints),
        forces=forces,
        positionGains=[0.30] * len(joints),
        velocityGains=[1.0] * len(joints),
    )


def captured_joint_positions(robot: int) -> list[float]:
    """Return the current position of every joint when T stops the controller."""
    return [p.getJointState(robot, joint)[0] for joint in range(p.getNumJoints(robot))]


def hold_current_joint_state(robot: int, targets: list[float]) -> None:
    """Use the motors as brakes at the posture captured when T was pressed."""
    joints = list(range(p.getNumJoints(robot)))
    p.setJointMotorControlArray(
        robot,
        joints,
        p.POSITION_CONTROL,
        targetPositions=targets,
        forces=[140.0] * len(joints),
        positionGains=[0.45] * len(joints),
    )


def preserve_hand_task(
    robot: int,
    indices: dict[str, int],
    targets: dict[int, float],
    desired_hand_position: np.ndarray,
) -> dict[int, float]:
    """Correct the right arm toward the hand position from the RRT frame."""
    hand = indices[RIGHT_HAND_NAME]
    _, rotation = hand_frame(robot, hand)
    desired_link_origin = desired_hand_position - rotation @ HAND_TARGET_LOCAL
    joints = movable_joints(robot)
    solution = p.calculateInverseKinematics(
        robot,
        hand,
        desired_link_origin,
        jointDamping=[0.08] * len(joints),
        maxNumIterations=40,
        residualThreshold=1.0e-4,
    )
    columns = {joint: column for column, joint in enumerate(joints)}
    corrected = targets.copy()
    for name in ACTIVE_JOINT_NAMES:
        if not name.startswith("right_"):
            continue
        joint = indices[name]
        if joint in columns and columns[joint] < len(solution):
            corrected[joint] = float(solution[columns[joint]])
    return corrected


def generate_plan(seed: int) -> Plan:
    client = p.connect(p.DIRECT)
    if client < 0:
        raise RuntimeError("PyBullet could not open the planning connection.")
    try:
        robot = load_g1()
        indices = joint_indices(robot)
        reset_neutral(robot)
        active_joints = [indices[name] for name in ACTIVE_JOINT_NAMES]
        hand = indices[RIGHT_HAND_NAME]
        table_parts = create_table()
        settings = Settings()
        rng = np.random.default_rng(seed)
        lower, upper = state_bounds(robot, active_joints)
        start = np.zeros(len(active_joints) + 1)
        if not state_is_valid(robot, indices, active_joints, start, table_parts):
            raise RuntimeError("The initial G1 posture is in collision.")

        started = time.perf_counter()
        goal = goal_state(
            robot,
            indices,
            active_joints,
            hand,
            table_parts,
            lower,
            upper,
            settings,
            rng,
        )
        raw_path, explored = basic_rrt(
            robot,
            indices,
            active_joints,
            start,
            goal,
            lower,
            upper,
            table_parts,
            settings,
            rng,
        )
        path = shorten_path(
            robot, indices, active_joints, raw_path, table_parts, settings, rng
        )
        elapsed = time.perf_counter() - started
        trajectory = interpolate(path)
        hand_targets = []
        for state in trajectory:
            if not state_is_valid(robot, indices, active_joints, state, table_parts):
                raise RuntimeError("Final G1 trajectory validation detected a collision.")
            hand_targets.append(hand_position(robot, hand).copy())
        final_error = np.linalg.norm(hand_position(robot, hand) - TARGET_POSITION)
        return Plan(
            trajectory,
            hand_targets,
            explored,
            len(path),
            elapsed,
            float(final_error),
        )
    finally:
        if p.isConnected(client):
            p.disconnect(client)


def print_plan(plan: Plan) -> None:
    print("Robot: Unitree G1")
    print("Planner: basic single-tree RRT")
    print(f"RRT explored nodes: {plan.explored_nodes}")
    print(f"Path nodes after shortening: {plan.path_nodes}")
    print(f"Planning time: {plan.planning_seconds:.3f} s")
    print(f"Animation samples: {len(plan.trajectory)}")
    print(f"Final hand-target error: {plan.final_error:.3f} m")
    print("Result: collision-free rough whole-body motion")
    print("Dynamic test: gravity on, free base, CoM-Reference Controller active during playback")


def run(gui: bool = True, seed: int = 6) -> None:
    plan = generate_plan(seed)
    print_plan(plan)

    mode = p.GUI if gui else p.DIRECT
    options = "--width=900 --height=700" if gui else ""
    client = p.connect(mode, options=options)
    if client < 0:
        raise RuntimeError("PyBullet could not open the presentation window.")
    try:
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.resetSimulation()
        p.setGravity(0.0, 0.0, -9.81)
        p.setTimeStep(TIME_STEP)
        p.setPhysicsEngineParameter(
            fixedTimeStep=TIME_STEP,
            numSolverIterations=150,
            enableConeFriction=1,
        )
        p.setRealTimeSimulation(0)
        plane = p.loadURDF("plane.urdf")
        p.changeDynamics(
            plane,
            -1,
            lateralFriction=1.3,
            spinningFriction=0.08,
            rollingFriction=0.1,
            restitution=0.0,
        )
        robot = load_g1(fixed_base=False)
        indices = joint_indices(robot)
        reset_neutral(robot)
        active_joints = [indices[name] for name in ACTIVE_JOINT_NAMES]
        table_parts = create_table()
        create_target()
        reset_dynamic_state(
            robot, indices, active_joints, plan.trajectory[0]
        )
        for link in range(-1, p.getNumJoints(robot)):
            p.changeDynamics(
                robot,
                link,
                lateralFriction=1.3,
                spinningFriction=0.08,
                rollingFriction=0.1,
                restitution=0.0,
                linearDamping=0.0,
                angularDamping=0.0,
            )
        if gui:
            p.resetDebugVisualizerCamera(
                cameraDistance=2.20,
                cameraYaw=112.0,
                cameraPitch=-14.0,
                cameraTargetPosition=(0.26, 0.0, 0.34),
            )

        elapsed = 0.0
        paused = False
        joints_locked = False
        locked_joint_targets: list[float] = []
        balance_state = BalanceState()
        support_reference: np.ndarray | None = None
        motion_progress = 0.0
        motion_limited = False
        maximum_com_error = 0.0
        table_contact_seen = False
        duration = Settings().animation_seconds
        next_diagnostic = 0.0
        headless_duration = SETTLE_SECONDS + duration + 1.0
        while p.isConnected(client) and (gui or elapsed < headless_duration):
            keys = p.getKeyboardEvents() if gui else {}
            if keys.get(ord("q"), 0) & p.KEY_WAS_TRIGGERED:
                break
            if keys.get(ord("e"), 0) & p.KEY_WAS_TRIGGERED:
                paused = not paused
            if (
                keys.get(ord("t"), 0) & p.KEY_WAS_TRIGGERED
                and not joints_locked
            ):
                joints_locked = True
                locked_joint_targets = captured_joint_positions(robot)
            if keys.get(ord("r"), 0) & p.KEY_WAS_TRIGGERED:
                reset_neutral(robot)
                reset_dynamic_state(
                    robot, indices, active_joints, plan.trajectory[0]
                )
                elapsed = 0.0
                maximum_com_error = 0.0
                paused = False
                joints_locked = False
                locked_joint_targets = []
                balance_state = BalanceState()
                support_reference = None
                motion_progress = 0.0
                motion_limited = False
            if not paused:
                if joints_locked:
                    hold_current_joint_state(robot, locked_joint_targets)
                    com, _ = center_of_mass(robot)
                    support = (
                        support_reference
                        if support_reference is not None
                        else support_center(robot, indices)
                    )
                else:
                    if support_reference is None and elapsed >= SETTLE_SECONDS:
                        support_reference = support_center(robot, indices)

                    if support_reference is None:
                        targets = state_joint_targets(
                            robot, indices, active_joints, plan.trajectory[0]
                        )
                        com, _ = center_of_mass(robot)
                        support = support_center(robot, indices)
                    else:
                        com_before, _ = center_of_mass(robot)
                        support_error = float(
                            np.linalg.norm((com_before - support_reference)[:2])
                        )
                        clearance = table_clearance(robot, table_parts)
                        can_advance = (
                            support_error < MAXIMUM_SUPPORT_ERROR
                            and clearance >= TABLE_SAFETY_DISTANCE
                        )
                        if can_advance:
                            motion_progress = min(
                                duration, motion_progress + TIME_STEP
                            )
                        else:
                            motion_limited = True
                            motion_progress = max(
                                0.0, motion_progress - 2.0 * TIME_STEP
                            )

                        fraction = min(1.0, motion_progress / duration)
                        frame = min(
                            int(fraction * (len(plan.trajectory) - 1)),
                            len(plan.trajectory) - 1,
                        )
                        rough_targets = state_joint_targets(
                            robot, indices, active_joints, plan.trajectory[frame]
                        )
                        balance_weight = min(
                            1.0,
                            max(0.0, elapsed - SETTLE_SECONDS)
                            / BALANCE_RAMP_SECONDS,
                        )
                        targets, com, balance_state = auto_balance_targets(
                            robot,
                            indices,
                            rough_targets,
                            support_reference,
                            balance_state,
                            balance_weight,
                        )
                        support = support_reference
                        # With waist tilt already bounded by the balance loop,
                        # solve only the right arm to preserve the RRT hand path.
                        targets = preserve_hand_task(
                            robot, indices, targets, plan.hand_targets[frame]
                        )
                    command_joint_targets(robot, targets)
                p.stepSimulation()
                table_contact_seen = table_contact_seen or any(
                    p.getContactPoints(bodyA=robot, bodyB=part)
                    for part in table_parts
                )
                maximum_com_error = max(
                    maximum_com_error,
                    float(np.linalg.norm((com - support)[:2])),
                )
                if not gui and elapsed >= next_diagnostic:
                    base_position, base_orientation = p.getBasePositionAndOrientation(robot)
                    roll, pitch, _ = p.getEulerFromQuaternion(base_orientation)
                    print(
                        f"t={elapsed:4.1f} s | pelvis={base_position[2]:.3f} m | "
                        f"CoM error={np.linalg.norm((com - support)[:2]):.3f} m | "
                        f"roll={np.rad2deg(roll):.1f} deg | pitch={np.rad2deg(pitch):.1f} deg | "
                        f"motion={motion_progress:.3f} s"
                    )
                    next_diagnostic += 1.0
                elapsed += TIME_STEP
            if gui:
                time.sleep(TIME_STEP)
    finally:
        if "maximum_com_error" in locals():
            base_position, base_orientation = p.getBasePositionAndOrientation(robot)
            roll, pitch, _ = p.getEulerFromQuaternion(base_orientation)
            upright = (
                base_position[2] > 0.30
                and abs(roll) < np.deg2rad(35.0)
                and abs(pitch) < np.deg2rad(35.0)
                and not table_contact_seen
            )
            print(f"Maximum horizontal CoM error: {maximum_com_error:.3f} m")
            print(f"Final pelvis height: {base_position[2]:.3f} m")
            dynamic_hand_error = np.linalg.norm(
                hand_position(robot, indices[RIGHT_HAND_NAME]) - TARGET_POSITION
            )
            dynamic_hand = hand_position(robot, indices[RIGHT_HAND_NAME])
            print(f"Dynamic final hand-target error: {dynamic_hand_error:.3f} m")
            print(
                "Dynamic final hand position: "
                f"({dynamic_hand[0]:.3f}, {dynamic_hand[1]:.3f}, {dynamic_hand[2]:.3f}) m"
            )
            print(f"Table contact during playback: {'yes' if table_contact_seen else 'no'}")
            print(f"Motion limited for balance: {'yes' if motion_limited else 'no'}")
            print(f"Balance result: {'upright' if upright else 'lost balance'}")
        if p.isConnected(client):
            p.disconnect(client)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--seed", type=int, default=8)
    args = parser.parse_args()
    run(gui=not args.headless, seed=args.seed)


if __name__ == "__main__":
    main()

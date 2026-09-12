"""Generate a rough G1 whole-body motion with a basic single-tree RRT.

The G1 reaches beneath a table while a single rapidly exploring random tree
searches a 15-dimensional posture vector: one coordinated squat variable,
three waist joints, seven right-arm joints, and four left-arm joints. Every node and
interpolated edge must respect joint limits and avoid self-collision and table
collision, including the visible hand mesh omitted by the G1 URDF's
collision model.

This is rough geometric motion generation only. Gravity, forces, ZMP, contact
dynamics, actuator limits, and dynamic stability are not evaluated.

Controls:
    E: pause or resume.
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
SQUAT_HIP = -1.35
SQUAT_KNEE = 2.22
SQUAT_ANKLE = -0.87

TABLETOP_HEIGHT = 0.50
TARGET_POSITION = np.array((0.43, -0.22, 0.32))

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

# The G1 hand link has a visual mesh but no collision element.
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
    max_iterations: int = 12000
    goal_bias: float = 0.18
    goal_tolerance: float = 0.055
    animation_seconds: float = 7.0


@dataclass
class Plan:
    trajectory: list[np.ndarray]
    explored_nodes: int
    path_nodes: int
    planning_seconds: float
    final_error: float


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
        base_position[0] - ankle_center[0],
        base_position[1],
        base_position[2] + GROUND_CLEARANCE - sole_z,
    )
    p.resetBasePositionAndOrientation(robot, corrected, (0.0, 0.0, 0.0, 1.0))


def load_g1() -> int:
    if not G1_URDF.exists():
        raise FileNotFoundError(f"G1 URDF not found: {G1_URDF}")
    p.setAdditionalSearchPath(str(G1_URDF.parent))
    return p.loadURDF(
        str(G1_URDF),
        basePosition=(0.0, 0.0, STARTING_BASE_HEIGHT),
        useFixedBase=True,
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
            basePosition=(0.72, 0.0, TABLETOP_HEIGHT),
        )
    )

    leg_half = (0.03, 0.03, (TABLETOP_HEIGHT - top_half[2]) / 2.0)
    leg_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=leg_half)
    leg_visual = p.createVisualShape(
        p.GEOM_BOX, halfExtents=leg_half, rgbaColor=(0.34, 0.16, 0.05, 1.0)
    )
    for x, y in ((0.46, -0.35), (0.46, 0.35), (0.98, -0.35), (0.98, 0.35)):
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

    # Solve the palm-point task with a finite-difference Jacobian over only the
    # torso and right-arm coordinates represented by the RRT state.
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
        for state in trajectory:
            if not state_is_valid(robot, indices, active_joints, state, table_parts):
                raise RuntimeError("Final G1 trajectory validation detected a collision.")
        final_error = np.linalg.norm(hand_position(robot, hand) - TARGET_POSITION)
        return Plan(trajectory, explored, len(path), elapsed, float(final_error))
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
    print("Dynamic stability: not evaluated")


def run(gui: bool = True, seed: int = 8) -> None:
    plan = generate_plan(seed)
    print_plan(plan)
    if not gui:
        return

    client = p.connect(p.GUI, options="--width=900 --height=700")
    if client < 0:
        raise RuntimeError("PyBullet could not open the presentation window.")
    try:
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0.0, 0.0, 0.0)
        p.setRealTimeSimulation(0)
        p.loadURDF("plane.urdf")
        robot = load_g1()
        indices = joint_indices(robot)
        reset_neutral(robot)
        active_joints = [indices[name] for name in ACTIVE_JOINT_NAMES]
        create_table()
        create_target()
        p.resetDebugVisualizerCamera(
            cameraDistance=1.65,
            cameraYaw=62.0,
            cameraPitch=-17.0,
            cameraTargetPosition=(0.34, 0.0, 0.55),
        )

        frame = 0
        paused = False
        delay = Settings().animation_seconds / len(plan.trajectory)
        while p.isConnected(client):
            keys = p.getKeyboardEvents()
            if keys.get(ord("q"), 0) & p.KEY_WAS_TRIGGERED:
                break
            if keys.get(ord("e"), 0) & p.KEY_WAS_TRIGGERED:
                paused = not paused
            if keys.get(ord("r"), 0) & p.KEY_WAS_TRIGGERED:
                reset_neutral(robot)
                frame = 0
                paused = False
            if not paused:
                apply_state(
                    robot,
                    indices,
                    active_joints,
                    plan.trajectory[min(frame, len(plan.trajectory) - 1)],
                )
                frame = min(frame + 1, len(plan.trajectory) - 1)
            time.sleep(delay)
    finally:
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

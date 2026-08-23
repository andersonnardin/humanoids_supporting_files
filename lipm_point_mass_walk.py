"""Run a physical multi-step LIPM experiment with a point mass and massless leg."""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np

try:
    import pybullet as p
    import pybullet_data
except ImportError as error:  # pragma: no cover - learner environment check
    raise SystemExit("Install PyBullet first: python3 -m pip install pybullet") from error


G = 9.81
LEG_COLORS = ((0.10, 0.36, 0.86), (0.90, 0.14, 0.10))


@dataclass(frozen=True)
class LIPMSettings:
    """Parameters shared by the point mass and every support phase."""

    mass: float = 5.0
    com_height: float = 0.80
    support_time: float = 0.75
    position_weight: float = 10.0
    velocity_weight: float = 1.0
    step_lengths: tuple[float, ...] = (0.0, 0.3, 0.3, 0.3, 0.0)
    step_widths: tuple[float, ...] = (0.2, 0.2, 0.2, 0.2, 0.2)
    # A fine fixed step keeps repeated physical support exchanges close to the
    # analytical hyperbolic solution rather than accumulating integrator error.
    time_step: float = 1.0 / 2000.0

    @property
    def time_constant(self) -> float:
        return np.sqrt(self.com_height / G)

    @property
    def phase_steps(self) -> int:
        return round(self.support_time / self.time_step)


def nominal_foot_placements(settings: LIPMSettings) -> np.ndarray:
    """Return alternating desired right/left foot centers."""
    lengths = np.asarray(settings.step_lengths)
    widths = np.asarray(settings.step_widths)
    feet = np.zeros((len(lengths) + 1, 2))
    feet[0] = (0.0, -widths[0] / 2.0)
    for step, (length, width) in enumerate(zip(lengths, widths), start=1):
        feet[step] = feet[step - 1] + (length, width if step % 2 else -width)
    return feet


def primitive_target(
    feet: np.ndarray, phase: int, settings: LIPMSettings
) -> tuple[np.ndarray, np.ndarray]:
    """Return the terminal CoM state of one symmetric LIPM primitive."""
    c = np.cosh(settings.support_time / settings.time_constant)
    s = np.sinh(settings.support_time / settings.time_constant)
    next_step = feet[min(phase + 1, len(feet) - 1)] - feet[phase]
    midpoint = next_step / 2.0
    velocity = midpoint * np.array(
        (
            (c + 1.0) / (settings.time_constant * s),
            (c - 1.0) / (settings.time_constant * s),
        )
    )
    return feet[phase] + midpoint, velocity


def optimized_support_coordinate(
    initial_position: float,
    initial_velocity: float,
    target_position: float,
    target_velocity: float,
    settings: LIPMSettings,
) -> float:
    """Choose the support coordinate that minimizes terminal state error."""
    c = np.cosh(settings.support_time / settings.time_constant)
    s = np.sinh(settings.support_time / settings.time_constant)
    tc = settings.time_constant
    denominator = (
        settings.position_weight * (c - 1.0) ** 2
        + settings.velocity_weight * (s / tc) ** 2
    )
    position_error = target_position - c * initial_position - tc * s * initial_velocity
    velocity_error = target_velocity - (s / tc) * initial_position - c * initial_velocity
    return (
        -settings.position_weight * (c - 1.0) * position_error
        -settings.velocity_weight * s * velocity_error / tc
    ) / denominator


def propagate_lipm(
    initial_position: np.ndarray,
    initial_velocity: np.ndarray,
    support: np.ndarray,
    settings: LIPMSettings,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the analytical hyperbolic CoM states for one support phase."""
    times = np.linspace(0.0, settings.support_time, settings.phase_steps + 1)
    scaled_time = times / settings.time_constant
    c = np.cosh(scaled_time)[:, None]
    s = np.sinh(scaled_time)[:, None]
    relative_position = initial_position - support
    positions = relative_position * c + settings.time_constant * initial_velocity * s + support
    velocities = relative_position * s / settings.time_constant + initial_velocity * c
    return positions, velocities


def plan_support_exchanges(
    settings: LIPMSettings,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], np.ndarray, np.ndarray]:
    """Plan the supports and the chained LIPM reference segments."""
    desired_feet = nominal_foot_placements(settings)
    current_position = desired_feet[0].copy()
    current_velocity = np.zeros(2)
    supports: list[np.ndarray] = []
    trajectories: list[np.ndarray] = []

    for phase in range(len(desired_feet)):
        target_position, target_velocity = primitive_target(desired_feet, phase, settings)
        support = np.array(
            [
                optimized_support_coordinate(
                    current_position[axis],
                    current_velocity[axis],
                    target_position[axis],
                    target_velocity[axis],
                    settings,
                )
                for axis in range(2)
            ]
        )
        positions, velocities = propagate_lipm(
            current_position, current_velocity, support, settings
        )
        supports.append(support)
        trajectories.append(positions)
        current_position, current_velocity = positions[-1], velocities[-1]

    return (
        desired_feet,
        np.asarray(supports),
        trajectories,
        desired_feet[0],
        np.zeros(2),
    )


def leg_force(
    com_position: np.ndarray, support: np.ndarray, settings: LIPMSettings
) -> np.ndarray:
    """Return the axial massless-leg force required by the LIPM.

    For a support p = (p_x, p_y, 0) and CoM c = (x, y, z_c),

        F_leg = (M g / z_c) (c - p).

    The vertical component is M g, cancelling gravity. The remaining
    horizontal components create the independent LIPM accelerations.
    """
    support_point = np.array((support[0], support[1], 0.0))
    return settings.mass * G / settings.com_height * (com_position - support_point)


def key_was_pressed(keys: dict[int, int], key: str) -> bool:
    """Return true when a lowercase keyboard key was pressed once."""
    return bool(keys.get(ord(key), 0) & p.KEY_WAS_TRIGGERED)


def create_com_body(settings: LIPMSettings, initial_position: np.ndarray) -> int:
    """Create the dynamic sphere that represents the complete CoM mass."""
    collision = p.createCollisionShape(p.GEOM_SPHERE, radius=0.055)
    visual = p.createVisualShape(
        p.GEOM_SPHERE,
        radius=0.055,
        rgbaColor=(0.92, 0.12, 0.08, 1.0),
    )
    return p.createMultiBody(
        baseMass=settings.mass,
        baseCollisionShapeIndex=collision,
        baseVisualShapeIndex=visual,
        basePosition=(*initial_position, settings.com_height),
    )


def reset_point_mass(
    body: int, initial_position: np.ndarray, initial_velocity: np.ndarray, settings: LIPMSettings
) -> None:
    """Restore the first planned CoM state before a new multi-step trial."""
    p.resetBasePositionAndOrientation(
        body,
        (*initial_position, settings.com_height),
        (0.0, 0.0, 0.0, 1.0),
    )
    p.resetBaseVelocity(body, (*initial_velocity, 0.0), (0.0, 0.0, 0.0))


def update_timer(timer_id: int, elapsed_time: float, paused: bool) -> None:
    """Update the GUI timer without creating new labels."""
    state = " (paused)" if paused else ""
    p.addUserDebugText(
        f"simulation time: {elapsed_time:.2f} s{state}",
        (-0.04, -0.32, 0.82),
        (0.08, 0.08, 0.08),
        textSize=1.20,
        replaceItemUniqueId=timer_id,
    )


def run_experiment(*, gui: bool = True) -> np.ndarray:
    """Run the multi-support physical LIPM experiment."""
    settings = LIPMSettings()
    _, supports, _, initial_position, initial_velocity = (
        plan_support_exchanges(settings)
    )
    client = (
        p.connect(p.GUI, options="--width=900 --height=700")
        if gui
        else p.connect(p.DIRECT)
    )
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0.0, 0.0, -G)
    p.setTimeStep(settings.time_step)
    p.setRealTimeSimulation(0)
    if gui:
        p.resetDebugVisualizerCamera(2.4, 43.0, -23.0, (0.55, 0.0, 0.40))
    p.loadURDF("plane.urdf")
    body = create_com_body(settings, initial_position)
    # The ideal LIPM contains no air or joint damping.
    p.changeDynamics(body, -1, linearDamping=0.0, angularDamping=0.0)

    elapsed_time = 0.0
    paused = False
    phase = 0
    phase_step = 0
    completed = False
    previous_position = np.asarray(p.getBasePositionAndOrientation(body)[0])
    timer_id = (
        p.addUserDebugText(
            "simulation time: 0.00 s",
            (-0.04, -0.32, 0.82),
            (0.08, 0.08, 0.08),
            textSize=1.20,
        )
        if gui
        else -1
    )

    while p.isConnected(client):
        keys = p.getKeyboardEvents() if gui else {}
        if gui and key_was_pressed(keys, "r"):
            p.removeAllUserDebugItems()
            reset_point_mass(body, initial_position, initial_velocity, settings)
            elapsed_time = 0.0
            paused = False
            phase = 0
            phase_step = 0
            completed = False
            previous_position = np.asarray(p.getBasePositionAndOrientation(body)[0])
            timer_id = p.addUserDebugText(
                "simulation time: 0.00 s",
                (-0.04, -0.32, 0.82),
                (0.08, 0.08, 0.08),
                textSize=1.20,
            )
        if gui and key_was_pressed(keys, "e"):
            paused = not paused

        if not completed and not paused:
            position = np.asarray(p.getBasePositionAndOrientation(body)[0])
            support = supports[phase]
            p.applyExternalForce(
                body,
                -1,
                leg_force(position, support, settings).tolist(),
                position.tolist(),
                p.WORLD_FRAME,
            )
            p.stepSimulation()
            elapsed_time += settings.time_step
            current_position = np.asarray(p.getBasePositionAndOrientation(body)[0])
            if gui:
                p.addUserDebugLine(
                    previous_position, current_position, (0.95, 0.12, 0.08), lineWidth=2.3
                )
                p.addUserDebugLine(
                    (*support, 0.0),
                    current_position,
                    LEG_COLORS[phase % len(LEG_COLORS)],
                    lineWidth=2.0,
                )
            previous_position = current_position
            phase_step += 1
            if phase_step >= settings.phase_steps:
                phase += 1
                phase_step = 0
                completed = phase >= len(supports)

        if gui:
            update_timer(timer_id, elapsed_time, paused)
            time.sleep(settings.time_step)
        elif completed:
            break

    final_position = np.asarray(p.getBasePositionAndOrientation(body)[0])
    if p.isConnected(client):
        p.disconnect(client)
    return final_position


if __name__ == "__main__":
    run_experiment()

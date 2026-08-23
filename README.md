# Humanoid Walking Support Files

Support files for the Humanoids course, LIPM Walking in a Physics Simulation.

## Contents

- `lipm_point_mass_walk.py`: physical point-mass Linear Inverted Pendulum Model experiment.
- `lipm_ik_toy_biped_walk.py`: multibody toy-biped walking experiment using LIPM foot targets, inverse kinematics, joint motors, and PyBullet contact physics.
- `toy_biped_physics.urdf`: physical toy-biped model used by the multibody simulation.

## Requirements

Python 3 with NumPy and PyBullet installed:

```bash
python -m pip install numpy pybullet
```

Run either Python script from this folder. The multibody script requires `toy_biped_physics.urdf` to remain in the same directory.

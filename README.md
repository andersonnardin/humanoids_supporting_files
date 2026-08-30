# Humanoid Walking Simulation Files

Support files for the Humanoid Masterclass. The examples connect LIPM and Cart-Table walking plans to PyBullet simulations.

## Contents

- `lipm_point_mass_walk.py`: physical point-mass Linear Inverted Pendulum Model experiment.
- `lipm_toy_biped_walk.py`: multibody toy-biped walking experiment using LIPM foot targets, inverse kinematics, joint motors, and PyBullet contact physics.
- `cart_table_toy_biped_walk.py`: offline cart-table walking experiment that defines a desired ZMP sequence, solves the complete CoM trajectory, and commands the multibody toy biped through inverse kinematics and joint motors.
- `lipm_rpo_walk.py`: LIPM walking experiment adapted to the RoboParty RPO multibody humanoid.
- `lipm_rpo_stabilized_walk.py`: the same RPO experiment with a support-ankle torque stabilizer.

- `toy_biped_physics.urdf`: physical toy-biped model used by both multibody simulations.

## Requirements

Python 3 with NumPy and PyBullet installed:

```bash
python -m pip install numpy pybullet
```

Run any Python script from this folder. The two toy-biped scripts require
`toy_biped_physics.urdf` to remain in the same directory.

The RPO scripts require the public `rpo_description` repository beside them,
with `rpo_description/urdf/rpo.urdf` and its `meshes/` directory available.

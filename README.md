# Humanoid Walking Support Files

Support files for the Humanoid Walking course practical simulations. The
examples connect simplified LIPM and cart-table walking plans to physical
PyBullet experiments.

## Contents

- `lipm_point_mass_walk.py`: physical point-mass Linear Inverted Pendulum Model experiment.
- `lipm_toy_biped_walk.py`: multibody toy-biped walking experiment using LIPM foot targets, inverse kinematics, joint motors, and PyBullet contact physics.
- `cart_table_toy_biped_walk.py`: offline cart-table walking experiment that defines a desired ZMP sequence, solves the complete CoM trajectory, and commands the multibody toy biped through inverse kinematics and joint motors.
- `cart_table_g1_walk.py`: offline cart-table walking experiment for the Unitree G1, including G1 inverse kinematics, physical joint-motor execution, multibody CoM measurement, and contact-wrench ZMP measurement.
- `toy_biped_physics.urdf`: physical toy-biped model used by both multibody simulations.

## Requirements

Python 3 with NumPy and PyBullet installed:

```bash
python -m pip install numpy pybullet
```

Run any Python script from this folder. The two toy-biped scripts require
`toy_biped_physics.urdf` to remain in the same directory.

`cart_table_g1_walk.py` additionally requires the Unitree
`g1_description` folder beside the script, with `g1_29dof.urdf` and its
`meshes` directory intact.

# Humanoid Robotics Supporting Files

Support files for the Humanoid Masterclass. The examples cover graphical
whole-body motion design, dynamic stabilization, fall-impact reduction, and
LIPM and Cart-Table walking simulations.

## Contents

- `graphical_motion_designer.py`: dual-view pin/drag editor for designing rough
  whole-body motion as a sequence of stick-figure keyframes.
- `graphical_motion_player_rpo.py`: maps the graphical keyframes to the
  RoboParty RPO, solves approximate whole-body postures, and interpolates them
  in PyBullet.
- `g1_fall_impact.py`: physical PyBullet experiment that
  compares a Chapter 5.5-inspired backward-fall routine with manually
  disabling that controller.
- `auto_balancer_g1_rrt_reach.py`: G1 experiment that
  turns a collision-free RRT reach into gravity-tested playback with a low
  double-support squat and a compact CoM-feedback Auto-Balancer.
- `g1_attention_point_drawing.py`: standalone G1 attention-point experiment
  with a mouse-driven right-hand drawing task, inverse-kinematics arm tracking,
  and a separate filtered CoM-feedback standing controller.
- `rpo_balanced_ball_push.py`: free-standing RoboParty RPO ball-pushing
  experiment with gravity, physical contact, a right-arm damped Jacobian
  controller, and a separate filtered CoM-feedback standing controller.
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

The graphical motion designer uses Tkinter, which is included with many Python
installations and may otherwise be installed through the operating system's
Python Tk package. RPO scripts require the original public `rpo_description`
repository beside them, with `rpo_description/urdf/rpo.urdf` and its `meshes/`
directory available.

# RoboMath

RoboMath is an educational website demonstrating how mathematics underpins robotic motion.
The site contains interactive 2D and 3D simulations of a robotic arm and supporting educational material.

Key topics covered
- Linear algebra (matrices, coordinate transforms, forward/inverse kinematics)
- Multivariate calculus (Jacobians, gradients, optimization)
- Differential equations (simple dynamic models and control concepts)

Included files (tracked in this repository)
- `index.html` — Project landing page
- `2d.html` — 2D robotic arm simulation / explanation
- `3d.html` — 3D robotic arm simulation / explanation
- `math-explorer.html` — Discussion of the underlying mathematics
- `README.md` — This file
 - `LICENSE` — MIT license for the project
 - `.gitignore` — Patterns for files kept out of version control

Notes for running locally
- This is a simple static website. Open `index.html` in a browser to view the content.
- For best results, host with a local static server (optional):

  ```powershell
  # using Python 3
  python -m http.server 8000
  # then open http://localhost:8000 in your browser
  ```

Files kept locally but not committed
- `robotic_arm.py` — simulation / helper script kept locally (not tracked in the repo)
- `robotic_arm_activity.md` and `robotic_arm_simulation.md` — working notes kept locally

License
- This project is released under the MIT License. See the `LICENSE` file for full terms.

Contact
- For questions or contributions, open an issue on the GitHub repo.

Simplified G1 Benchmark (PID)

### What this is
Minimal MuJoCo demo of Unitree G1 upper-body reaching with a PID policy and a safety filter (Safe Set Algorithm). Files are consolidated for clarity.

### Files (category-wise)
- Core (entry + loop)
  - `g1_benchmark_pid/core.py`: `Config`, `BenchmarkTask`, `SimplifiedBenchmark`, `main()`
  - `g1_benchmark_pid/runner.py`: thin wrapper that imports `main` from `core.py`
- Robot (model + kinematics)
  - `g1_benchmark_pid/robot.py`: `G1BasicConfig` (DoFs, limits, frames, mapping) and `G1BasicKinematics` (Pinocchio-based FK/IK)
- Simulation (MuJoCo)
  - `g1_benchmark_pid/sim.py`: `G1BasicMujocoAgent` and rendering utils
- Control (policy + safety)
  - `g1_benchmark_pid/control.py`: `PIDPolicy`, `SafeController` (SafeSetAlgorithm + collision safety index)
- Utilities
  - `g1_benchmark_pid/utils.py`: geometry, colors, distance, logging
- Resources
  - `g1_benchmark_pid/resources/g1/`: MJCF, meshes, scene XMLs

### Run
1) Ensure conda env is active: `conda activate humanoidwalk_extended`
2) From repo root, run:
```bash
python runner.py
```

Environment variables:
- Optionally set `SPARK_G1_RESOURCE_DIR` to point to a folder containing `g1/scene_29dof.xml` and meshes. If unset, code uses `g1_benchmark_pid/resources`.

### Notes
- Logging goes to `log/debug_g1_benchmark` (TensorBoard if tensorboardX is available).
- Keyboard controls (viewer focused): move debug obstacles with arrows, E/Q, SPACE to cycle, PAGE_UP/DOWN to add/remove.


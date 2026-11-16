# Trajectory Tracking

A small project for following planned trajectories, collecting aligned motion
data, and inspecting robot rollouts in MuJoCo.

The core code implements:

- Path projection, a 0.8 m lookahead, and gradual recovery using GT odometry.
- A normal speed command of 0.6 m/s, smooth braking, an 18 cm arrival latch,
  terminal heading alignment, and at least 1.2 s of actual simulated standing.
- Predictive terminal control and comparisons across ten different path shapes.
- Aligned 50 Hz motion, state, action, and motion-token recording.
- Offline arm softening, error reports, robot videos, and local HTML playback.

The current controller backend is **SONIC**, installed separately as an external
dependency. This repository contains our tracking and data tools plus a small
decoder adapter; it does not vendor the controller repository or model weights.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[setup]'
python scripts/setup_sonic.py
```

The setup script downloads the external checkout, the G1 simulation assets, and
the matching release encoder, decoder, and planner. Everything downloaded goes
under the ignored `external/` directory. To reuse an existing installation:

```bash
export SONIC_ROOT=/absolute/path/to/GR00T-WholeBodyControl
```

`SONIC_ROOT` must contain `gear_sonic_deploy/g1/scene_29dof.xml`, release models
under `gear_sonic_deploy/policy/release/`, and the V2 target-velocity planner under
`gear_sonic_deploy/planner/target_vel/V2/`.

## Collect and inspect a trajectory

```bash
python -m trajectory_tracking.collect \
  --terminal-controller predictive --seed 43 --output outputs/s_curve
python -m trajectory_tracking.validate_export outputs/s_curve
python -m trajectory_tracking.render_robot outputs/s_curve
```

For ten different route definitions:

```bash
python -m trajectory_tracking.generate_routes --help
```

Pass a generated JSON file using `collect --path-file <route.json>`. Collection
uses 50 Hz control and 500 Hz physics. A 0.6 m/s command does not imply the robot
achieves exactly that speed; reports use measured GT motion.

## Replay external motion tokens

```bash
python -m trajectory_tracking.run_token_decoder simulate \
  --tokens outputs/s_curve/tokens_50hz.npz \
  --initial-state outputs/s_curve/episode.npz \
  --reference-episode outputs/s_curve/episode.npz \
  --output outputs/replay
```

Replay uses only the decoder, the recorded tokens, and the first initial state.
All subsequent feedback comes from the new physical simulation. See the
[decoder interface](docs/token_decoder.md) for the observation and joint-order
contracts, and [experiment details](docs/experiments.md) for collection modes,
exports, stopping logic, and prior results.

## Tests

```bash
python -m unittest discover -s tests -p 'test_*.py'
```

Decoder tests require the external release model; tests using recorded episodes
are skipped when those local fixtures are unavailable. Generated datasets,
videos, downloaded dependencies, and virtual environments are excluded from Git.

## External dependency attribution

The SONIC backend and G1 resources come from
[NVIDIA GR00T-WholeBodyControl](https://github.com/NVlabs/GR00T-WholeBodyControl).
They remain separate dependencies under their upstream licenses. Adapter
provenance and the applicable upstream notice are recorded in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

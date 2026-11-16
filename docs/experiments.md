# SONIC trajectory collection — v2

Run the official SONIC release encoder (mode 0), decoder and V2 target-velocity
planner in free-base G1 MuJoCo. No hardware, elastic support, root forces or root
teleportation are used. The original simulation stays untouched by arm processing.

## Path tracking and stopping

The 24 m longitudinal S path has opposite 2 m bends, straight run-in/out and
26.04 m arc length. Initial lateral offset is 0.35 m. Every 20 ms, GT root XY is
projected onto nearby original path segments. The target is 0.8 m farther along
arc length; a direction slew limit permits gradual recovery.

Normal planner speed command is 0.6 m/s after a startup acceleration ramp.
At Euclidean goal distance 1.20 m, the command follows:

```python
u = clip((1.20 - distance) / (1.20 - 0.30), 0, 1)
speed = 0.15 + (0.60 - 0.15) * 0.5 * (1 + cos(pi * u))
```

This has zero endpoint slope and reaches 0.15 m/s at the 0.30 m arrival radius.
The first distance <= 0.30 m latches `position_arrived`, permanently. After this,
there is no goal-position pursuit, even if inertia carries the robot outside.
The desired facing is the planned final path tangent. The actual GT root yaw
error is wrapped to [-pi, pi]. If its magnitude exceeds 15 degrees, the planner
receives zero movement direction with final facing direction (in-place alignment).
Once actual yaw error is <= 15 degrees, the system latches standing mode 0.

The simulation then continues for at least 1.20 s, running fresh SONIC inference
and physics at each tick. Completion additionally requires final heading <= 15 deg
and max horizontal speed < 0.12 m/s in the last 0.30 s. If stabilization takes
longer, continue simulating. No hold frames are copied. A fallen/nonfinite rollout
is diagnostic only and is not marked successful.

Speed commands go directly to the SONIC planner; no speed PI correction is active
in v2. A commanded 0.6 m/s does not imply an achieved speed of exactly 0.6 m/s;
report GT along-path speed and horizontal speed magnitude separately.

## Offline arm export

`arm_export.py` runs after collection and is never called by the control loop.
The natural-stand baseline is `DEFAULT` from upstream policy parameters: shoulder
pitch 0.2, shoulder roll +0.2/-0.2, elbow 0.6, remaining arm joints zero (radians).
For each arm joint:

```python
q_export = q_stand + 0.20 * clip(q_raw - q_stand, -limit, limit)
```

Pre-scale limits, in radians:

| Joint | Limit |
| --- | --- |
| Shoulder pitch / roll / yaw | 0.45 / 0.30 / 0.35 |
| Elbow | 0.50 |
| Wrist roll / pitch / yaw | 0.25 / 0.20 / 0.20 |

Root position/orientation, waist, legs, original actions and original SONIC tokens
are unchanged. Arm joint velocities are recomputed from processed joint angles;
link FK and world velocities are recomputed for the processed export. The video
shows raw simulated motion and the processed pose replay side by side. The latter
is offline pose playback, not a new physics rollout with constrained arms.

Tokens remain the original action labels for the raw simulation. They are not
newly encoded from the arm-processed motion. Both motions are retained explicitly.

## Reproduce

From the repository root, with dependencies in `requirements.lock.txt` and official
ONNX models in the external SONIC checkout (`$SONIC_ROOT/gear_sonic_deploy/`):

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 python -m trajectory_tracking.collect --output outputs/s_curve_v2
python -m trajectory_tracking.validate_export outputs/s_curve_v2
python -m trajectory_tracking.visualize outputs/s_curve_v2 --video
# Extract a desired frame to preview.png with imageio before HTML bundling.
python -m trajectory_tracking.export_html outputs/s_curve_v2 --web-directory outputs/s_curve_v2/web
.venv/bin/python -m unittest discover -s tests -p 'test_*.py'
```

Planner runs at 10 Hz simulated time. Its 30 Hz predictions are resampled to
50 Hz using quaternion slerp and crossfaded over eight frames, two frames ahead,
following upstream. SONIC inference and collection run at 50 Hz; PD/physics at
500 Hz. CPU inference runs as fast as possible, not wall-clock synchronized.

## Artifacts and alignment

Each tick has `t[k] = k / 50`. GT state at k is before `action[k]` drives the next
20 ms; `token[k]` is exactly the encoder output passed to the decoder at k.

| Artifact | Contents |
| --- | --- |
| `episode.npz` | Raw state, actions, reference, tokens, observations, commands, phase, goal distance, actual heading error and position latch |
| `motion_50hz.npz` | Unmodified simulated GT motion, link FK, velocities and names |
| `motion_arms_50hz.npz` | Arm-only processed motion, same root/waist/legs, recomputed arm velocities and link FK |
| `tokens_50hz.npz` | Original aligned 64-D SONIC tokens and encoder mode 0 |
| `arm_export.json` | Limits, baseline, retention, provenance and arm deviation metrics |
| `summary.json` | Stop metrics, timing, actual speed, validation and model SHA256 hashes |
| `tracking.png` | Path, tracking, speed, goal distance/heading, arm comparison and standing |
| `rollout.mp4` | 25 FPS raw/processed side-by-side replay with GT path |
| `sonic-s-curve.html` | Single offline file with embedded video, synchronized plots and data downloads |
| `web/` | Equivalent smaller HTML with separate local assets for LAN hosting |

`qpos = [world_xyz, root_quat_wxyz, 29 MuJoCo-order joints]`.
`qvel = [world_linear, root_local_angular, 29 joint velocities]`.
World link motion excludes the world body; explicit body/joint names and an
IsaacLab joint-order permutation are included. Policy actions/proprioception use
IsaacLab order. Stable tracking metrics exclude the first 12 s and braking.

Validation checks 50 Hz alignment, finite values, unit quaternions, token/input
alignment, 21 fresh encoder/decoder replays, cosine formula, first-arrival latch,
actual final heading, simulated hold duration and stable final velocity. It also
asserts raw/export root/waist/legs equality and corresponding body FK equality.

## 18 cm arrival-radius rerun

The current default is `--arrival-radius 0.18`, with output under
`outputs/s_curve_v3`. Only the arrival trigger changes: the cosine brake still
ends at 0.30 m and then holds a 0.15 m/s command until entry into 0.18 m.
Use `--arrival-radius 0.30` to reproduce v2.

This rerun timed out at 100 s without entering 18 cm. Minimum goal distance was
23.83 cm; final goal distance about 24.75 cm. The position latch and standing mode
were never triggered. `validate_export.py --allow-incomplete` exports this as
diagnostic data with `training_success=false`; data integrity passing does not
mean task completion. The report handles untriggered arrival/idle timestamps.
`export_html.py --compare outputs/s_curve_v2` includes the previous comparison.

## Repeated 18 cm trials

`collect.py --seed <integer>` controls the planner random-seed sequence (seed +
replan call index); default 42 reproduces the earlier trials. Six additional runs
in `outputs/batch_18cm/trial-43` through `trial-48` keep path, state and control
parameters fixed. Each has full 50 Hz raw state, tokens and observations. Both
raw and arm-processed motion are exported after validation.

```bash
python -m trajectory_tracking.batch_report outputs/batch_18cm
```

This validates every trial, exports motion, and writes a local batch dashboard,
CSV, JSON and diagnostic PNG under `outputs/batch_18cm/web`. Chart display is
10 Hz; extrema, outcomes, summary statistics and training files use original
50 Hz records. The server permits named report assets under `trial-<seed>/` and
`previous/` without directory listings. Trials that never entered the threshold
remain diagnostics (`training_success=false`).

In these six trials, seeds 44 and 46 entered 18 cm and completed 1.2 s simulated
standing. Their final distances are 14.45 and 15.64 cm. The other four timed out
without entering the radius. Final-distance median is 20.33 cm, range 14.45–26.56
cm. This is a seed comparison on one path and does not establish general success
rates across paths or robots.

## SONIC waypoint interface experiment

Official reference in the external SONIC checkout: `docs/source/references/planner_onnx.md`. The ONNX input
`has_specific_target=1` enables `specific_target_positions [1,4,3]` and
`specific_target_headings [1,4]`, in the planner's world coordinate frame.
These are four frames of a target token, not four independent sparse route nodes.
A direct fixed-input test confirmed changing this flag changes generated poses;
shifting target Y by 0.4 m also changes generated lateral motion.

Run `collect.py --planner-mode waypoints --seed <seed>`. This experiment fixes
prediction to 11 tokens (44 frames at 30 Hz). Path target samples are placed at
`progress + command_speed * [40,41,42,43]/30`, capped at path end. At 0.6 m/s,
the first sample is 0.8 m ahead. During cosine braking the target distance shrinks
with speed. Targets retain nominal pelvis height 0.788740 m, and heading follows
the path tangent. Planner reference XY is translated to match GT root XY at each
replan; physical qpos/qvel are never overwritten. Idle and in-place heading
alignment use the original non-waypoint mode. This is a particular integration
of the interface; the model's targets are not guaranteed exact physical poses.

Per-replan context, actual waypoint flag, target positions/headings and prediction
length are saved in `episode.npz`. Validation asserts waypoint calls really use
flag=1, predictions are 44 frames and target height is correct; encoder/decoder
replay and raw/export consistency checks remain in place.

Six trials under `outputs/batch_waypoints_18cm/trial-43` through `trial-48` use the
same path, initial state, seeds and 18 cm arrival criteria as the velocity-mode
batch. Use:

```bash
python -m trajectory_tracking.batch_report outputs/batch_waypoints_18cm --baseline outputs/batch_18cm
```

Paired middle metrics use the same physical path interval, progress 5 m through
2 m before the endpoint, excluding braking. This is not a speed-matched ablation:
velocity mode uses a variable 24–44-frame horizon, waypoint mode a fixed 44-frame
horizon and GT reference-coordinate alignment. Achieved speeds are reported.
Waypoint mode completed 0/6 arrivals; final-distance median 40.07 cm versus 20.33
cm for velocity mode. Common-path RMSE medians are 9.11 and 7.01 cm, respectively.
These results describe this integration and configuration, not all waypoint uses.

## Velocity-mode terminal precision improvement

`collect.py --terminal-controller predictive --seed <seed>` selects
`PrecisionTracker`; default `legacy` preserves earlier experiments. Normal S-path
tracking and all model parameters are unchanged until goal distance < 0.55 m.
Within 0.55 m, the command is floored at 0.28 m/s, because the original 0.15 m/s
command frequently stopped making forward progress. The original cosine brake
still controls the rest of the approach.

First entry into the 18 cm radius remains permanently latched. Instead of always
switching immediately to idle, the terminal controller estimates final longitudinal
position using GT closing speed, low-pass filtered with 0.20 s time constant:

```python
predicted_remaining = remaining_along_final_tangent - 0.25 * max(0, filtered_speed)
```

At predicted remaining <= 0.045 m, or passing the goal, or after at most 1.0 s
since position latch, the controller ends the straight terminal motion. Actual
heading must still satisfy 15 degrees before idle is latched. The terminal stage
uses the final path heading and never turns back to pursue the goal. Standing
continues as real SONIC/physics steps for at least 1.2 s, with the original final
velocity and heading acceptance criteria. No waypoint inputs, root forces, root
position edits or online arm constraints are used. `phase=5` identifies the
bounded terminal stage. Speed/phase commands and all raw tokens remain recorded.

Under `outputs/batch_precision`, seeds 43–48 are paired with the existing velocity
batch; seeds 49–52 are four additional trials. All 10 completed and finished within
18 cm. Final errors range 1.61–11.29 cm, median 6.99 cm. For the original six only,
final-error median improves from 20.33 to 7.08 cm and completion from 2/6 to 6/6.
Raw motion and tokens are bit-identical to the baseline up to the first time each
run reaches 0.55 m. These are empirical results on the same simulated S trajectory,
not a guarantee on arbitrary paths or real hardware.

```bash
python -m trajectory_tracking.batch_report outputs/batch_precision --baseline outputs/batch_18cm
```

The dashboard serves under `/precision/` on the existing LAN service. Its replay
shows the run with the largest final error (Seed 50), rather than only the best run.
Per-run 50 Hz raw/arm-processed motion, full state/observation records and original
SONIC tokens are downloadable. Validation additionally checks the modified speed
floor, bounded terminal-stage duration and irreversible position latch.

## Additional 10 unchanged-parameter trials (Seed 53–62)

`outputs/batch_precision_extra10` contains 10 additional runs with the same
velocity-mode predictive terminal controller, path and initial state as the prior
10. The collector source SHA256 and seed list are recorded in `run_manifest.json`;
the collector was not changed or tuned during this batch.

```bash
python -m trajectory_tracking.batch_report outputs/batch_precision_extra10 --previous-cohort outputs/batch_precision
```

The report separately shows this cohort, the previous 10 and the combined 20.
All 10 additional runs entered 18 cm, completed heading/standing, and finished
inside 18 cm. New final-error median is 7.21 cm, mean 8.19 cm, range 4.97–13.15 cm.
Combined 20-run median is 7.17 cm, mean 7.35 cm, range 1.61–13.15 cm, with 20/20
completion on this same simulated path. The new batch is hosted at
`/precision-extra10/`; its replay is the maximum-error run, Seed 60.

## Ten distinct geometries (not seed repetitions)

The user's corrected request is evaluated in `outputs/distinct_trajectories`.
`generate_routes.py` creates 10 geometrically distinct reference paths: narrow,
mirrored, wide and gentle S curves; a double S; left/right lane changes;
left/right 90-degree corners; and a U-turn. All start along +X, while their
terminal headings include 0, +90, -90 and 180 degrees. Paths are resampled at
approximately 5 mm arc spacing. Every run uses the same planner seed 43 and
unchanged predictive velocity-mode terminal parameters, initial 0.35 m offset,
0.6 m/s nominal command and 0.8 m lookahead.

```bash
python -m trajectory_tracking.generate_routes outputs/distinct_trajectories/routes
python -m trajectory_tracking.collect --path-file outputs/distinct_trajectories/routes/route-01.json --output outputs/distinct_trajectories/trial-01 --seed 43 --terminal-controller predictive
python -m trajectory_tracking.routes_report outputs/distinct_trajectories
python -m trajectory_tracking.visualize outputs/distinct_trajectories/trial-01 --video --video-fps 10
```

Repeat collection and rendering for route/trial 01 through 10. `--path-file`
accepts JSON with `points [N,2]`, `name` and optional `id`; default path generation
is unchanged when omitted. Path projection handles non-X-monotone geometry,
including U-turns. The terminal controller uses the final path tangent for GT
closing velocity and stopping prediction, and physical robot root states are
never overwritten. Tests cover rotated terminal headings and nonmonotone paths.

The LAN report `/different-paths/` includes 10 path thumbnails, per-path metrics,
all ten complete raw-versus-arm-export videos, synchronized plots, and separate
50 Hz motion/token/full-record downloads. Display/video are 10 Hz/FPS. Middle
metrics use phase 1 and progress from min(5 m, 25% of path length) through 2 m
before the endpoint. All 10 completed real heading/standing acceptance; final
error median is 8.61 cm, range 6.22–10.16 cm. Middle RMSE ranges 4.59–11.75 cm;
the double-S run reaches a maximum instantaneous lateral error of 28.75 cm.
`routes_results.json` and `web/batch.csv` retain all per-path metrics.

## Robot-first walking view

`render_robot.py <episode-directory>` renders an uncropped, robot-only 1280x720
25 FPS video directly from the original physically simulated qpos at every second
50 Hz sample. Camera translation and yaw follow the G1; yellow path capsules and
the green 18 cm arrival ring are visual-only overlays. No arm postprocessing or
synthetic root motion is applied to this view. Ten complete videos are served
at `/robot-walking/`, with muted autoplay starting at 6 s so opening the page shows
walking immediately. Controls permit restarting at 0, slowing playback, switching
all ten geometries, fullscreen and viewing the physical stopping phase.

The player is generated from `robot.template.html` under
`outputs/distinct_trajectories/robot-web`. Browser validation confirmed all ten
videos autoplay on selection and successive rendered robot frames change.

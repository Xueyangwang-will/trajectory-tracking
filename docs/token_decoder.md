# SONIC motion-token decoder

`trajectory_tracking/sonic_decoder.py` independently consumes one 64-dimensional motion token and
the current robot state at **50 Hz**, retaining its own temporal history. It loads
only `policy/release/model_decoder.onnx`. It does not instantiate the SONIC encoder
or planner, and it does not need future reference motions or GT root translation.

This implements the **current SONIC release** input layout: `obs_dict [1,994]`
to `action [1,29]`. Tokens must come from the same checkpoint's latent space.
Tokens are float values, not discrete planner token IDs; do not renormalize them.
Other model variants require matching observation layouts.

## Online API

From the repository root:

```python
from trajectory_tracking.sonic_decoder import SonicDecoder, RobotState, pd_torque

decoder = SonicDecoder()  # Call reset() only at the start of a new episode.

# Once every 20 ms: get the next external motion token and the CURRENT state.
state = RobotState.from_mujoco(data.qpos, data.qvel)
result = decoder.step(motion_token, state)  # motion_token: numpy [64]
joint_target = result.joint_target         # 29 joint angles in MuJoCo order

# Hold that target for ten 2 ms physics ticks (500 Hz).
for _ in range(10):
    data.ctrl[:] = pd_torque(joint_target, data.qpos[7:], data.qvel[6:],
                            model.actuator_ctrlrange)
    mujoco.mj_step(model, data)
```

For a non-MuJoCo state source, construct the state explicitly:

```python
state = RobotState(
    base_quat_wxyz=imu_quaternion,   # [4], world orientation, wxyz
    base_ang_vel=gyro_body,         # [3], BODY frame, rad/s
    joint_pos=measured_joint_pos,   # [29], MuJoCo/Unitree order, rad
    joint_vel=measured_joint_vel,   # [29], same order, rad/s
)
result = decoder.step(motion_token, state)
```

Use a separate decoder instance per robot/environment. Keep the instance across
token chunks; resetting at every chunk destroys the action/state history.
For a PyTorch token, pass `token.detach().cpu().numpy().reshape(64)`.
Input shape/finite-value checks and the model interface are validated. Invalid
input or failed inference leaves history and previous action unchanged.

`DecoderOutput` provides:

| Field | Shape/order | Meaning |
| --- | --- | --- |
| `joint_target` | `[29]`, MuJoCo/Unitree | Target joint angles in radians |
| `action` | `[29]`, IsaacLab | Original unscaled decoder output |
| `observation` | `[994]`, float32 | Exact model input, for logging |

The wrapper applies the upstream per-joint action scale and natural-stand offset.
`pd_torque` uses the upstream Kp/Kd and explicitly supplied actuator torque limits;
the action-scaling effort constants are not assumed to be the actuator limits.
The wrapper does not apply offline arm softening to the executed actions.

## Observation contract

All history blocks are **oldest to newest**, including the current measured
state. The previous-action entry at tick k is the raw policy action from k-1.
The initial measured state is repeated to fill the first 10-frame window, with
zero previous actions, matching the existing collection pipeline.

| Slice | Dimension | Contents |
| --- | --- | --- |
| `0:64` | 64 | Current motion token |
| `64:94` | 30 | 10 body-frame angular velocities |
| `94:384` | 290 | 10 joint positions, default pose subtracted, IsaacLab order |
| `384:674` | 290 | 10 joint velocities, IsaacLab order |
| `674:964` | 290 | 10 previous unscaled policy actions, IsaacLab order |
| `964:994` | 30 | 10 gravity vectors expressed in body frame |

MuJoCo `qpos` is `[world_xyz, quaternion_wxyz, 29 joints]`; `qvel` is
`[world_linear_velocity, body_angular_velocity, 29 joint velocities]`.
Root position and root linear velocity are not inputs to the decoder. The joint
mapping, default angles, PD gains and action scales come from upstream
`src/g1/g1_deploy_onnx_ref/include/policy_parameters.hpp`.

## Run the existing recorded tokens in a fresh simulation

With the prepared repository `.venv` and downloaded release decoder:

```bash
python -m trajectory_tracking.run_token_decoder simulate \
  --tokens outputs/distinct_trajectories/trial-08/tokens_50hz.npz \
  --initial-state outputs/distinct_trajectories/trial-08/episode.npz \
  --reference-episode outputs/distinct_trajectories/trial-08/episode.npz \
  --output outputs/token_decoder/left90
```

Add `--viewer` for a local MuJoCo window; `--realtime` runs the headless loop at
wall-clock 50 Hz. Omit those flags to run as fast as possible. Use
`--model /absolute/path/model_decoder.onnx` **before** the `simulate` subcommand
to choose another compatible model.

Token NPZ files contain `token [N,64]`, optionally `fps=50` (or `frequency_hz=50`)
and `t [N]`. Missing rate/time metadata defaults to a 50 Hz sequence starting at
zero; supplied metadata must match. Tokens are consumed in sequence without
interpolation. `--initial-state` reads only the first qpos/qvel as initialization.
Subsequent policy feedback comes only from the new MuJoCo simulation. Optional
reference comparisons are made after simulation has finished.

Outputs are `episode.npz` and `summary.json`. Each logged state is immediately
before applying the action/token at the same timestamp. `final_qpos/final_qvel`
additionally contain the state after the last token's 20 ms interval. A consumed
sequence means `completed=true`; this flag describes consumption, not an
independent path-arrival check. Existing saved token sequences include their
actual stopping/standing tokens. An all-zero token is **not** defined as a stop
command; this consumer does not synthesize new tokens at end-of-stream.

## Verification

```bash
python -m trajectory_tracking.run_token_decoder verify \
  outputs/distinct_trajectories/trial-01/episode.npz \
  outputs/distinct_trajectories/trial-08/episode.npz \
  outputs/distinct_trajectories/trial-10/episode.npz \
  --output outputs/token_decoder/verify_pilot.json

.venv/bin/python -m unittest discover -s tests -p 'test_sonic_decoder.py'
```

Verification checks every frame's complete decoder input, raw action and scaled
joint target against the existing records. The ten distinct-geometry episodes
pass with zero numerical difference. Separate closed-loop simulations for an
S path, a 90-degree turn and a U-turn reproduce the original recorded robot
states/actions using only external tokens and one initial state. Detailed
reports are saved under `outputs/token_decoder/`.

The trajectory collector now calls this same decoder class, so online collection
and independently supplied-token execution share one observation/history/action
implementation. The decoder itself requires NumPy, SciPy and ONNX Runtime;
the simulation runner additionally requires MuJoCo. It uses no robot SDK or DDS.

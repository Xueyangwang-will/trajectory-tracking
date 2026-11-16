"""Stateful SONIC release ONNX decoder: 64-D tokens + live proprioception -> G1 targets.

No encoder, planner, motion-reference sequence or MuJoCo import is required.
Call ``step`` exactly once per 20 ms control tick; retain this object across token
chunks, and call ``reset`` only at a new episode. All state/target joints are in
MuJoCo/Unitree hardware order. Raw policy actions use IsaacLab order.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import onnxruntime as ort
from scipy.spatial.transform import Rotation

from trajectory_tracking.paths import DEPLOY

CONTROL_HZ = 50
HISTORY_FRAMES = 10
TOKEN_DIM = 64
OBSERVATION_DIM = 994
JOINT_COUNT = 29

# policy[i] corresponds to MuJoCo joint POLICY_TO_MUJOCO[i].
# Upstream source: src/g1/g1_deploy_onnx_ref/include/policy_parameters.hpp.
POLICY_TO_MUJOCO = np.array(
    [0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10,
     16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28]
)
DEFAULT_JOINT_POS = np.array(
    [-.312, 0, 0, .669, -.363, 0] * 2
    + [0, 0, 0, .2, .2, 0, .6, 0, 0, 0, .2, -.2, 0, .6, 0, 0, 0]
)
ARMATURE = np.array(
    [.025101925, .025101925, .010177520, .025101925, .003609725, .003609725] * 2
    + [.010177520, .003609725, .003609725]
    + [.003609725] * 5 + [.00425] * 2 + [.003609725] * 5 + [.00425] * 2
)
EFFORT_FOR_ACTION_SCALE = np.array(
    [139, 139, 88, 139, 25, 25] * 2 + [88, 25, 25] + [25] * 5 + [5] * 2
    + [25] * 5 + [5] * 2
)
KP = ARMATURE * (20 * np.pi) ** 2
KD = 4 * ARMATURE * 20 * np.pi
# Compute action scale BEFORE doubling ankle/waist PD gains, as upstream does.
ACTION_SCALE = .25 * EFFORT_FOR_ACTION_SCALE / KP
KP[[4, 5, 10, 11, 13, 14]] *= 2
KD[[4, 5, 10, 11, 13, 14]] *= 2
for _constant in (POLICY_TO_MUJOCO, DEFAULT_JOINT_POS, ARMATURE,
                  EFFORT_FOR_ACTION_SCALE, KP, KD, ACTION_SCALE):
    _constant.setflags(write=False)


def _vector(value, length: int, name: str, dtype=np.float64) -> np.ndarray:
    result = np.array(value, dtype=dtype, copy=True)
    if result.shape != (length,):
        raise ValueError(f"{name} must have shape ({length},), got {result.shape}")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values")
    return result


@dataclass(frozen=True)
class RobotState:
    """One control-tick observation (SI units).

    base_quat_wxyz: root orientation in world, quaternion order w,x,y,z.
    base_ang_vel: angular velocity expressed in the root/body frame, rad/s.
    joint_pos/joint_vel: 29 joints in MuJoCo/Unitree hardware order, rad and rad/s.
    Root translation and linear velocity are not inputs to this release decoder.
    """
    base_quat_wxyz: np.ndarray
    base_ang_vel: np.ndarray
    joint_pos: np.ndarray
    joint_vel: np.ndarray

    @classmethod
    def from_mujoco(cls, qpos, qvel) -> RobotState:
        qpos = _vector(qpos, 36, "qpos")
        qvel = _vector(qvel, 35, "qvel")
        return cls(qpos[3:7], qvel[3:6], qpos[7:], qvel[6:])


@dataclass(frozen=True)
class DecoderOutput:
    action: np.ndarray  # [29], unscaled action, IsaacLab order
    joint_target: np.ndarray  # [29], radians, MuJoCo/Unitree order
    observation: np.ndarray  # [994], float32 model input, for debugging/logging


class SonicDecoder:
    """Consume one SONIC release token per tick and retain the 10-frame history.

    This wrapper implements the release observation layout, not an arbitrary
    SONIC model variant. The supplied tokens must belong to the same checkpoint.
    The first tick repeats its measured state to fill history, with zero past
    actions; this is the initialization used by the existing collection pipeline.
    Invalid inputs or failed inference do not advance history or previous action.
    """
    def __init__(self, model_path: str | Path | None = None, *,
                 providers: Sequence[str] = ("CPUExecutionProvider",), threads: int = 2):
        self.model_path = Path(model_path) if model_path else (
            DEPLOY / "policy/release/model_decoder.onnx"
        )
        if not self.model_path.is_file():
            raise FileNotFoundError(f"Decoder model not found: {self.model_path}. "
                                    "Set SONIC_ROOT to the external SONIC checkout or pass model_path.")
        if threads < 1:
            raise ValueError("threads must be positive")
        unavailable = set(providers) - set(ort.get_available_providers())
        if not providers or unavailable:
            raise ValueError(f"Unavailable ONNX providers: {sorted(unavailable)}")
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(self.model_path), options, providers=list(providers))
        inputs, outputs = self.session.get_inputs(), self.session.get_outputs()
        if (len(inputs) != 1 or inputs[0].name != "obs_dict"
                or inputs[0].shape != [1, OBSERVATION_DIM] or inputs[0].type != "tensor(float)"
                or len(outputs) != 1 or outputs[0].shape != [1, JOINT_COUNT]
                or outputs[0].type != "tensor(float)"):
            raise ValueError("Expected the SONIC release float32 decoder: obs_dict [1,994] -> [1,29]")
        self.reset()

    def reset(self) -> None:
        self._history = [deque(maxlen=HISTORY_FRAMES) for _ in range(5)]
        self._last_action = np.zeros(JOINT_COUNT)
        self.ticks = 0

    @property
    def last_action(self) -> np.ndarray:
        return self._last_action.copy()

    def step(self, token, state: RobotState) -> DecoderOutput:
        token = _vector(token, TOKEN_DIM, "motion token", np.float32)
        quat = _vector(state.base_quat_wxyz, 4, "base_quat_wxyz")
        if abs(np.linalg.norm(quat) - 1.) > .01:
            raise ValueError("base_quat_wxyz must be a unit quaternion in wxyz order")
        omega = _vector(state.base_ang_vel, 3, "base_ang_vel")
        q = _vector(state.joint_pos, JOINT_COUNT, "joint_pos")
        dq = _vector(state.joint_vel, JOINT_COUNT, "joint_vel")
        orientation = Rotation.from_quat(quat[[1, 2, 3, 0]])
        features = [omega, (q - DEFAULT_JOINT_POS)[POLICY_TO_MUJOCO],
                    dq[POLICY_TO_MUJOCO], self._last_action.copy(),
                    orientation.inv().apply([0., 0., -1.])]
        # Build a candidate state so malformed output/inference failure cannot
        # corrupt temporal alignment if the caller retries this tick.
        histories = []
        for previous, feature in zip(self._history, features):
            history = deque(previous, maxlen=HISTORY_FRAMES)
            if not history:
                history.extend(feature.copy() for _ in range(HISTORY_FRAMES - 1))
            history.append(feature.copy())
            histories.append(history)
        observation = np.concatenate(
            [token] + [np.asarray(h).ravel() for h in histories]
        ).astype(np.float32)
        action = self.session.run(None, {"obs_dict": observation[None]})[0]
        if action.shape != (1, JOINT_COUNT) or not np.isfinite(action).all():
            raise RuntimeError("Decoder returned an invalid joint action")
        action = action[0].copy()
        target = DEFAULT_JOINT_POS.copy()
        target[POLICY_TO_MUJOCO] += action * ACTION_SCALE[POLICY_TO_MUJOCO]
        self._history = histories
        self._last_action = action.copy()
        self.ticks += 1
        return DecoderOutput(action, target, observation)


def pd_torque(joint_target, joint_pos, joint_vel, torque_limits) -> np.ndarray:
    """500 Hz PD torque; limits [29,2] come from the robot/simulation model.

    The effort constants used for action scaling are NOT necessarily the actual
    actuator limits. In MuJoCo pass ``model.actuator_ctrlrange`` explicitly.
    """
    target = _vector(joint_target, JOINT_COUNT, "joint_target")
    q = _vector(joint_pos, JOINT_COUNT, "joint_pos")
    dq = _vector(joint_vel, JOINT_COUNT, "joint_vel")
    limits = np.asarray(torque_limits, dtype=np.float64)
    if limits.shape != (JOINT_COUNT, 2) or not np.isfinite(limits).all():
        raise ValueError("torque_limits must be finite with shape [29,2]")
    if np.any(limits[:, 0] > limits[:, 1]):
        raise ValueError("torque lower limits must not exceed upper limits")
    return np.clip(KP * (target - q) - KD * dq, limits[:, 0], limits[:, 1])

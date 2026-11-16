"""Verify or simulate external SONIC tokens. Loads the decoder model only."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from trajectory_tracking.sonic_decoder import CONTROL_HZ, RobotState, SonicDecoder, pd_torque
from trajectory_tracking.paths import DEPLOY


def load_tokens(path: Path):
    with np.load(path, allow_pickle=False) as source:
        token = np.array(source["token"], dtype=np.float32)
        if token.ndim != 2 or token.shape[1] != 64 or not len(token) or not np.isfinite(token).all():
            raise ValueError("token file must contain a finite nonempty token [N,64] array")
        rate = source["fps"] if "fps" in source else source.get("frequency_hz", CONTROL_HZ)
        if np.asarray(rate).shape != () or float(rate) != CONTROL_HZ:
            raise ValueError("This decoder loop expects 50 Hz tokens")
        timestamps = np.array(source["t"]) if "t" in source else np.arange(len(token)) / CONTROL_HZ
        if (timestamps.shape != (len(token),) or not np.isfinite(timestamps).all()
                or not np.allclose(timestamps, np.arange(len(token)) / CONTROL_HZ, atol=1e-8, rtol=0)):
            raise ValueError("Token file must start at t=0 and use exactly 20 ms intervals")
    return token, timestamps


def verify(args):
    """Measured states are supplied here only for numerical decoder verification."""
    decoder = SonicDecoder(args.model)
    results = []
    for filename in args.episodes:
        with np.load(filename, allow_pickle=False) as z:
            decoder.reset()
            token, _ = load_tokens(filename)
            reference = {key: np.array(z[key]) for key in
                         ["qpos", "qvel", "decoder_obs", "action", "target_joint_pos"]}
            maxima = dict(observation=0., action=0., joint_target=0.)
            start = time.perf_counter()
            for k in range(len(token)):
                result = decoder.step(token[k], RobotState.from_mujoco(reference["qpos"][k], reference["qvel"][k]))
                for key, actual, expected in [
                    ("observation", result.observation, reference["decoder_obs"][k]),
                    ("action", result.action, reference["action"][k]),
                    ("joint_target", result.joint_target, reference["target_joint_pos"][k]),
                ]:
                    maxima[key] = max(maxima[key], float(np.max(abs(actual - expected))))
            elapsed = time.perf_counter() - start
        row = dict(episode=str(filename.resolve()), frames=len(token), max_abs_error=maxima,
                   wall_seconds=elapsed, mean_tick_ms=elapsed / len(token) * 1000,
                   passed=all(v <= args.atol for v in maxima.values()))
        results.append(row)
        print(json.dumps(row), flush=True)
        if not row["passed"]:
            raise AssertionError(f"Decoder parity failed: {row}")
    report = dict(passed=True, loaded_models=[str(decoder.model_path.resolve())], episodes=results)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2))
    return report


def simulate(args):
    """Closed-loop simulation: feedback comes ONLY from the running MuJoCo data."""
    import mujoco
    if args.viewer:
        import mujoco.viewer

    token, timestamps = load_tokens(args.tokens)
    with np.load(args.initial_state, allow_pickle=False) as source:
        q = np.array(source["qpos"])
        dq = np.array(source["qvel"])
        initial_qpos = q[0].copy() if q.ndim == 2 else q.copy()
        initial_qvel = dq[0].copy() if dq.ndim == 2 else dq.copy()
    del q, dq
    RobotState.from_mujoco(initial_qpos, initial_qvel)  # Shape/finite validation.
    model_path = DEPLOY / "g1/scene_29dof.xml"
    model = mujoco.MjModel.from_xml_path(str(model_path))
    if (model.nq, model.nv, model.nu) != (36, 35, 29):
        raise ValueError("Expected the 29-DOF G1 free-base MuJoCo model")
    model.opt.timestep = .002
    data = mujoco.MjData(model)
    data.qpos[:] = initial_qpos
    data.qvel[:] = initial_qvel
    mujoco.mj_forward(model, data)
    decoder = SonicDecoder(args.model)
    logs = {name: [] for name in ["qpos", "qvel", "action", "target_joint_pos", "decoder_obs"]}
    start = time.perf_counter()
    reason = "token_sequence_consumed"
    context = mujoco.viewer.launch_passive(model, data) if args.viewer else nullcontext(None)
    with context as viewer:
        if viewer is not None:
            viewer.cam.distance = 3.
            viewer.cam.azimuth = 135
        for k, current_token in enumerate(token):
            if viewer is not None and not viewer.is_running():
                reason = "viewer_closed"
                break
            tick_start = time.perf_counter()
            # No future reference frames, recorded states or saved actions are
            # read here: all policy feedback comes from this new simulation.
            state = RobotState.from_mujoco(data.qpos, data.qvel)
            result = decoder.step(current_token, state)
            for key, value in zip(logs, [data.qpos.copy(), data.qvel.copy(), result.action,
                                         result.joint_target, result.observation]):
                logs[key].append(value)
            for _ in range(10):
                data.ctrl[:] = pd_torque(result.joint_target, data.qpos[7:], data.qvel[6:],
                                         model.actuator_ctrlrange)
                mujoco.mj_step(model, data)
            if not np.isfinite(data.qpos).all() or data.qpos[2] < .4:
                reason = "fall_or_nonfinite"
                break
            if viewer is not None:
                viewer.cam.lookat[:] = data.qpos[:3]
                viewer.sync()
            if args.realtime or args.viewer:
                time.sleep(max(0., 1 / CONTROL_HZ - (time.perf_counter() - tick_start)))
    n = len(logs["qpos"])
    arrays = {name: np.array(value) for name, value in logs.items()}
    arrays.update(token=token[:n], t=timestamps[:n], fps=np.array(CONTROL_HZ),
                  final_qpos=data.qpos.copy(), final_qvel=data.qvel.copy())
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output / "episode.npz", **arrays)
    report = dict(completed=reason == "token_sequence_consumed", termination_reason=reason,
                  frames=n, supplied_tokens=len(token), control_hz=CONTROL_HZ, physics_hz=500,
                  loaded_models=[str(decoder.model_path.resolve())],
                  decoder_sha256=hashlib.sha256(decoder.model_path.read_bytes()).hexdigest(),
                  tokens_sha256=hashlib.sha256(token.tobytes()).hexdigest(),
                  initial_state=str(args.initial_state.resolve()),
                  minimum_root_height_m=float(min(arrays["qpos"][:, 2].min(), data.qpos[2])),
                  start_xy=initial_qpos[:2].tolist(), final_xy=arrays["qpos"][-1, :2].tolist(),
                  wall_seconds=time.perf_counter() - start)
    # Comparison is performed only AFTER the closed-loop simulation has finished.
    if args.reference_episode:
        with np.load(args.reference_episode, allow_pickle=False) as source:
            if len(source["qpos"]) != n:
                raise ValueError("Reference episode and consumed tokens have different lengths")
            report["reference_max_abs_error"] = {
                key: float(np.max(abs(arrays[key] - source[key])))
                for key in ["qpos", "qvel", "action", "target_joint_pos", "decoder_obs"]
            }
    (args.output / "summary.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    if not report["completed"]:
        raise RuntimeError(f"Token simulation ended early: {reason}")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, help="SONIC release model_decoder.onnx")
    sub = parser.add_subparsers(dest="command", required=True)
    v = sub.add_parser("verify", help="Check all frames against recorded decoder inputs/actions")
    v.add_argument("episodes", nargs="+", type=Path)
    v.add_argument("--atol", type=float, default=1e-5)
    v.add_argument("--output", type=Path)
    s = sub.add_parser("simulate", help="Consume tokens in a fresh closed-loop MuJoCo simulation")
    s.add_argument("--tokens", type=Path, required=True)
    s.add_argument("--initial-state", type=Path, required=True, help="NPZ qpos/qvel; only first state is used")
    s.add_argument("--output", type=Path, required=True)
    s.add_argument("--reference-episode", type=Path, help="Optional post-run numerical comparison")
    s.add_argument("--viewer", action="store_true")
    s.add_argument("--realtime", action="store_true")
    args = parser.parse_args()
    verify(args) if args.command == "verify" else simulate(args)


if __name__ == "__main__":
    main()

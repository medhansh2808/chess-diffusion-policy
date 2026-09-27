import argparse
import json
import os
from collections import deque
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import cv2
import numpy as np
import torch

import abc_sim
from chess_dp import CAMERA_KEYS, ChessDiffusionPolicy


def image_tensor(observation, image_size):
    frames = []
    for camera in CAMERA_KEYS:
        if camera not in observation["images"]:
            raise KeyError(f"Missing {camera} camera; available: {list(observation['images'])}")
        chw = np.asarray(observation["images"][camera])
        if chw.ndim != 3 or chw.shape[0] != 3:
            raise ValueError(f"Unexpected {camera} camera shape: {chw.shape}")
        rgb = np.ascontiguousarray(chw.transpose(1, 2, 0))
        resized = cv2.resize(rgb, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
        frames.append(torch.from_numpy(resized.transpose(2, 0, 1).copy()).float() / 255.0)
    return torch.stack(frames, dim=0)


def top_frame(observation):
    return np.ascontiguousarray(observation["images"]["top"].transpose(1, 2, 0))


def json_safe(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def make_plot(x, lines, title, ylabel, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 4))
    for label, values in lines.items():
        if not any(value is not None for value in values):
            continue
        ax.plot(x, [float("nan") if value is None else value for value in values], label=label)
    ax.set_title(title)
    ax.set_xlabel("Executed actions")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    if len(lines) > 1:
        ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def save_episode_plots(trace, output, episode_index):
    prefix = output / f"episode_{episode_index:03d}"
    x = trace["action"]
    make_plot(x, {"Reward": trace["reward"]}, "Reward during rollout", "Reward", prefix.with_name(prefix.name + "_reward.png"))
    make_plot(x, {"Targets placed": trace["targets_placed"], "Non-target pieces disturbed": trace["non_target_disturbed"]}, "Chessboard progress", "Pieces", prefix.with_name(prefix.name + "_board.png"))
    make_plot(x, {"Left gripper": trace["left_gripper"], "Right gripper": trace["right_gripper"]}, "Predicted gripper commands", "Command (0-1)", prefix.with_name(prefix.name + "_grippers.png"))
    make_plot(x, {"Joint target change": trace["joint_target_change"]}, "Change between consecutive joint target commands", "L2 norm", prefix.with_name(prefix.name + "_action_changes.png"))
    return str(prefix.with_name(prefix.name + "_reward.png"))


def plot_existing_summary(output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = output / "summary.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    episodes = json.loads(path.read_text())["episodes"]
    x = [item["episode"] for item in episodes]
    metrics = {
        "final_reward": ("Final reward", "Reward"),
        "targets_placed": ("Targets placed", "Pieces"),
        "non_target_disturbed": ("Non-target pieces disturbed", "Pieces"),
    }
    for key, (title, ylabel) in metrics.items():
        values = [item.get("final_reward") if key == "final_reward" else (item.get("task_eval") or {}).get(("num_targets_placed" if key == "targets_placed" else "num_non_target_disturbed")) for item in episodes]
        if not any(value is not None for value in values):
            continue
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(x, [0 if value is None else value for value in values])
        ax.set_title(title)
        ax.set_xlabel("Episode")
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        fig.tight_layout()
        target = output / f"summary_{key}.png"
        fig.savefig(target, dpi=160)
        plt.close(fig)
        print(f"Saved: {target}")
    for item in episodes:
        trace_path = output / f"episode_{item['episode']:03d}_trace.json"
        if trace_path.is_file():
            save_episode_plots(json.loads(trace_path.read_text()), output, item["episode"])
            print(f"Saved step-by-step plots for episode {item['episode']}")


def load_policy(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    stats = {key: np.asarray(checkpoint["stats"][key], dtype=np.float32)
             for key in ("state_mean", "state_std", "action_mean", "action_std")}
    for key, value in stats.items():
        if value.shape != (14,):
            raise ValueError(f"Checkpoint {key} has shape {value.shape}, expected (14,)")
    model = ChessDiffusionPolicy(
        observation_horizon=config.get("observation_horizon", 2),
        prediction_horizon=config.get("prediction_horizon", 20),
        state_dim=config.get("state_dim", 14),
        action_dim=config.get("action_dim", 14),
        vision_feature_dim=config.get("vision_feature_dim", 128),
        pretrained_vision=False,
        freeze_backbone=config.get("freeze_backbone", True),
    ).to(device)
    model.load_state_dict(checkpoint["ema"], strict=True)
    model.eval()
    if model.observation_horizon != 2:
        raise ValueError("This rollout script expects observation_horizon=2")
    if model.action_dim != 14:
        raise ValueError("This rollout script expects 14-dimensional actions")
    return model, stats, int(config["image_size"]), int(checkpoint["step"])


@torch.inference_mode()
def predict_chunk(model, stats, history, image_size, device):
    states = np.stack([np.asarray(obs["state"], dtype=np.float32) for obs in history])
    normalized = (states - stats["state_mean"]) / stats["state_std"]
    state_tensor = torch.from_numpy(np.ascontiguousarray(normalized)).unsqueeze(0).to(device)
    images = torch.stack([image_tensor(obs, image_size) for obs in history], dim=0)
    image_batch = images.unsqueeze(0).to(device)
    actions = model.predict_actions(state_tensor, image_batch)[0].cpu().numpy()
    actions = actions * stats["action_std"] + stats["action_mean"]
    if actions.shape != (model.prediction_horizon, 14) or not np.isfinite(actions).all():
        raise RuntimeError(f"Invalid predicted actions: shape={actions.shape}")
    actions = actions.astype(np.float32)
    actions[:, [6, 13]] = np.clip(actions[:, [6, 13]], 0.0, 1.0)
    return actions


def run_episode(args, model, stats, image_size, episode_index, device):
    seed = args.seed + episode_index
    torch.manual_seed(seed)
    spec = abc_sim.get_task_spec("sim_set_up_chess_pieces_on_the_board")
    env = abc_sim.make_env(
        task=spec.env_task,
        prompt=spec.prompt,
        render_cameras=True,
        camera_backend="mujoco",
        camera_height=168,
        camera_width=224,
        max_episode_steps=args.max_actions,
        terminate_on_success=True,
    )
    writer = None
    video_path = args.output / f"episode_{episode_index:03d}.mp4"
    try:
        obs, reset_info = env.reset(seed=seed, randomize=not args.fixed_scene)
        history = deque([obs, obs], maxlen=2)
        if not args.no_video:
            rgb = top_frame(obs)
            height, width = rgb.shape[:2]
            writer = cv2.VideoWriter(
                str(video_path), cv2.VideoWriter_fourcc(*"mp4v"),
                30.0 / args.video_every, (width, height),
            )
            if not writer.isOpened():
                raise RuntimeError(f"Cannot create video {video_path}")
            writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        actions_done = 0
        chunks_done = 0
        best_reward = float("-inf")
        ever_success = False
        terminated = False
        truncated = False
        reward = 0.0
        info = {}
        trace = {
            "action": [], "reward": [], "targets_placed": [],
            "non_target_disturbed": [], "left_gripper": [],
            "right_gripper": [], "joint_target_change": [],
        }
        previous_action = None
        while actions_done < args.max_actions and not (terminated or truncated):
            chunk = predict_chunk(model, stats, history, image_size, device)
            count = min(args.execute_actions, len(chunk), args.max_actions - actions_done)
            chunks_done += 1
            for action in chunk[:count]:
                obs, reward, terminated, truncated, info = env.step(action)
                history.append(obs)
                actions_done += 1
                best_reward = max(best_reward, float(reward))
                ever_success |= bool(info.get("task_success", False))
                task_eval = info.get("task_eval") or {}
                trace["action"].append(actions_done)
                trace["reward"].append(float(reward))
                trace["targets_placed"].append(task_eval.get("num_targets_placed"))
                trace["non_target_disturbed"].append(task_eval.get("num_non_target_disturbed"))
                trace["left_gripper"].append(float(action[6]))
                trace["right_gripper"].append(float(action[13]))
                trace["joint_target_change"].append(
                    float(np.linalg.norm(np.concatenate([action[:6] - previous_action[:6], action[7:13] - previous_action[7:13]])))
                    if previous_action is not None else 0.0
                )
                previous_action = action.copy()
                if writer is not None and actions_done % args.video_every == 0:
                    writer.write(cv2.cvtColor(top_frame(obs), cv2.COLOR_RGB2BGR))
                if terminated or truncated:
                    break
            print(
                f"episode={episode_index} seed={seed} chunks={chunks_done} "
                f"actions={actions_done}/{args.max_actions} "
                f"reward={float(reward):.4f} success={ever_success}",
                flush=True,
            )
        if writer is not None and actions_done % args.video_every != 0:
            writer.write(cv2.cvtColor(top_frame(obs), cv2.COLOR_RGB2BGR))
        trace_path = args.output / f"episode_{episode_index:03d}_trace.json"
        trace_path.write_text(json.dumps(trace, indent=2) + "\n")
        graph_path = save_episode_plots(trace, args.output, episode_index)
        result = {
            "episode": episode_index,
            "seed": seed,
            "actions": actions_done,
            "chunks": chunks_done,
            "success": bool(ever_success),
            "final_reward": float(reward),
            "best_reward": best_reward if actions_done else None,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "task_eval": json_safe(info.get("task_eval")),
            "video": str(video_path) if writer is not None else None,
            "trace": str(trace_path),
            "reward_graph": graph_path,
        }
        print(json.dumps(result, indent=2), flush=True)
        return result
    finally:
        if writer is not None:
            writer.release()
        env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/chess-full/best.pt"))
    parser.add_argument("--output", type=Path, default=Path("runs/chess-rollout"))
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-actions", type=int, default=100)
    parser.add_argument("--execute-actions", type=int, default=10)
    parser.add_argument("--video-every", type=int, default=5)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--fixed-scene", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    if args.plot_only:
        plot_existing_summary(args.output)
        return
    if args.episodes < 1 or args.max_actions < 1 or args.video_every < 1:
        parser.error("episodes, max-actions and video-every must be positive")
    if not 1 <= args.execute_actions <= 20:
        parser.error("execute-actions must be between 1 and 20")
    if not args.checkpoint.is_file():
        parser.error(f"Checkpoint not found: {args.checkpoint}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, stats, image_size, checkpoint_step = load_policy(args.checkpoint, device)
    if args.execute_actions > model.prediction_horizon:
        parser.error("execute-actions exceeds the model prediction horizon")
    args.output.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}; EMA checkpoint step: {checkpoint_step}", flush=True)
    results = []
    for episode_index in range(args.episodes):
        result = run_episode(args, model, stats, image_size, episode_index, device)
        results.append(result)
        summary = {
            "checkpoint": str(args.checkpoint),
            "checkpoint_step": checkpoint_step,
            "episodes_completed": len(results),
            "success_rate": sum(item["success"] for item in results) / len(results),
            "episodes": results,
        }
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    plot_existing_summary(args.output)
    print(f"Summary: {args.output / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()

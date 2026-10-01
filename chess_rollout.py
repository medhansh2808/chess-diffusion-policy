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
from chess_dp import (
    CAMERA_KEYS,
    ChessDiffusionPolicy,
    preprocess_rgb_image,
)


def image_tensor(
    observation,
    image_size,
):
    """
    Use the exact same RGB preprocessing as training.
    """
    frames = []

    if "images" not in observation:
        raise KeyError(
            "Observation has no 'images' entry"
        )

    for camera in CAMERA_KEYS:
        if camera not in observation["images"]:
            raise KeyError(
                f"Missing {camera} camera; "
                f"available: {list(observation['images'])}"
            )

        chw = np.asarray(
            observation["images"][camera]
        )

        if (
            chw.ndim != 3
            or chw.shape[0] != 3
        ):
            raise ValueError(
                f"Unexpected {camera} camera shape: "
                f"{chw.shape}"
            )

        rgb = np.ascontiguousarray(
            chw.transpose(1, 2, 0)
        )

        frames.append(
            preprocess_rgb_image(
                rgb,
                image_size=image_size,
            )
        )

    return torch.stack(
        frames,
        dim=0,
    )


def top_frame(observation):
    chw = np.asarray(
        observation["images"]["top"]
    )

    if (
        chw.ndim != 3
        or chw.shape[0] != 3
    ):
        raise ValueError(
            f"Unexpected top camera shape: "
            f"{chw.shape}"
        )

    rgb = np.ascontiguousarray(
        chw.transpose(1, 2, 0)
    )

    if rgb.dtype != np.uint8:
        if np.issubdtype(
            rgb.dtype,
            np.floating,
        ):
            max_value = (
                float(rgb.max())
                if rgb.size
                else 0.0
            )

            if max_value <= 1.0:
                rgb = np.clip(
                    rgb * 255.0,
                    0.0,
                    255.0,
                ).astype(
                    np.uint8
                )
            else:
                rgb = np.clip(
                    rgb,
                    0.0,
                    255.0,
                ).astype(
                    np.uint8
                )
        else:
            rgb = np.clip(
                rgb,
                0,
                255,
            ).astype(
                np.uint8
            )

    return rgb


def json_safe(value):
    if isinstance(
        value,
        np.ndarray,
    ):
        return value.tolist()

    if isinstance(
        value,
        np.generic,
    ):
        return value.item()

    if isinstance(
        value,
        dict,
    ):
        return {
            str(k): json_safe(v)
            for k, v in value.items()
        }

    if isinstance(
        value,
        (list, tuple),
    ):
        return [
            json_safe(v)
            for v in value
        ]

    return value


def make_plot(
    x,
    lines,
    title,
    ylabel,
    output,
):
    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(
        figsize=(10, 4)
    )

    plotted = 0

    for label, values in lines.items():
        if not any(
            value is not None
            for value in values
        ):
            continue

        ax.plot(
            x,
            [
                float("nan")
                if value is None
                else value
                for value in values
            ],
            label=label,
        )
        plotted += 1

    ax.set_title(
        title
    )
    ax.set_xlabel(
        "Executed actions"
    )
    ax.set_ylabel(
        ylabel
    )
    ax.grid(
        alpha=0.3
    )

    if plotted > 1:
        ax.legend()

    fig.tight_layout()
    fig.savefig(
        output,
        dpi=160,
    )
    plt.close(
        fig
    )


def save_episode_plots(
    trace,
    output,
    episode_index,
):
    prefix = (
        output
        / f"episode_{episode_index:03d}"
    )

    reward_path = prefix.with_name(
        prefix.name
        + "_reward.png"
    )

    make_plot(
        trace["action"],
        {
            "Reward":
                trace["reward"],
        },
        "Reward during rollout",
        "Reward",
        reward_path,
    )

    make_plot(
        trace["action"],
        {
            "Targets placed":
                trace["targets_placed"],
            "Non-target pieces disturbed":
                trace["non_target_disturbed"],
        },
        "Chessboard progress",
        "Pieces",
        prefix.with_name(
            prefix.name
            + "_board.png"
        ),
    )

    make_plot(
        trace["action"],
        {
            "Left gripper":
                trace["left_gripper"],
            "Right gripper":
                trace["right_gripper"],
        },
        "Executed gripper commands",
        "Command",
        prefix.with_name(
            prefix.name
            + "_grippers.png"
        ),
    )

    make_plot(
        trace["action"],
        {
            "Joint target change":
                trace[
                    "joint_target_change"
                ],
        },
        (
            "Change between consecutive "
            "joint target commands"
        ),
        "L2 norm",
        prefix.with_name(
            prefix.name
            + "_action_changes.png"
        ),
    )

    make_plot(
        trace["action"],
        {
            "Clipped values":
                trace["clipped_values"],
            "Outside training range":
                trace[
                    "outside_training_range"
                ],
        },
        "Action-range diagnostics",
        "Number of values",
        prefix.with_name(
            prefix.name
            + "_action_range.png"
        ),
    )

    return str(
        reward_path
    )


def plot_existing_summary(
    output,
):
    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    path = (
        output
        / "summary.json"
    )

    if not path.is_file():
        raise FileNotFoundError(
            path
        )

    episodes = json.loads(
        path.read_text()
    )["episodes"]

    x = [
        item["episode"]
        for item in episodes
    ]

    metrics = {
        "final_reward": (
            "Final reward",
            "Reward",
        ),
        "targets_placed": (
            "Targets placed",
            "Pieces",
        ),
        "non_target_disturbed": (
            "Non-target pieces disturbed",
            "Pieces",
        ),
    }

    for key, (
        title,
        ylabel,
    ) in metrics.items():
        values = []

        for item in episodes:
            if key == "final_reward":
                value = item.get(
                    "final_reward"
                )
            else:
                task_eval = (
                    item.get(
                        "task_eval"
                    )
                    or {}
                )

                task_key = (
                    "num_targets_placed"
                    if key
                    == "targets_placed"
                    else "num_non_target_disturbed"
                )

                value = task_eval.get(
                    task_key
                )

            values.append(
                value
            )

        if not any(
            value is not None
            for value in values
        ):
            continue

        fig, ax = plt.subplots(
            figsize=(8, 4)
        )

        ax.bar(
            x,
            [
                0
                if value is None
                else value
                for value in values
            ],
        )

        ax.set_title(
            title
        )
        ax.set_xlabel(
            "Episode"
        )
        ax.set_ylabel(
            ylabel
        )
        ax.set_xticks(
            x
        )

        fig.tight_layout()

        target = (
            output
            / f"summary_{key}.png"
        )

        fig.savefig(
            target,
            dpi=160,
        )
        plt.close(
            fig
        )

        print(
            f"Saved: {target}",
            flush=True,
        )

    for item in episodes:
        trace_path = (
            output
            / (
                f"episode_"
                f"{item['episode']:03d}"
                f"_trace.json"
            )
        )

        if trace_path.is_file():
            save_episode_plots(
                json.loads(
                    trace_path.read_text()
                ),
                output,
                item["episode"],
            )

            print(
                "Saved step-by-step plots "
                f"for episode "
                f"{item['episode']}",
                flush=True,
            )


def load_policy(
    checkpoint_path,
    device,
):
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    if "config" not in checkpoint:
        raise KeyError(
            "Checkpoint is missing config"
        )

    if "stats" not in checkpoint:
        raise KeyError(
            "Checkpoint is missing stats"
        )

    config = checkpoint[
        "config"
    ]

    required_config = (
        "image_size",
        "observation_horizon",
        "prediction_horizon",
        "state_dim",
        "action_dim",
        "vision_feature_dim",
        "freeze_backbone",
        "num_diffusion_steps",
        "noise_schedule",
        "diffusion_step_embed_dim",
        "unet_down_dims",
        "unet_kernel_size",
        "unet_n_groups",
    )

    for key in required_config:
        if key not in config:
            raise KeyError(
                f"Checkpoint config is missing {key}"
            )

    saved_camera_keys = tuple(
        config.get(
            "camera_keys",
            (),
        )
    )

    if (
        saved_camera_keys
        and saved_camera_keys
        != CAMERA_KEYS
    ):
        raise ValueError(
            "Checkpoint camera order does not "
            f"match rollout code: "
            f"{saved_camera_keys} vs {CAMERA_KEYS}"
        )

    stats = {}

    for key in (
        "state_mean",
        "state_std",
        "action_mean",
        "action_std",
        "state_min",
        "state_max",
        "action_min",
        "action_max",
    ):
        if key not in checkpoint["stats"]:
            continue

        stats[key] = np.asarray(
            checkpoint["stats"][key],
            dtype=np.float32,
        )

        if stats[key].shape != (14,):
            raise ValueError(
                f"Checkpoint {key} has shape "
                f"{stats[key].shape}, expected (14,)"
            )

        if not np.isfinite(
            stats[key]
        ).all():
            raise ValueError(
                f"Checkpoint {key} contains NaN/Inf"
            )

    for key in (
        "state_mean",
        "state_std",
        "action_mean",
        "action_std",
    ):
        if key not in stats:
            raise KeyError(
                f"Checkpoint is missing {key}"
            )

    if np.any(
        stats["state_std"] <= 0
    ):
        raise ValueError(
            "Invalid state_std in checkpoint"
        )

    if np.any(
        stats["action_std"] <= 0
    ):
        raise ValueError(
            "Invalid action_std in checkpoint"
        )

    model = ChessDiffusionPolicy(
        observation_horizon=int(
            config[
                "observation_horizon"
            ]
        ),
        prediction_horizon=int(
            config[
                "prediction_horizon"
            ]
        ),
        state_dim=int(
            config[
                "state_dim"
            ]
        ),
        action_dim=int(
            config[
                "action_dim"
            ]
        ),
        vision_feature_dim=int(
            config[
                "vision_feature_dim"
            ]
        ),
        pretrained_vision=False,
        freeze_backbone=bool(
            config[
                "freeze_backbone"
            ]
        ),
        num_diffusion_steps=int(
            config[
                "num_diffusion_steps"
            ]
        ),
        noise_schedule=str(
            config[
                "noise_schedule"
            ]
        ),
        diffusion_step_embed_dim=int(
            config[
                "diffusion_step_embed_dim"
            ]
        ),
        unet_down_dims=tuple(
            config[
                "unet_down_dims"
            ]
        ),
        unet_kernel_size=int(
            config[
                "unet_kernel_size"
            ]
        ),
        unet_n_groups=int(
            config[
                "unet_n_groups"
            ]
        ),
    ).to(
        device
    )

    model.load_state_dict(
        checkpoint["ema"],
        strict=True,
    )

    model.eval()

    if model.action_dim != 14:
        raise ValueError(
            "This rollout script expects "
            "14-dimensional ABC actions"
        )

    return (
        model,
        stats,
        int(
            config[
                "image_size"
            ]
        ),
        int(
            checkpoint[
                "step"
            ]
        ),
    )


@torch.inference_mode()
def predict_chunk(
    model,
    stats,
    history,
    image_size,
    device,
):
    if (
        len(history)
        != model.observation_horizon
    ):
        raise ValueError(
            f"History has {len(history)} observations; "
            f"model expects "
            f"{model.observation_horizon}"
        )

    states = np.stack([
        np.asarray(
            obs["state"],
            dtype=np.float32,
        )
        for obs in history
    ])

    if states.shape != (
        model.observation_horizon,
        model.state_dim,
    ):
        raise ValueError(
            f"Unexpected state history shape: "
            f"{states.shape}"
        )

    if not np.isfinite(
        states
    ).all():
        raise RuntimeError(
            "Observation state contains NaN/Inf"
        )

    normalized = (
        states
        - stats[
            "state_mean"
        ]
    ) / stats[
        "state_std"
    ]

    state_tensor = (
        torch.from_numpy(
            np.ascontiguousarray(
                normalized
            )
        )
        .unsqueeze(0)
        .to(
            device
        )
    )

    images = torch.stack(
        [
            image_tensor(
                obs,
                image_size,
            )
            for obs in history
        ],
        dim=0,
    )

    image_batch = (
        images
        .unsqueeze(0)
        .to(
            device
        )
    )

    actions = (
        model.predict_actions(
            state_tensor,
            image_batch,
        )[0]
        .cpu()
        .numpy()
    )

    expected_shape = (
        model.prediction_horizon,
        model.action_dim,
    )

    if actions.shape != expected_shape:
        raise RuntimeError(
            f"Invalid predicted action shape: "
            f"{actions.shape}; "
            f"expected {expected_shape}"
        )

    if not np.isfinite(
        actions
    ).all():
        raise RuntimeError(
            "Normalized predicted actions "
            "contain NaN/Inf"
        )

    actions = (
        actions
        * stats[
            "action_std"
        ]
        + stats[
            "action_mean"
        ]
    )

    if not np.isfinite(
        actions
    ).all():
        raise RuntimeError(
            "Denormalized predicted actions "
            "contain NaN/Inf"
        )

    return actions.astype(
        np.float32
    )


def sanitize_action(
    action,
):
    """
    Final safety boundary before env.step().

    ABC policy actions are bounded to [-1, 1]. Gripper commands in this policy
    use [0, 1]. We do NOT clamp the model output earlier, because we want the
    diagnostics to reveal when the model is predicting bad values.
    """
    action = np.asarray(
        action,
        dtype=np.float32,
    )

    if action.shape != (14,):
        raise ValueError(
            f"Expected action shape (14,), "
            f"got {action.shape}"
        )

    if not np.isfinite(
        action
    ).all():
        raise RuntimeError(
            "Refusing to execute action "
            "containing NaN/Inf"
        )

    sanitized = np.clip(
        action,
        -1.0,
        1.0,
    )

    sanitized[
        [6, 13]
    ] = np.clip(
        sanitized[
            [6, 13]
        ],
        0.0,
        1.0,
    )

    clipped_values = int(
        np.count_nonzero(
            np.abs(
                sanitized
                - action
            )
            > 1e-7
        )
    )

    return (
        sanitized.astype(
            np.float32,
            copy=False,
        ),
        clipped_values,
    )


def count_outside_training_range(
    action,
    stats,
):
    if (
        "action_min"
        not in stats
        or "action_max"
        not in stats
    ):
        return 0

    action = np.asarray(
        action,
        dtype=np.float32,
    )

    return int(
        np.count_nonzero(
            (
                action
                < stats[
                    "action_min"
                ]
            )
            |
            (
                action
                > stats[
                    "action_max"
                ]
            )
        )
    )


def run_episode(
    args,
    model,
    stats,
    image_size,
    episode_index,
    device,
):
    seed = (
        args.seed
        + episode_index
    )

    torch.manual_seed(
        seed
    )
    np.random.seed(
        seed
    )

    spec = abc_sim.get_task_spec(
        "sim_set_up_chess_pieces_on_the_board"
    )

    # Render square cameras at the model's training image size. Combined with
    # preprocess_rgb_image(), this removes the previous 168x224 -> 224x224
    # geometric distortion.
    env = abc_sim.make_env(
        task=spec.env_task,
        prompt=spec.prompt,
        render_cameras=True,
        camera_backend="mujoco",
        camera_height=image_size,
        camera_width=image_size,
        max_episode_steps=args.max_actions,
        terminate_on_success=True,
    )

    writer = None

    video_path = (
        args.output
        / f"episode_{episode_index:03d}.mp4"
    )

    try:
        obs, reset_info = env.reset(
            seed=seed,
            randomize=(
                not args.fixed_scene
            ),
        )

        history = deque(
            [
                obs
                for _ in range(
                    model.observation_horizon
                )
            ],
            maxlen=(
                model.observation_horizon
            ),
        )

        if not args.no_video:
            rgb = top_frame(
                obs
            )

            height, width = (
                rgb.shape[:2]
            )

            writer = cv2.VideoWriter(
                str(
                    video_path
                ),
                cv2.VideoWriter_fourcc(
                    *"mp4v"
                ),
                30.0
                / args.video_every,
                (
                    width,
                    height,
                ),
            )

            if not writer.isOpened():
                raise RuntimeError(
                    f"Cannot create video "
                    f"{video_path}"
                )

            writer.write(
                cv2.cvtColor(
                    rgb,
                    cv2.COLOR_RGB2BGR,
                )
            )

        actions_done = 0
        chunks_done = 0
        best_reward = float(
            "-inf"
        )
        ever_success = False
        terminated = False
        truncated = False
        reward = 0.0
        info = {}

        total_clipped_values = 0
        total_action_values = 0
        total_outside_training_range = 0

        raw_prediction_min = float(
            "inf"
        )
        raw_prediction_max = float(
            "-inf"
        )

        trace = {
            "action": [],
            "reward": [],
            "targets_placed": [],
            "non_target_disturbed": [],
            "left_gripper": [],
            "right_gripper": [],
            "joint_target_change": [],
            "clipped_values": [],
            "outside_training_range": [],
        }

        previous_action = None

        while (
            actions_done
            < args.max_actions
            and not (
                terminated
                or truncated
            )
        ):
            chunk = predict_chunk(
                model,
                stats,
                history,
                image_size,
                device,
            )

            raw_prediction_min = min(
                raw_prediction_min,
                float(
                    chunk.min()
                ),
            )
            raw_prediction_max = max(
                raw_prediction_max,
                float(
                    chunk.max()
                ),
            )

            count = min(
                args.execute_actions,
                len(chunk),
                (
                    args.max_actions
                    - actions_done
                ),
            )

            chunks_done += 1

            for raw_action in chunk[
                :count
            ]:
                outside_training_range = (
                    count_outside_training_range(
                        raw_action,
                        stats,
                    )
                )

                action, clipped_values = (
                    sanitize_action(
                        raw_action
                    )
                )

                total_outside_training_range += (
                    outside_training_range
                )
                total_clipped_values += (
                    clipped_values
                )
                total_action_values += (
                    action.size
                )

                obs, reward, terminated, truncated, info = (
                    env.step(
                        action
                    )
                )

                history.append(
                    obs
                )
                actions_done += 1

                best_reward = max(
                    best_reward,
                    float(
                        reward
                    ),
                )

                ever_success |= bool(
                    info.get(
                        "task_success",
                        False,
                    )
                )

                task_eval = (
                    info.get(
                        "task_eval"
                    )
                    or {}
                )

                trace[
                    "action"
                ].append(
                    actions_done
                )

                trace[
                    "reward"
                ].append(
                    float(
                        reward
                    )
                )

                trace[
                    "targets_placed"
                ].append(
                    task_eval.get(
                        "num_targets_placed"
                    )
                )

                trace[
                    "non_target_disturbed"
                ].append(
                    task_eval.get(
                        "num_non_target_disturbed"
                    )
                )

                trace[
                    "left_gripper"
                ].append(
                    float(
                        action[6]
                    )
                )

                trace[
                    "right_gripper"
                ].append(
                    float(
                        action[13]
                    )
                )

                if previous_action is None:
                    action_change = 0.0
                else:
                    action_change = float(
                        np.linalg.norm(
                            np.concatenate([
                                (
                                    action[:6]
                                    - previous_action[:6]
                                ),
                                (
                                    action[7:13]
                                    - previous_action[7:13]
                                ),
                            ])
                        )
                    )

                trace[
                    "joint_target_change"
                ].append(
                    action_change
                )

                trace[
                    "clipped_values"
                ].append(
                    clipped_values
                )

                trace[
                    "outside_training_range"
                ].append(
                    outside_training_range
                )

                previous_action = (
                    action.copy()
                )

                if (
                    writer is not None
                    and actions_done
                    % args.video_every
                    == 0
                ):
                    writer.write(
                        cv2.cvtColor(
                            top_frame(
                                obs
                            ),
                            cv2.COLOR_RGB2BGR,
                        )
                    )

                if (
                    terminated
                    or truncated
                ):
                    break

            clipped_fraction = (
                total_clipped_values
                / max(
                    total_action_values,
                    1,
                )
            )

            outside_fraction = (
                total_outside_training_range
                / max(
                    total_action_values,
                    1,
                )
            )

            print(
                f"episode={episode_index} "
                f"seed={seed} "
                f"chunks={chunks_done} "
                f"actions="
                f"{actions_done}/{args.max_actions} "
                f"reward={float(reward):.4f} "
                f"success={ever_success} "
                f"clip={100.0 * clipped_fraction:.2f}% "
                f"outside_train="
                f"{100.0 * outside_fraction:.2f}%",
                flush=True,
            )

        if (
            writer is not None
            and actions_done
            % args.video_every
            != 0
        ):
            writer.write(
                cv2.cvtColor(
                    top_frame(
                        obs
                    ),
                    cv2.COLOR_RGB2BGR,
                )
            )

        trace_path = (
            args.output
            / (
                f"episode_"
                f"{episode_index:03d}"
                f"_trace.json"
            )
        )

        trace_path.write_text(
            json.dumps(
                trace,
                indent=2,
            )
            + "\n"
        )

        graph_path = (
            save_episode_plots(
                trace,
                args.output,
                episode_index,
            )
        )

        clipped_fraction = (
            total_clipped_values
            / max(
                total_action_values,
                1,
            )
        )

        outside_fraction = (
            total_outside_training_range
            / max(
                total_action_values,
                1,
            )
        )

        result = {
            "episode":
                episode_index,
            "seed":
                seed,
            "actions":
                actions_done,
            "chunks":
                chunks_done,
            "success":
                bool(
                    ever_success
                ),
            "final_reward":
                float(
                    reward
                ),
            "best_reward": (
                best_reward
                if actions_done
                else None
            ),
            "terminated":
                bool(
                    terminated
                ),
            "truncated":
                bool(
                    truncated
                ),
            "task_eval":
                json_safe(
                    info.get(
                        "task_eval"
                    )
                ),
            "raw_prediction_min": (
                raw_prediction_min
                if actions_done
                else None
            ),
            "raw_prediction_max": (
                raw_prediction_max
                if actions_done
                else None
            ),
            "clipped_values":
                total_clipped_values,
            "executed_action_values":
                total_action_values,
            "clipped_fraction":
                clipped_fraction,
            "outside_training_range_values":
                total_outside_training_range,
            "outside_training_range_fraction":
                outside_fraction,
            "video": (
                str(
                    video_path
                )
                if writer is not None
                else None
            ),
            "trace":
                str(
                    trace_path
                ),
            "reward_graph":
                graph_path,
            "reset_info":
                json_safe(
                    reset_info
                ),
        }

        print(
            json.dumps(
                result,
                indent=2,
            ),
            flush=True,
        )

        return result

    finally:
        if writer is not None:
            writer.release()

        env.close()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "runs/chess-full/best.pt"
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "runs/chess-rollout"
        ),
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--max-actions",
        type=int,
        default=3540,
        help=(
            "Maximum control actions per episode. "
            "100 actions was only about a few seconds "
            "and was not a meaningful chess evaluation."
        ),
    )

    parser.add_argument(
        "--execute-actions",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--video-every",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--no-video",
        action="store_true",
    )

    parser.add_argument(
        "--fixed-scene",
        action="store_true",
    )

    parser.add_argument(
        "--plot-only",
        action="store_true",
    )

    args = parser.parse_args()

    if args.plot_only:
        plot_existing_summary(
            args.output
        )
        return

    if (
        args.episodes < 1
        or args.max_actions < 1
        or args.video_every < 1
    ):
        parser.error(
            "episodes, max-actions and "
            "video-every must be positive"
        )

    if args.execute_actions < 1:
        parser.error(
            "execute-actions must be positive"
        )

    if not args.checkpoint.is_file():
        parser.error(
            f"Checkpoint not found: "
            f"{args.checkpoint}"
        )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    (
        model,
        stats,
        image_size,
        checkpoint_step,
    ) = load_policy(
        args.checkpoint,
        device,
    )

    if (
        args.execute_actions
        > model.prediction_horizon
    ):
        parser.error(
            "execute-actions exceeds "
            "the model prediction horizon"
        )

    args.output.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        f"Device: {device}; "
        f"EMA checkpoint step: "
        f"{checkpoint_step}",
        flush=True,
    )

    print(
        f"Observation horizon: "
        f"{model.observation_horizon}; "
        f"prediction horizon: "
        f"{model.prediction_horizon}; "
        f"execute actions: "
        f"{args.execute_actions}",
        flush=True,
    )

    results = []

    for episode_index in range(
        args.episodes
    ):
        result = run_episode(
            args,
            model,
            stats,
            image_size,
            episode_index,
            device,
        )

        results.append(
            result
        )

        summary = {
            "checkpoint":
                str(
                    args.checkpoint
                ),
            "checkpoint_step":
                checkpoint_step,
            "episodes_completed":
                len(
                    results
                ),
            "success_rate": (
                sum(
                    item["success"]
                    for item in results
                )
                / len(
                    results
                )
            ),
            "episodes":
                results,
        }

        (
            args.output
            / "summary.json"
        ).write_text(
            json.dumps(
                summary,
                indent=2,
            )
            + "\n"
        )

    plot_existing_summary(
        args.output
    )

    print(
        f"Summary: "
        f"{args.output / 'summary.json'}",
        flush=True,
    )


if __name__ == "__main__":
    main()

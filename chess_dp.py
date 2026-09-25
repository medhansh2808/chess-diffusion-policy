import argparse
import copy
import json
import math
import sqlite3
import time
import zlib
from pathlib import Path


import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.models import resnet18, ResNet18_Weights


from abc_minimal.episode_io import (
    load_episode,
    discover_episodes,
)
from diffusion_policy.model.diffusion.conditional_unet1d import (
    ConditionalUnet1D,
)




TASK_NAME = "sim_set_up_chess_pieces_on_the_board"
CAMERA_KEYS = ("top", "left", "right")




def prepare_frame_cache(episode_dir, expected_frames):
    video_path = (
        Path(episode_dir)
        / "combined_camera-images-rgb.mp4"
    )

    if not video_path.is_file():
        raise FileNotFoundError(video_path)

    cache_path = video_path.with_suffix(".frames.sqlite3")
    video_stat = video_path.stat()
    source_info = (
        video_stat.st_size,
        video_stat.st_mtime_ns,
        int(expected_frames),
    )

    if cache_path.is_file():
        try:
            connection = sqlite3.connect(cache_path)
            try:
                cached_info = connection.execute(
                    "SELECT source_size, source_mtime_ns, frame_count "
                    "FROM metadata"
                ).fetchone()
            finally:
                connection.close()

            if cached_info == source_info:
                return cache_path
        except sqlite3.Error:
            pass

    print(f"Building frame cache: {video_path}", flush=True)
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    temporary_path = cache_path.with_suffix(".tmp")
    temporary_path.unlink(missing_ok=True)
    connection = None

    try:
        connection = sqlite3.connect(temporary_path)
        connection.execute(
            "CREATE TABLE frames ("
            "frame_idx INTEGER PRIMARY KEY, "
            "height INTEGER NOT NULL, "
            "width INTEGER NOT NULL, "
            "data BLOB NOT NULL)"
        )
        connection.execute(
            "CREATE TABLE metadata ("
            "source_size INTEGER, "
            "source_mtime_ns INTEGER, "
            "frame_count INTEGER)"
        )

        frame_count = 0

        while True:
            ok, frame = cap.read()

            if not ok:
                break

            height, width, channels = frame.shape

            if channels != 3:
                raise ValueError(
                    f"Unexpected frame shape: {frame.shape}"
                )

            connection.execute(
                "INSERT INTO frames VALUES (?, ?, ?, ?)",
                (
                    frame_count,
                    height,
                    width,
                    sqlite3.Binary(
                        zlib.compress(frame.tobytes(), level=1)
                    ),
                ),
            )
            frame_count += 1

        if frame_count != expected_frames:
            raise RuntimeError(
                f"Video/state length mismatch in {episode_dir}: "
                f"{frame_count} video frames, "
                f"{expected_frames} state frames"
            )

        connection.execute(
            "INSERT INTO metadata VALUES (?, ?, ?)",
            source_info,
        )
        connection.commit()
        connection.close()
        connection = None
        cap.release()
        temporary_path.replace(cache_path)
        return cache_path

    finally:
        if connection is not None:
            connection.close()
        cap.release()
        temporary_path.unlink(missing_ok=True)


def load_camera_frame(
    episode_dir,
    frame_idx,
    source_cameras,
    image_size=224,
):
    """
    Load one timestep from ABC's sequentially decoded frame cache.


    Returns:
        (3, 3, image_size, image_size)


    Camera order:
        top, left, right
    """


    video_path = (
        Path(episode_dir)
        / "combined_camera-images-rgb.mp4"
    )


    source_cameras = tuple(source_cameras)


    if not set(CAMERA_KEYS).issubset(source_cameras):
        raise ValueError(
            f"Missing cameras: {source_cameras}"
        )


    cache_path = video_path.with_suffix(".frames.sqlite3")

    if not cache_path.is_file():
        raise RuntimeError(
            f"Missing frame cache: {cache_path}. "
            "Initialize ChessDataset before reading images."
        )

    connection = sqlite3.connect(cache_path)
    try:
        row = connection.execute(
            "SELECT height, width, data FROM frames "
            "WHERE frame_idx = ?",
            (int(frame_idx),),
        ).fetchone()
    finally:
        connection.close()

    if row is None:
        raise IndexError(
            f"Frame {frame_idx} not found in {cache_path}"
        )

    height, width, data = row
    frame = np.frombuffer(
        zlib.decompress(data),
        dtype=np.uint8,
    ).reshape(height, width, 3)


    # OpenCV uses BGR. ResNet expects RGB.
    frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB,
    )


    num_cameras = len(source_cameras)


    if frame.shape[0] % num_cameras != 0:
        raise ValueError(
            f"Unexpected video shape: {frame.shape}"
        )


    camera_height = (
        frame.shape[0] // num_cameras
    )


    cameras = {}


    for i, name in enumerate(source_cameras):
        image = frame[
            i * camera_height:
            (i + 1) * camera_height
        ]


        image = cv2.resize(
            image,
            (image_size, image_size),
            interpolation=cv2.INTER_LINEAR,
        )


        cameras[name] = (
            torch.from_numpy(image.copy())
            .permute(2, 0, 1)
            .float()
            / 255.0
        )


    return torch.stack(
        [cameras[name] for name in CAMERA_KEYS],
        dim=0,
    )


class ChessDataset(Dataset):
    def __init__(
        self,
        episode_dirs,
        observation_horizon=2,
        prediction_horizon=20,
        stats=None,
        load_images=True,
        image_size=224,
    ):
        super().__init__()


        if observation_horizon < 1:
            raise ValueError(
                "Observation horizon must be positive"
            )


        if prediction_horizon < 1:
            raise ValueError(
                "Prediction horizon must be positive"
            )


        self.observation_horizon = (
            observation_horizon
        )
        self.prediction_horizon = (
            prediction_horizon
        )
        self.load_images = load_images
        self.image_size = image_size


        self.episodes = []


        for episode_dir in episode_dirs:
            episode_dir = Path(episode_dir)


            metadata, states, actions = (
                load_episode(episode_dir)
            )


            if len(states) != len(actions):
                raise ValueError(
                    f"State/action length mismatch: "
                    f"{episode_dir}"
                )


            if (
                states.ndim != 2
                or states.shape[1] != 14
                or actions.ndim != 2
                or actions.shape[1] != 14
            ):
                raise ValueError(
                    f"Expected 14D states and actions: "
                    f"{episode_dir}"
                )


            if len(states) == 0:
                continue


            cameras = tuple(
                metadata.get("cameras", ())
            )


            if (
                load_images
                and not set(CAMERA_KEYS).issubset(
                    cameras
                )
            ):
                raise ValueError(
                    f"Missing camera metadata: "
                    f"{episode_dir}"
                )


            if load_images:
                prepare_frame_cache(
                    episode_dir=episode_dir,
                    expected_frames=len(states),
                )

            self.episodes.append({
                "path": episode_dir,
                "cameras": cameras,
                "states": states.astype(
                    np.float32
                ),
                "actions": actions.astype(
                    np.float32
                ),
            })


        if not self.episodes:
            raise ValueError(
                "No usable episodes provided"
            )


        # Compute statistics on training data only.
        # Validation reuses the same statistics.
        if stats is None:
            all_states = np.concatenate(
                [
                    ep["states"]
                    for ep in self.episodes
                ],
                axis=0,
            )


            all_actions = np.concatenate(
                [
                    ep["actions"]
                    for ep in self.episodes
                ],
                axis=0,
            )


            stats = {
                "state_mean":
                    all_states.mean(axis=0),


                "state_std":
                    np.maximum(
                        all_states.std(axis=0),
                        1e-6,
                    ),


                "action_mean":
                    all_actions.mean(axis=0),


                "action_std":
                    np.maximum(
                        all_actions.std(axis=0),
                        1e-6,
                    ),
            }


        required_keys = (
            "state_mean",
            "state_std",
            "action_mean",
            "action_std",
        )


        self.stats = {
            key: np.asarray(
                stats[key],
                dtype=np.float32,
            ).copy()
            for key in required_keys
        }


        for key, value in self.stats.items():
            if value.shape != (14,):
                raise ValueError(
                    f"{key} must have shape (14,)"
                )


        self.state_mean = (
            self.stats["state_mean"]
        )
        self.state_std = (
            self.stats["state_std"]
        )
        self.action_mean = (
            self.stats["action_mean"]
        )
        self.action_std = (
            self.stats["action_std"]
        )


        # Every episode timestep can start a sample.
        self.indices = [
            (episode_idx, t)
            for episode_idx, ep in enumerate(
                self.episodes
            )
            for t in range(
                len(ep["states"])
            )
        ]


    def __len__(self):
        return len(self.indices)


    def __getitem__(self, index):
        episode_idx, t = self.indices[index]


        episode = self.episodes[episode_idx]


        states = episode["states"]
        actions = episode["actions"]


        episode_length = len(states)


        # Observation history.
        obs_indices = np.arange(
            t - self.observation_horizon + 1,
            t + 1,
        )


        obs_indices = np.clip(
            obs_indices,
            0,
            episode_length - 1,
        )


        # Future action chunk.
        action_indices = np.arange(
            t,
            t + self.prediction_horizon,
        )


        is_pad = (
            action_indices >= episode_length
        )


        action_indices = np.clip(
            action_indices,
            0,
            episode_length - 1,
        )


        observation_states = (
            states[obs_indices]
        )


        action_sequence = (
            actions[action_indices]
        )


        # Normalize with training statistics.
        observation_states = (
            observation_states
            - self.state_mean
        ) / self.state_std


        action_sequence = (
            action_sequence
            - self.action_mean
        ) / self.action_std


        sample = {
            "states": torch.from_numpy(
                observation_states.copy()
            ),
            "actions": torch.from_numpy(
                action_sequence.copy()
            ),
            "is_pad": torch.from_numpy(
                is_pad.copy()
            ),
        }


        if self.load_images:
            frame_cache = {}


            for frame_idx in obs_indices:
                frame_idx = int(frame_idx)


                if frame_idx not in frame_cache:
                    frame_cache[frame_idx] = (
                        load_camera_frame(
                            episode_dir=episode["path"],
                            frame_idx=frame_idx,
                            source_cameras=episode["cameras"],
                            image_size=self.image_size,
                        )
                    )


            sample["images"] = torch.stack(
                [
                    frame_cache[int(idx)]
                    for idx in obs_indices
                ],
                dim=0,
            )


        return sample


    def denormalize_actions(self, actions):
        """Restore actions to the original robot units."""


        if isinstance(actions, torch.Tensor):
            mean = torch.as_tensor(
                self.action_mean,
                device=actions.device,
                dtype=actions.dtype,
            )


            std = torch.as_tensor(
                self.action_std,
                device=actions.device,
                dtype=actions.dtype,
            )


            return actions * std + mean


        return (
            actions * self.action_std
            + self.action_mean
        )




class DDPMScheduler:
    def __init__(
        self,
        num_train_steps=100,
        beta_start=1e-4,
        beta_end=0.02,
        schedule="cosine",
    ):
        if num_train_steps < 2:
            raise ValueError(
                "At least two diffusion steps required"
            )


        self.num_train_steps = (
            num_train_steps
        )


        if schedule == "linear":
            self.betas = torch.linspace(
                beta_start,
                beta_end,
                num_train_steps,
                dtype=torch.float32,
            )


        elif schedule == "cosine":
            steps = torch.arange(
                num_train_steps + 1,
                dtype=torch.float64,
            )


            s = 0.008


            alpha_bar = torch.cos(
                (
                    (steps / num_train_steps + s)
                    / (1 + s)
                ) * math.pi / 2
            ) ** 2


            alpha_bar = (
                alpha_bar / alpha_bar[0]
            )


            self.betas = (
                1
                - alpha_bar[1:]
                / alpha_bar[:-1]
            ).clamp(
                1e-4,
                0.999,
            ).float()


        else:
            raise ValueError(
                f"Unknown schedule: {schedule}"
            )


        self.alphas = (
            1.0 - self.betas
        )


        self.alpha_bars = torch.cumprod(
            self.alphas,
            dim=0,
        )


        self.alpha_bars_prev = torch.cat([
            torch.ones(1),
            self.alpha_bars[:-1],
        ])


    def add_noise(
        self,
        clean_actions,
        noise,
        timesteps,
    ):
        alpha_bar = self.alpha_bars.to(
            device=clean_actions.device,
            dtype=clean_actions.dtype,
        )[timesteps.long()]


        alpha_bar = (
            alpha_bar[:, None, None]
        )


        return (
            alpha_bar.sqrt()
            * clean_actions
            + (1 - alpha_bar).sqrt()
            * noise
        )


    def step(
        self,
        predicted_noise,
        timestep,
        sample,
    ):
        t = int(timestep)


        device = sample.device
        dtype = sample.dtype


        beta_t = self.betas[t].to(
            device=device,
            dtype=dtype,
        )


        alpha_t = self.alphas[t].to(
            device=device,
            dtype=dtype,
        )


        alpha_bar_t = self.alpha_bars[t].to(
            device=device,
            dtype=dtype,
        )


        alpha_bar_prev = (
            self.alpha_bars_prev[t].to(
                device=device,
                dtype=dtype,
            )
        )


        predicted_x0 = (
            sample
            - (1 - alpha_bar_t).sqrt()
            * predicted_noise
        ) / alpha_bar_t.sqrt()


        coefficient_x0 = (
            alpha_bar_prev.sqrt()
            * beta_t
            / (1 - alpha_bar_t)
        )


        coefficient_xt = (
            alpha_t.sqrt()
            * (1 - alpha_bar_prev)
            / (1 - alpha_bar_t)
        )


        mean = (
            coefficient_x0 * predicted_x0
            + coefficient_xt * sample
        )


        if t == 0:
            return mean


        variance = (
            beta_t
            * (1 - alpha_bar_prev)
            / (1 - alpha_bar_t)
        )


        return (
            mean
            + variance.sqrt()
            * torch.randn_like(sample)
        )


    @torch.no_grad()
    def sample(
        self,
        model,
        shape,
        condition,
        device,
    ):
        actions = torch.randn(
            shape,
            device=device,
        )


        for t in reversed(
            range(self.num_train_steps)
        ):
            timesteps = torch.full(
                (shape[0],),
                t,
                device=device,
                dtype=torch.long,
            )


            predicted_noise = model(
                actions,
                timesteps,
                condition,
            )


            actions = self.step(
                predicted_noise,
                t,
                actions,
            )


        return actions




class VisionEncoder(nn.Module):
    def __init__(
        self,
        feature_dim=128,
        pretrained=True,
        freeze_backbone=True,
    ):
        super().__init__()


        weights = (
            ResNet18_Weights.DEFAULT
            if pretrained
            else None
        )


        backbone = resnet18(
            weights=weights
        )


        # Replace ImageNet classification head.
        backbone.fc = nn.Identity()


        self.backbone = backbone
        self.freeze_backbone = (
            freeze_backbone
        )


        self.projection = nn.Sequential(
            nn.Linear(512, feature_dim),
            nn.ReLU(),
        )


        # ImageNet normalization.
        self.register_buffer(
            "image_mean",
            torch.tensor([
                0.485,
                0.456,
                0.406,
            ]).view(1, 3, 1, 1),
        )


        self.register_buffer(
            "image_std",
            torch.tensor([
                0.229,
                0.224,
                0.225,
            ]).view(1, 3, 1, 1),
        )


        if freeze_backbone:
            self.backbone.requires_grad_(
                False
            )
            self.backbone.eval()


    def train(self, mode=True):
        super().train(mode)


        # Frozen BatchNorm statistics stay fixed.
        if self.freeze_backbone:
            self.backbone.eval()


        return self


    def forward(self, images):
        """
        images: (B, O, 3, 3, H, W)


        O = observation horizon
        3 cameras
        3 RGB channels
        """


        B, O, C, RGB, H, W = (
            images.shape
        )


        images = images.reshape(
            B * O * C,
            RGB,
            H,
            W,
        )


        images = (
            images - self.image_mean
        ) / self.image_std


        if self.freeze_backbone:
            with torch.no_grad():
                features = (
                    self.backbone(images)
                )
        else:
            features = (
                self.backbone(images)
            )


        features = (
            self.projection(features)
        )


        return features.reshape(
            B,
            O,
            C,
            -1,
        )






class ChessDiffusionPolicy(nn.Module):
    def __init__(
        self,
        observation_horizon=2,
        prediction_horizon=20,
        state_dim=14,
        action_dim=14,
        vision_feature_dim=128,
        pretrained_vision=True,
        freeze_backbone=True,
    ):
        super().__init__()


        self.observation_horizon = (
            observation_horizon
        )
        self.prediction_horizon = (
            prediction_horizon
        )
        self.state_dim = state_dim
        self.action_dim = action_dim


        self.vision_encoder = VisionEncoder(
            feature_dim=vision_feature_dim,
            pretrained=pretrained_vision,
            freeze_backbone=freeze_backbone,
        )


        condition_dim = (
            observation_horizon
            * (
                3 * vision_feature_dim
                + state_dim
            )
        )


        self.unet = ConditionalUnet1D(
            input_dim=action_dim,
            global_cond_dim=condition_dim,
            diffusion_step_embed_dim=128,
            down_dims=[128, 256, 512],
            kernel_size=5,
            n_groups=8,
        )


        self.scheduler = DDPMScheduler(
            num_train_steps=100,
            schedule="cosine",
        )


    def encode_observations(
        self,
        states,
        images,
    ):
        visual_features = (
            self.vision_encoder(images)
        )


        B, O = states.shape[:2]


        visual_features = (
            visual_features.reshape(
                B,
                O,
                -1,
            )
        )


        condition = torch.cat(
            [
                visual_features,
                states,
            ],
            dim=-1,
        )


        return condition.flatten(
            start_dim=1
        )


    def forward(
        self,
        states,
        images,
        actions,
        is_pad=None,
    ):
        condition = (
            self.encode_observations(
                states,
                images,
            )
        )


        batch_size = actions.shape[0]


        timesteps = torch.randint(
            0,
            self.scheduler.num_train_steps,
            (batch_size,),
            device=actions.device,
            dtype=torch.long,
        )


        noise = (
            torch.randn_like(actions)
        )


        noisy_actions = (
            self.scheduler.add_noise(
                actions,
                noise,
                timesteps,
            )
        )


        predicted_noise = self.unet(
            noisy_actions,
            timesteps,
            global_cond=condition,
        )


        error = (
            predicted_noise - noise
        ).square()


        if is_pad is not None:
            mask = (
                ~is_pad.bool()
            ).unsqueeze(-1)


            valid_values = (
                mask.sum().clamp(min=1)
                * actions.shape[-1]
            )


            return (
                (error * mask).sum()
                / valid_values
            )


        return error.mean()


    @torch.no_grad()
    def predict_actions(
        self,
        states,
        images,
    ):
        condition = (
            self.encode_observations(
                states,
                images,
            )
        )


        shape = (
            states.shape[0],
            self.prediction_horizon,
            self.action_dim,
        )


        def predict_noise(
            noisy,
            timesteps,
            cond,
        ):
            return self.unet(
                noisy,
                timesteps,
                global_cond=cond,
            )


        return self.scheduler.sample(
            model=predict_noise,
            shape=shape,
            condition=condition,
            device=states.device,
        )


def find_chess_episodes(directory):
    episodes = []


    for path in discover_episodes(
        Path(directory)
    ):
        metadata = json.loads(
            (
                path / "episode_metadata.json"
            ).read_text()
        )


        if (
            metadata.get("task_name")
            == TASK_NAME
        ):
            episodes.append(path)


    return sorted(episodes)






@torch.no_grad()
def update_ema(
    ema_model,
    model,
    decay=0.995,
):
    for ema_param, param in zip(
        ema_model.parameters(),
        model.parameters(),
    ):
        ema_param.lerp_(
            param,
            1.0 - decay,
        )


    for ema_buffer, buffer in zip(
        ema_model.buffers(),
        model.buffers(),
    ):
        ema_buffer.copy_(buffer)




@torch.no_grad()
def validate(
    model,
    val_loader,
    device,
    max_batches=20,
    seed=12345,
):
    """
    Validate using a fixed subset and fixed random seed.


    This makes noise-prediction losses more comparable
    across checkpoints.
    """


    model.eval()


    total_loss = 0.0
    total_samples = 0


    cuda_devices = (
        [torch.cuda.current_device()]
        if device.type == "cuda"
        else []
    )


    # Preserve training RNG state.
    with torch.random.fork_rng(
        devices=cuda_devices
    ):
        torch.manual_seed(seed)


        for batch_idx, batch in enumerate(
            val_loader
        ):
            if batch_idx >= max_batches:
                break


            states = batch["states"].to(
                device,
                non_blocking=True,
            )


            images = batch["images"].to(
                device,
                non_blocking=True,
            )


            actions = batch["actions"].to(
                device,
                non_blocking=True,
            )


            is_pad = batch["is_pad"].to(
                device,
                non_blocking=True,
            )


            loss = model(
                states,
                images,
                actions,
                is_pad,
            )


            if not torch.isfinite(loss):
                raise RuntimeError(
                    "Non-finite validation loss"
                )


            batch_size = states.shape[0]


            total_loss += (
                loss.item() * batch_size
            )


            total_samples += batch_size


    if total_samples == 0:
        raise RuntimeError(
            "Validation loader is empty"
        )


    return (
        total_loss / total_samples
    )




def save_training_checkpoint(
    path,
    step,
    model,
    ema_model,
    optimizer,
    lr_scheduler,
    stats,
    config,
    best_val,
):
    path = Path(path)


    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    checkpoint = {
        "step": step,


        "model": model.state_dict(),


        "ema": ema_model.state_dict(),


        "optimizer": optimizer.state_dict(),


        "lr_scheduler":
            lr_scheduler.state_dict(),


        "stats": {
            key: value.copy()
            for key, value in stats.items()
        },


        "config": config,


        "best_val": best_val,


        "torch_rng_state":
            torch.get_rng_state(),
    }


    if torch.cuda.is_available():
        checkpoint["cuda_rng_state"] = (
            torch.cuda.get_rng_state_all()
        )


    temporary_path = (
        path.with_suffix(".tmp")
    )


    torch.save(
        checkpoint,
        temporary_path,
    )


    # Atomic replacement on the same filesystem.
    temporary_path.replace(path)




def train(args):
    if args.steps < 1:
        raise ValueError(
            "--steps must be positive"
        )


    if args.batch_size < 1:
        raise ValueError(
            "--batch-size must be positive"
        )


    if args.val_samples < 1:
        raise ValueError(
            "--val-samples must be positive"
        )


    torch.manual_seed(args.seed)
    np.random.seed(args.seed)


    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


    print(
        f"Device: {device}",
        flush=True,
    )


    args.output.mkdir(
        parents=True,
        exist_ok=True,
    )


    started = time.monotonic()


    metrics_path = (
        args.output / "metrics.jsonl"
    )






    train_paths = find_chess_episodes(
        args.abc_dir / "cache/train_sim"
    )


    val_paths = find_chess_episodes(
        args.abc_dir / "cache/val_sim"
    )


    print(
        f"Training episodes: {len(train_paths)}",
        flush=True,
    )


    print(
        f"Validation episodes: {len(val_paths)}",
        flush=True,
    )


    if not train_paths or not val_paths:
        raise RuntimeError(
            "Missing chess episodes"
        )


    # Ensure no episode appears in both splits.
    train_names = {
        path.name for path in train_paths
    }


    val_names = {
        path.name for path in val_paths
    }


    if train_names & val_names:
        raise RuntimeError(
            "Training and validation episodes overlap"
        )


    train_dataset = ChessDataset(
        episode_dirs=train_paths,
        observation_horizon=2,
        prediction_horizon=20,
        load_images=True,
        image_size=args.image_size,
    )


    val_dataset = ChessDataset(
        episode_dirs=val_paths,
        observation_horizon=2,
        prediction_horizon=20,
        stats=train_dataset.stats,
        load_images=True,
        image_size=args.image_size,
    )


    # Fixed, evenly spaced validation examples.
    val_indices = np.linspace(
        0,
        len(val_dataset) - 1,
        min(
            args.val_samples,
            len(val_dataset),
        ),
        dtype=int,
    ).tolist()


    val_subset = Subset(
        val_dataset,
        val_indices,
    )


    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(
            device.type == "cuda"
        ),
    )


    val_loader = DataLoader(
        val_subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(
            device.type == "cuda"
        ),
    )


    print(
        f"Training samples: {len(train_dataset)}",
        flush=True,
    )


    print(
        f"Fixed validation samples: {len(val_subset)}",
        flush=True,
    )


    # Checkpoint weights supersede ImageNet initialization.
    use_pretrained = (
        not args.no_pretrained
        and args.resume is None
    )


    model = ChessDiffusionPolicy(
        pretrained_vision=use_pretrained,
        freeze_backbone=True,
    ).to(device)


    ema_model = copy.deepcopy(
        model
    )


    ema_model.requires_grad_(
        False
    )


    ema_model.eval()


    optimizer = torch.optim.AdamW(
        (
            p
            for p in model.parameters()
            if p.requires_grad
        ),
        lr=args.lr,
        weight_decay=1e-6,
    )


    lr_scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.steps,
        )
    )


    config = {
        "task": TASK_NAME,
        "observation_horizon": 2,
        "prediction_horizon": 20,
        "state_dim": 14,
        "action_dim": 14,
        "camera_keys": CAMERA_KEYS,
        "image_size": args.image_size,
        "vision_feature_dim": 128,
        "freeze_backbone": True,
        "pretrained_vision": (
            not args.no_pretrained
        ),
        "num_diffusion_steps": 100,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "total_steps": args.steps,
        "ema_decay": args.ema_decay,
        "seed": args.seed,
        "val_samples": args.val_samples,
        "val_seed": args.val_seed,
    }


    start_step = 0
    best_val = float("inf")


    if args.resume is not None:
        checkpoint = torch.load(
            args.resume,
            map_location=device,
            weights_only=False,
        )


        saved_config = (
            checkpoint["config"]
        )


        for key in (
            "task",
            "image_size",
            "observation_horizon",
            "prediction_horizon",
            "vision_feature_dim",
            "total_steps",
            "val_samples",
            "val_seed",
        ):
            if (
                key in saved_config
                and saved_config[key] != config[key]
            ):
                raise ValueError(
                    f"Resume configuration mismatch: "
                    f"{key}"
                )


        # Verify training normalization has not changed.
        for key, original in (
            checkpoint["stats"].items()
        ):
            current = (
                train_dataset.stats[key]
            )


            if not np.allclose(
                original,
                current,
                rtol=1e-5,
                atol=1e-6,
            ):
                raise ValueError(
                    f"Normalization mismatch: {key}"
                )


        model.load_state_dict(
            checkpoint["model"]
        )


        ema_model.load_state_dict(
            checkpoint["ema"]
        )


        optimizer.load_state_dict(
            checkpoint["optimizer"]
        )


        lr_scheduler.load_state_dict(
            checkpoint["lr_scheduler"]
        )


        start_step = int(
            checkpoint["step"]
        )


        best_val = checkpoint.get(
            "best_val",
            float("inf"),
        )


        if "torch_rng_state" in checkpoint:
            torch.set_rng_state(
                checkpoint[
                    "torch_rng_state"
                ].cpu()
            )


        if (
            device.type == "cuda"
            and "cuda_rng_state" in checkpoint
        ):
            torch.cuda.set_rng_state_all(
                checkpoint[
                    "cuda_rng_state"
                ]
            )


        print(
            f"Resumed from step {start_step}",
            flush=True,
        )




    model.train()


    train_iterator = iter(
        train_loader
    )


    for step in range(
        start_step,
        args.steps,
    ):
        try:
            batch = next(
                train_iterator
            )


        except StopIteration:
            train_iterator = iter(
                train_loader
            )


            batch = next(
                train_iterator
            )


        states = batch["states"].to(
            device,
            non_blocking=True,
        )


        images = batch["images"].to(
            device,
            non_blocking=True,
        )


        actions = batch["actions"].to(
            device,
            non_blocking=True,
        )


        is_pad = batch["is_pad"].to(
            device,
            non_blocking=True,
        )


        optimizer.zero_grad(
            set_to_none=True
        )


        loss = model(
            states,
            images,
            actions,
            is_pad,
        )


        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Non-finite loss at step {step + 1}"
            )


        loss.backward()


        grad_norm = (
            torch.nn.utils.clip_grad_norm_(
                (
                    p
                    for p in model.parameters()
                    if p.requires_grad
                ),
                max_norm=1.0,
                error_if_nonfinite=True,
            )
        )


        optimizer.step()


        lr_scheduler.step()


        update_ema(
            ema_model,
            model,
            decay=args.ema_decay,
        )


        completed_step = (
            step + 1
        )




        if (
            completed_step == 1
            or completed_step % args.log_every == 0
        ):
            print(
                f"Step {completed_step}/{args.steps} | "
                f"Loss {loss.item():.5f} | "
                f"Grad {grad_norm.item():.3f} | "
                f"LR {optimizer.param_groups[0]['lr']:.2e}",
                flush=True,
            )




        should_validate = (
            completed_step % args.val_every == 0
            or completed_step == args.steps
        )


        if should_validate:
            val_loss = validate(
                model=ema_model,
                val_loader=val_loader,
                device=device,
                max_batches=args.val_batches,
                seed=args.val_seed,
            )


            improved = (
                val_loss < best_val
            )


            if improved:
                best_val = (
                    val_loss
                )


                save_training_checkpoint(
                    path=(
                        args.output / "best.pt"
                    ),
                    step=completed_step,
                    model=model,
                    ema_model=ema_model,
                    optimizer=optimizer,
                    lr_scheduler=lr_scheduler,
                    stats=train_dataset.stats,
                    config=config,
                    best_val=best_val,
                )


            record = {
                "step": completed_step,
                "train_loss": loss.item(),
                "val_loss": val_loss,
                "best_val_loss": best_val,
                "learning_rate": (
                    optimizer.param_groups[0]["lr"]
                ),
                "elapsed_seconds": (
                    time.monotonic() - started
                ),
            }


            with metrics_path.open(
                "a"
            ) as f:
                f.write(
                    json.dumps(record)
                    + "\n"
                )


            print(
                f"Validation: {val_loss:.5f} | "
                f"Best: {best_val:.5f} | "
                f"New best: {improved}",
                flush=True,
            )


            model.train()


        if (
            completed_step % args.save_every == 0
        ):
            checkpoint_path = (
                args.output
                / f"step_{completed_step}.pt"
            )


            save_training_checkpoint(
                path=checkpoint_path,
                step=completed_step,
                model=model,
                ema_model=ema_model,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                stats=train_dataset.stats,
                config=config,
                best_val=best_val,
            )


            print(
                f"Saved: {checkpoint_path}",
                flush=True,
            )




    final_path = (
        args.output / "final.pt"
    )


    save_training_checkpoint(
        path=final_path,
        step=args.steps,
        model=model,
        ema_model=ema_model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        stats=train_dataset.stats,
        config=config,
        best_val=best_val,
    )


    print(
        f"TRAINING COMPLETE: {final_path}",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "ABC Chess Diffusion Policy"
        )
    )


    parser.add_argument(
        "--mode",
        choices=("train",),
        default="train",
    )


    parser.add_argument(
        "--abc-dir",
        type=Path,
        default=(
            Path.home()
            / "abcdp/external/abc"
        ),
    )


    parser.add_argument(
        "--steps",
        type=int,
        default=10000,
    )


    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
    )


    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
    )


    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
    )


    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
    )


    parser.add_argument(
        "--val-every",
        type=int,
        default=500,
    )


    parser.add_argument(
        "--val-batches",
        type=int,
        default=20,
    )


    parser.add_argument(
        "--val-samples",
        type=int,
        default=128,
    )


    parser.add_argument(
        "--val-seed",
        type=int,
        default=12345,
    )


    parser.add_argument(
        "--save-every",
        type=int,
        default=1000,
    )


    parser.add_argument(
        "--log-every",
        type=int,
        default=20,
    )


    parser.add_argument(
        "--ema-decay",
        type=float,
        default=0.995,
    )


    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )


    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "runs/chess"
        ),
    )


    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
    )


    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help=(
            "Use random ResNet weights "
            "for local smoke tests"
        ),
    )


    args = parser.parse_args()


    if args.mode == "train":
        train(args)




if __name__ == "__main__":
    main()
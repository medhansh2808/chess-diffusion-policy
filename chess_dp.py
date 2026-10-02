import argparse
import copy
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.models import ResNet18_Weights, resnet18

from abc_minimal.episode_io import discover_episodes, load_episode
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D


TASK_NAME = "sim_set_up_chess_pieces_on_the_board"
CAMERA_KEYS = ("top", "left", "right")

DEFAULT_OBSERVATION_HORIZON = 2
DEFAULT_PREDICTION_HORIZON = 20
DEFAULT_STATE_DIM = 14
DEFAULT_ACTION_DIM = 14
DEFAULT_VISION_FEATURE_DIM = 128

DEFAULT_DIFFUSION_STEPS = 100
DEFAULT_NOISE_SCHEDULE = "cosine"
DEFAULT_DIFFUSION_STEP_EMBED_DIM = 128
DEFAULT_UNET_DOWN_DIMS = (128, 256, 512)
DEFAULT_UNET_KERNEL_SIZE = 5
DEFAULT_UNET_N_GROUPS = 8
DEFAULT_ACTION_RANGE_EPS = 1e-4
DEFAULT_CLIP_SAMPLE = True
DEFAULT_CLIP_SAMPLE_RANGE = 1.0

REQUIRED_STAT_KEYS = (
    "state_mean",
    "state_std",
    "action_mean",
    "action_std",
    "action_min",
    "action_max",
)

OPTIONAL_STAT_KEYS = (
    "state_min",
    "state_max",
)


def preprocess_rgb_image(image, image_size):
    """
    Convert one RGB HWC image into a float CHW tensor in [0, 1].

    The resize preserves aspect ratio and pads to a square. This function is
    shared by training and rollout so the model sees identical preprocessing.
    """
    image = np.asarray(image)

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected RGB HWC image, got shape {image.shape}")

    if image_size < 1:
        raise ValueError("image_size must be positive")

    if image.dtype != np.uint8:
        if np.issubdtype(image.dtype, np.floating):
            if not np.isfinite(image).all():
                raise ValueError("Image contains NaN/Inf")
            max_value = float(image.max()) if image.size else 0.0
            if max_value <= 1.0:
                image = np.clip(image * 255.0, 0.0, 255.0).astype(np.uint8)
            else:
                image = np.clip(image, 0.0, 255.0).astype(np.uint8)
        else:
            image = np.clip(image, 0, 255).astype(np.uint8)

    height, width = image.shape[:2]

    if height < 1 or width < 1:
        raise ValueError(f"Invalid image shape: {image.shape}")

    scale = min(
        image_size / float(width),
        image_size / float(height),
    )

    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))

    resized = cv2.resize(
        image,
        (resized_width, resized_height),
        interpolation=cv2.INTER_LINEAR,
    )

    pad_left = (image_size - resized_width) // 2
    pad_right = image_size - resized_width - pad_left
    pad_top = (image_size - resized_height) // 2
    pad_bottom = image_size - resized_height - pad_top

    padded = cv2.copyMakeBorder(
        resized,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        borderType=cv2.BORDER_REPLICATE,
    )

    if padded.shape != (image_size, image_size, 3):
        raise RuntimeError(f"Unexpected preprocessed image shape: {padded.shape}")

    return (
        torch.from_numpy(np.ascontiguousarray(padded))
        .permute(2, 0, 1)
        .float()
        .div_(255.0)
    )


def augment_image_sequence(
    images,
    max_shift=8,
    brightness=0.10,
    contrast=0.10,
    saturation=0.10,
):
    """
    Mild augmentation for training only.

    images:
        (..., 3, H, W)

    A single spatial/color transform is shared across every camera and every
    observation timestep in the sample. This preserves temporal consistency.
    """
    if images.ndim < 4 or images.shape[-3] != 3:
        raise ValueError(f"Unexpected image tensor shape: {tuple(images.shape)}")

    original_shape = images.shape
    height, width = images.shape[-2:]

    flat = images.reshape(-1, 3, height, width)

    if max_shift > 0:
        padded = F.pad(
            flat,
            (max_shift, max_shift, max_shift, max_shift),
            mode="replicate",
        )

        offset_y = int(
            torch.randint(
                0,
                2 * max_shift + 1,
                (1,),
            ).item()
        )
        offset_x = int(
            torch.randint(
                0,
                2 * max_shift + 1,
                (1,),
            ).item()
        )

        flat = padded[
            :,
            :,
            offset_y:offset_y + height,
            offset_x:offset_x + width,
        ]

    brightness_factor = 1.0 + float(
        torch.empty(1).uniform_(-brightness, brightness).item()
    )
    contrast_factor = 1.0 + float(
        torch.empty(1).uniform_(-contrast, contrast).item()
    )
    saturation_factor = 1.0 + float(
        torch.empty(1).uniform_(-saturation, saturation).item()
    )

    flat = flat * brightness_factor

    spatial_mean = flat.mean(
        dim=(-2, -1),
        keepdim=True,
    )
    flat = (
        spatial_mean
        + contrast_factor * (flat - spatial_mean)
    )

    grayscale = flat.mean(
        dim=1,
        keepdim=True,
    )
    flat = (
        grayscale
        + saturation_factor * (flat - grayscale)
    )

    flat = flat.clamp(0.0, 1.0)

    return flat.reshape(original_shape)


def validate_episode_video(
    video_path,
    expected_frames,
    source_cameras,
):
    """
    Catch broken videos/camera metadata before training starts.
    """
    video_path = Path(video_path)

    if not video_path.is_file():
        raise FileNotFoundError(video_path)

    source_cameras = tuple(source_cameras)

    if not source_cameras:
        raise ValueError(f"No camera metadata for {video_path}")

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    try:
        reported_frames = int(
            round(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        )

        ok, first_frame = cap.read()
    finally:
        cap.release()

    if not ok:
        raise RuntimeError(f"Cannot decode first frame: {video_path}")

    if reported_frames > 0 and reported_frames < expected_frames:
        raise ValueError(
            f"Video/state mismatch for {video_path}: "
            f"video reports {reported_frames} frames, "
            f"but episode has {expected_frames} timesteps"
        )

    if first_frame.ndim != 3 or first_frame.shape[2] != 3:
        raise ValueError(
            f"Unexpected video frame shape for {video_path}: "
            f"{first_frame.shape}"
        )

    if first_frame.shape[0] % len(source_cameras) != 0:
        raise ValueError(
            f"Vertically stacked video height {first_frame.shape[0]} "
            f"is not divisible by {len(source_cameras)} cameras: "
            f"{source_cameras}"
        )


def load_camera_frame(
    episode_dir,
    frame_idx,
    source_cameras,
    image_size=224,
):
    """
    Decode one timestep from ABC's vertically stacked MP4.

    Returns:
        (3 cameras, 3 RGB channels, image_size, image_size)

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
            f"Missing required cameras. "
            f"Required={CAMERA_KEYS}, available={source_cameras}"
        )

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    try:
        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            int(frame_idx),
        )
        ok, frame = cap.read()
    finally:
        cap.release()

    if not ok:
        raise RuntimeError(
            f"Failed to decode frame {frame_idx}: {video_path}"
        )

    frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB,
    )

    num_cameras = len(source_cameras)

    if frame.shape[0] % num_cameras != 0:
        raise ValueError(
            f"Unexpected stacked video shape {frame.shape} "
            f"for {num_cameras} cameras"
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

        cameras[name] = preprocess_rgb_image(
            image,
            image_size=image_size,
        )

    return torch.stack(
        [cameras[name] for name in CAMERA_KEYS],
        dim=0,
    )


def normalize_states_array(states, stats):
    states = np.asarray(states, dtype=np.float32)
    return (
        states - stats["state_mean"]
    ) / stats["state_std"]


def normalize_actions_array(actions, stats, range_eps=DEFAULT_ACTION_RANGE_EPS):
    """Map training action limits to [-1, 1] per dimension.

    Diffusion sampling is substantially more stable when the variable being
    diffused lives in a known bounded range. Constant/near-constant dimensions
    map to exactly zero.
    """
    actions = np.asarray(actions, dtype=np.float32)
    action_min = np.asarray(stats["action_min"], dtype=np.float32)
    action_max = np.asarray(stats["action_max"], dtype=np.float32)
    span = action_max - action_min
    constant = span < float(range_eps)
    safe_span = np.where(constant, 1.0, span).astype(np.float32)
    normalized = 2.0 * (actions - action_min) / safe_span - 1.0
    if np.any(constant):
        normalized[..., constant] = 0.0
    if not np.isfinite(normalized).all():
        raise ValueError("Normalized actions contain NaN/Inf")
    return normalized.astype(np.float32, copy=False)


def denormalize_actions_array(actions, stats, range_eps=DEFAULT_ACTION_RANGE_EPS):
    """Invert normalize_actions_array.

    Inputs are clipped to [-1, 1], so a sampled action can never denormalize
    beyond the training action limits. Constant dimensions return the recorded
    constant value exactly.
    """
    actions = np.asarray(actions, dtype=np.float32)
    action_min = np.asarray(stats["action_min"], dtype=np.float32)
    action_max = np.asarray(stats["action_max"], dtype=np.float32)
    span = action_max - action_min
    constant = span < float(range_eps)
    clipped = np.clip(actions, -1.0, 1.0)
    restored = action_min + 0.5 * (clipped + 1.0) * span
    if np.any(constant):
        restored[..., constant] = action_min[constant]
    return restored.astype(np.float32, copy=False)


class ChessDataset(Dataset):
    def __init__(
        self,
        episode_dirs,
        observation_horizon=DEFAULT_OBSERVATION_HORIZON,
        prediction_horizon=DEFAULT_PREDICTION_HORIZON,
        stats=None,
        load_images=True,
        image_size=224,
        augment=False,
        validate_videos=True,
    ):
        super().__init__()

        if observation_horizon < 1:
            raise ValueError("Observation horizon must be positive")

        if prediction_horizon < 1:
            raise ValueError("Prediction horizon must be positive")

        self.observation_horizon = int(observation_horizon)
        self.prediction_horizon = int(prediction_horizon)
        self.load_images = bool(load_images)
        self.image_size = int(image_size)
        self.augment = bool(augment)

        self.episodes = []

        for episode_dir in episode_dirs:
            episode_dir = Path(episode_dir)

            metadata, states, actions = load_episode(
                episode_dir
            )

            states = np.asarray(
                states,
                dtype=np.float32,
            )
            actions = np.asarray(
                actions,
                dtype=np.float32,
            )

            if len(states) != len(actions):
                raise ValueError(
                    f"State/action length mismatch: {episode_dir}"
                )

            if (
                states.ndim != 2
                or states.shape[1] != DEFAULT_STATE_DIM
                or actions.ndim != 2
                or actions.shape[1] != DEFAULT_ACTION_DIM
            ):
                raise ValueError(
                    f"Expected 14D states/actions in {episode_dir}; "
                    f"got states={states.shape}, actions={actions.shape}"
                )

            if not np.isfinite(states).all():
                raise ValueError(
                    f"Non-finite state values in {episode_dir}"
                )

            if not np.isfinite(actions).all():
                raise ValueError(
                    f"Non-finite action values in {episode_dir}"
                )

            # We only train on samples that have a complete future action chunk.
            if len(states) < self.prediction_horizon:
                continue

            cameras = tuple(
                metadata.get("cameras", ())
            )

            if (
                self.load_images
                and not set(CAMERA_KEYS).issubset(cameras)
            ):
                raise ValueError(
                    f"Missing camera metadata in {episode_dir}. "
                    f"Required={CAMERA_KEYS}, available={cameras}"
                )

            if self.load_images and validate_videos:
                validate_episode_video(
                    episode_dir / "combined_camera-images-rgb.mp4",
                    expected_frames=len(states),
                    source_cameras=cameras,
                )

            self.episodes.append({
                "path": episode_dir,
                "cameras": cameras,
                "states": states,
                "actions": actions,
            })

        if not self.episodes:
            raise ValueError(
                "No usable episodes provided. "
                "Episodes must contain at least prediction_horizon timesteps."
            )

        if stats is None:
            all_states = np.concatenate(
                [ep["states"] for ep in self.episodes],
                axis=0,
            )
            all_actions = np.concatenate(
                [ep["actions"] for ep in self.episodes],
                axis=0,
            )

            stats = {
                "state_mean": all_states.mean(axis=0),
                "state_std": np.maximum(
                    all_states.std(axis=0),
                    1e-6,
                ),
                "action_mean": all_actions.mean(axis=0),
                "action_std": np.maximum(
                    all_actions.std(axis=0),
                    1e-6,
                ),
                "state_min": all_states.min(axis=0),
                "state_max": all_states.max(axis=0),
                "action_min": all_actions.min(axis=0),
                "action_max": all_actions.max(axis=0),
            }

        self.stats = {}

        for key in REQUIRED_STAT_KEYS:
            if key not in stats:
                raise KeyError(f"Missing normalization statistic: {key}")

            value = np.asarray(
                stats[key],
                dtype=np.float32,
            ).copy()

            if value.shape != (DEFAULT_STATE_DIM,):
                raise ValueError(
                    f"{key} must have shape (14,), got {value.shape}"
                )

            if not np.isfinite(value).all():
                raise ValueError(f"{key} contains NaN/Inf")

            self.stats[key] = value

        for key in OPTIONAL_STAT_KEYS:
            if key in stats:
                value = np.asarray(
                    stats[key],
                    dtype=np.float32,
                ).copy()

                if value.shape != (DEFAULT_STATE_DIM,):
                    raise ValueError(
                        f"{key} must have shape (14,), got {value.shape}"
                    )

                if not np.isfinite(value).all():
                    raise ValueError(f"{key} contains NaN/Inf")

                self.stats[key] = value

        if np.any(self.stats["state_std"] <= 0):
            raise ValueError("state_std must be strictly positive")

        if np.any(self.stats["action_std"] <= 0):
            raise ValueError("action_std must be strictly positive")

        self.state_mean = self.stats["state_mean"]
        self.state_std = self.stats["state_std"]
        self.action_mean = self.stats["action_mean"]
        self.action_std = self.stats["action_std"]

        # IMPORTANT:
        # Only timesteps with a complete prediction horizon become samples.
        self.indices = []

        for episode_idx, episode in enumerate(self.episodes):
            usable = (
                len(episode["states"])
                - self.prediction_horizon
                + 1
            )

            for t in range(usable):
                self.indices.append(
                    (episode_idx, t)
                )

        if not self.indices:
            raise ValueError("Dataset contains no complete action chunks")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        episode_idx, t = self.indices[index]
        episode = self.episodes[episode_idx]

        states = episode["states"]
        actions = episode["actions"]

        episode_length = len(states)

        obs_indices = np.arange(
            t - self.observation_horizon + 1,
            t + 1,
        )

        # Repeating the first observation is deliberate for the start of an
        # episode. Unlike the action horizon, this does not invent future data.
        obs_indices = np.clip(
            obs_indices,
            0,
            episode_length - 1,
        )

        action_indices = np.arange(
            t,
            t + self.prediction_horizon,
        )

        if action_indices[-1] >= episode_length:
            raise RuntimeError(
                "Internal dataset bug: incomplete action chunk reached __getitem__"
            )

        observation_states = normalize_states_array(
            states[obs_indices],
            self.stats,
        )

        action_sequence = normalize_actions_array(
            actions[action_indices],
            self.stats,
        )

        sample = {
            "states": torch.from_numpy(
                np.ascontiguousarray(observation_states)
            ),
            "actions": torch.from_numpy(
                np.ascontiguousarray(action_sequence)
            ),
        }

        if self.load_images:
            frame_cache = {}

            for frame_idx in obs_indices:
                frame_idx = int(frame_idx)

                if frame_idx not in frame_cache:
                    frame_cache[frame_idx] = load_camera_frame(
                        episode_dir=episode["path"],
                        frame_idx=frame_idx,
                        source_cameras=episode["cameras"],
                        image_size=self.image_size,
                    )

            images = torch.stack(
                [
                    frame_cache[int(idx)]
                    for idx in obs_indices
                ],
                dim=0,
            )

            if self.augment:
                images = augment_image_sequence(
                    images
                )

            sample["images"] = images

        return sample

    def denormalize_actions(self, actions):
        if isinstance(actions, torch.Tensor):
            action_min = torch.as_tensor(
                self.stats["action_min"],
                device=actions.device,
                dtype=actions.dtype,
            )
            action_max = torch.as_tensor(
                self.stats["action_max"],
                device=actions.device,
                dtype=actions.dtype,
            )
            span = action_max - action_min
            constant = span < DEFAULT_ACTION_RANGE_EPS
            clipped = actions.clamp(-1.0, 1.0)
            restored = action_min + 0.5 * (clipped + 1.0) * span
            if bool(constant.any()):
                restored[..., constant] = action_min[constant]
            return restored

        return denormalize_actions_array(
            actions,
            self.stats,
        )


class DDPMScheduler(nn.Module):
    def __init__(
        self,
        num_train_steps=DEFAULT_DIFFUSION_STEPS,
        beta_start=1e-4,
        beta_end=0.02,
        schedule=DEFAULT_NOISE_SCHEDULE,
        clip_sample=DEFAULT_CLIP_SAMPLE,
        clip_sample_range=DEFAULT_CLIP_SAMPLE_RANGE,
    ):
        super().__init__()

        if num_train_steps < 2:
            raise ValueError("At least two diffusion steps required")

        self.num_train_steps = int(num_train_steps)
        self.schedule = str(schedule)
        self.clip_sample = bool(clip_sample)
        self.clip_sample_range = float(clip_sample_range)
        if self.clip_sample_range <= 0:
            raise ValueError("clip_sample_range must be positive")

        if schedule == "linear":
            betas = torch.linspace(
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
                )
                * math.pi
                / 2
            ) ** 2

            alpha_bar = (
                alpha_bar / alpha_bar[0]
            )

            betas = (
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

        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(
            alphas,
            dim=0,
        )
        alpha_bars_prev = torch.cat([
            torch.ones(
                1,
                dtype=alpha_bars.dtype,
            ),
            alpha_bars[:-1],
        ])

        self.register_buffer(
            "betas",
            betas,
        )
        self.register_buffer(
            "alphas",
            alphas,
        )
        self.register_buffer(
            "alpha_bars",
            alpha_bars,
        )
        self.register_buffer(
            "alpha_bars_prev",
            alpha_bars_prev,
        )

    def add_noise(
        self,
        clean_actions,
        noise,
        timesteps,
    ):
        if clean_actions.shape != noise.shape:
            raise ValueError(
                f"clean/noise shape mismatch: "
                f"{clean_actions.shape} vs {noise.shape}"
            )

        if timesteps.ndim != 1 or timesteps.shape[0] != clean_actions.shape[0]:
            raise ValueError(
                f"Unexpected timestep shape: {timesteps.shape}"
            )

        alpha_bar = self.alpha_bars[
            timesteps.long()
        ].to(dtype=clean_actions.dtype)

        alpha_bar = alpha_bar[:, None, None]

        return (
            alpha_bar.sqrt()
            * clean_actions
            + (1.0 - alpha_bar).sqrt()
            * noise
        )

    def step(
        self,
        predicted_noise,
        timestep,
        sample,
    ):
        if predicted_noise.shape != sample.shape:
            raise ValueError(
                f"predicted_noise/sample shape mismatch: "
                f"{predicted_noise.shape} vs {sample.shape}"
            )

        t = int(timestep)

        if not 0 <= t < self.num_train_steps:
            raise ValueError(f"Invalid diffusion timestep: {t}")

        dtype = sample.dtype

        beta_t = self.betas[t].to(dtype=dtype)
        alpha_t = self.alphas[t].to(dtype=dtype)
        alpha_bar_t = self.alpha_bars[t].to(dtype=dtype)
        alpha_bar_prev = self.alpha_bars_prev[t].to(dtype=dtype)

        predicted_x0 = (
            sample
            - (1.0 - alpha_bar_t).sqrt()
            * predicted_noise
        ) / alpha_bar_t.sqrt()

        if not torch.isfinite(predicted_x0).all():
            raise RuntimeError(
                f"Non-finite predicted x0 at diffusion timestep {t}"
            )

        if self.clip_sample:
            predicted_x0 = predicted_x0.clamp(
                -self.clip_sample_range,
                self.clip_sample_range,
            )

        coefficient_x0 = (
            alpha_bar_prev.sqrt()
            * beta_t
            / (1.0 - alpha_bar_t)
        )

        coefficient_xt = (
            alpha_t.sqrt()
            * (1.0 - alpha_bar_prev)
            / (1.0 - alpha_bar_t)
        )

        mean = (
            coefficient_x0 * predicted_x0
            + coefficient_xt * sample
        )

        if t == 0:
            return mean

        variance = (
            beta_t
            * (1.0 - alpha_bar_prev)
            / (1.0 - alpha_bar_t)
        )

        return (
            mean
            + variance.clamp_min(0.0).sqrt()
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

            if not torch.isfinite(predicted_noise).all():
                raise RuntimeError(
                    f"Denoiser produced NaN/Inf at diffusion timestep {t}"
                )

            if predicted_noise.shape != actions.shape:
                raise RuntimeError(
                    f"Denoiser returned {predicted_noise.shape}; "
                    f"expected {actions.shape}"
                )

            actions = self.step(
                predicted_noise,
                t,
                actions,
            )

        if not torch.isfinite(actions).all():
            raise RuntimeError("Final sampled actions contain NaN/Inf")

        if self.clip_sample:
            actions = actions.clamp(
                -self.clip_sample_range,
                self.clip_sample_range,
            )

        return actions


class VisionEncoder(nn.Module):
    def __init__(
        self,
        feature_dim=DEFAULT_VISION_FEATURE_DIM,
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

        backbone.fc = nn.Identity()

        self.backbone = backbone
        self.freeze_backbone = bool(
            freeze_backbone
        )

        self.projection = nn.Sequential(
            nn.Linear(512, feature_dim),
            nn.ReLU(),
        )

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

        if self.freeze_backbone:
            self.backbone.requires_grad_(
                False
            )
            self.backbone.eval()

    def train(self, mode=True):
        super().train(mode)

        if self.freeze_backbone:
            self.backbone.eval()

        return self

    def forward(self, images):
        """
        images:
            (B, O, C, 3, H, W)

        C is the number of cameras.
        """
        if images.ndim != 6:
            raise ValueError(
                f"Expected 6D image tensor, got {images.shape}"
            )

        B, O, C, RGB, H, W = images.shape

        if C != len(CAMERA_KEYS):
            raise ValueError(
                f"Expected {len(CAMERA_KEYS)} cameras, got {C}"
            )

        if RGB != 3:
            raise ValueError(
                f"Expected RGB images, got {RGB} channels"
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
                features = self.backbone(
                    images
                )
        else:
            features = self.backbone(
                images
            )

        features = self.projection(
            features
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
        observation_horizon=DEFAULT_OBSERVATION_HORIZON,
        prediction_horizon=DEFAULT_PREDICTION_HORIZON,
        state_dim=DEFAULT_STATE_DIM,
        action_dim=DEFAULT_ACTION_DIM,
        vision_feature_dim=DEFAULT_VISION_FEATURE_DIM,
        pretrained_vision=True,
        freeze_backbone=True,
        num_diffusion_steps=DEFAULT_DIFFUSION_STEPS,
        noise_schedule=DEFAULT_NOISE_SCHEDULE,
        diffusion_step_embed_dim=DEFAULT_DIFFUSION_STEP_EMBED_DIM,
        unet_down_dims=DEFAULT_UNET_DOWN_DIMS,
        unet_kernel_size=DEFAULT_UNET_KERNEL_SIZE,
        unet_n_groups=DEFAULT_UNET_N_GROUPS,
        clip_sample=DEFAULT_CLIP_SAMPLE,
        clip_sample_range=DEFAULT_CLIP_SAMPLE_RANGE,
    ):
        super().__init__()

        self.observation_horizon = int(
            observation_horizon
        )
        self.prediction_horizon = int(
            prediction_horizon
        )
        self.state_dim = int(
            state_dim
        )
        self.action_dim = int(
            action_dim
        )
        self.vision_feature_dim = int(
            vision_feature_dim
        )

        self.vision_encoder = VisionEncoder(
            feature_dim=self.vision_feature_dim,
            pretrained=pretrained_vision,
            freeze_backbone=freeze_backbone,
        )

        condition_dim = (
            self.observation_horizon
            * (
                len(CAMERA_KEYS)
                * self.vision_feature_dim
                + self.state_dim
            )
        )

        self.condition_dim = int(
            condition_dim
        )

        self.unet = ConditionalUnet1D(
            input_dim=self.action_dim,
            global_cond_dim=self.condition_dim,
            diffusion_step_embed_dim=int(
                diffusion_step_embed_dim
            ),
            down_dims=[
                int(value)
                for value in unet_down_dims
            ],
            kernel_size=int(
                unet_kernel_size
            ),
            n_groups=int(
                unet_n_groups
            ),
        )

        self.scheduler = DDPMScheduler(
            num_train_steps=int(
                num_diffusion_steps
            ),
            schedule=str(
                noise_schedule
            ),
            clip_sample=bool(
                clip_sample
            ),
            clip_sample_range=float(
                clip_sample_range
            ),
        )

    def encode_observations(
        self,
        states,
        images,
    ):
        if states.ndim != 3:
            raise ValueError(
                f"Expected states (B,O,D), got {states.shape}"
            )

        if (
            states.shape[1]
            != self.observation_horizon
            or states.shape[2]
            != self.state_dim
        ):
            raise ValueError(
                f"Unexpected state shape {states.shape}; "
                f"expected (*,{self.observation_horizon},{self.state_dim})"
            )

        if (
            images.ndim != 6
            or images.shape[0] != states.shape[0]
            or images.shape[1] != self.observation_horizon
            or images.shape[2] != len(CAMERA_KEYS)
            or images.shape[3] != 3
        ):
            raise ValueError(
                f"Unexpected image shape: {images.shape}"
            )

        visual_features = self.vision_encoder(
            images
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

        condition = condition.flatten(
            start_dim=1
        )

        if condition.shape != (
            B,
            self.condition_dim,
        ):
            raise RuntimeError(
                f"Condition shape {condition.shape}; "
                f"expected {(B, self.condition_dim)}"
            )

        return condition

    def forward(
        self,
        states,
        images,
        actions,
    ):
        if (
            actions.ndim != 3
            or actions.shape[1]
            != self.prediction_horizon
            or actions.shape[2]
            != self.action_dim
        ):
            raise ValueError(
                f"Unexpected action shape {actions.shape}; "
                f"expected (*,{self.prediction_horizon},{self.action_dim})"
            )

        condition = self.encode_observations(
            states,
            images,
        )

        batch_size = actions.shape[0]

        timesteps = torch.randint(
            0,
            self.scheduler.num_train_steps,
            (batch_size,),
            device=actions.device,
            dtype=torch.long,
        )

        noise = torch.randn_like(
            actions
        )

        noisy_actions = self.scheduler.add_noise(
            actions,
            noise,
            timesteps,
        )

        predicted_noise = self.unet(
            noisy_actions,
            timesteps,
            global_cond=condition,
        )

        if predicted_noise.shape != noise.shape:
            raise RuntimeError(
                f"U-Net returned {predicted_noise.shape}; "
                f"expected {noise.shape}"
            )

        return (
            predicted_noise - noise
        ).square().mean()

    @torch.no_grad()
    def predict_actions(
        self,
        states,
        images,
    ):
        condition = self.encode_observations(
            states,
            images,
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

    # Includes frozen BatchNorm buffers and the diffusion schedule buffers.
    for ema_buffer, buffer in zip(
        ema_model.buffers(),
        model.buffers(),
    ):
        ema_buffer.copy_(buffer)


@torch.no_grad()
def validate_noise_loss(
    model,
    val_loader,
    device,
    max_batches=20,
    seed=12345,
):
    model.eval()

    total_loss = 0.0
    total_samples = 0

    cuda_devices = (
        [torch.cuda.current_device()]
        if device.type == "cuda"
        else []
    )

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

            loss = model(
                states,
                images,
                actions,
            )

            if not torch.isfinite(loss):
                raise RuntimeError(
                    "Non-finite validation noise loss"
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


@torch.no_grad()
def validate_generated_actions(
    model,
    val_loader,
    stats,
    device,
    max_batches=1,
    seed=54321,
):
    """
    Run the actual reverse diffusion process and compare generated action chunks
    with their target chunks.

    This is especially important for --overfit-one. A tiny denoising loss is not
    enough; the complete sampler must reconstruct the memorized action chunk.
    """
    model.eval()

    action_min = torch.as_tensor(
        stats["action_min"],
        device=device,
        dtype=torch.float32,
    ).view(1, 1, -1)

    action_max = torch.as_tensor(
        stats["action_max"],
        device=device,
        dtype=torch.float32,
    ).view(1, 1, -1)
    action_span = action_max - action_min
    constant_action_dim = action_span < DEFAULT_ACTION_RANGE_EPS

    total_norm_sq = 0.0
    total_action_sq = 0.0
    total_values = 0

    per_dim_sq = torch.zeros(
        model.action_dim,
        device=device,
        dtype=torch.float64,
    )
    per_dim_count = 0

    prediction_min = float("inf")
    prediction_max = float("-inf")
    target_min = float("inf")
    target_max = float("-inf")
    normalized_prediction_min = float("inf")
    normalized_prediction_max = float("-inf")

    cuda_devices = (
        [torch.cuda.current_device()]
        if device.type == "cuda"
        else []
    )

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
            target = batch["actions"].to(
                device,
                non_blocking=True,
            )

            prediction = model.predict_actions(
                states,
                images,
            )

            if prediction.shape != target.shape:
                raise RuntimeError(
                    f"Generated/target shape mismatch: "
                    f"{prediction.shape} vs {target.shape}"
                )

            if not torch.isfinite(prediction).all():
                raise RuntimeError(
                    "Generated actions contain NaN/Inf"
                )

            normalized_prediction_min = min(
                normalized_prediction_min,
                float(prediction.min().item()),
            )
            normalized_prediction_max = max(
                normalized_prediction_max,
                float(prediction.max().item()),
            )

            normalized_error = (
                prediction - target
            ).square()

            prediction_clipped = prediction.clamp(-1.0, 1.0)
            target_clipped = target.clamp(-1.0, 1.0)
            prediction_actions = (
                action_min
                + 0.5 * (prediction_clipped + 1.0) * action_span
            )
            target_actions = (
                action_min
                + 0.5 * (target_clipped + 1.0) * action_span
            )
            if bool(constant_action_dim.any()):
                prediction_actions[..., constant_action_dim.view(-1)] = (
                    action_min.view(-1)[constant_action_dim.view(-1)]
                )
                target_actions[..., constant_action_dim.view(-1)] = (
                    action_min.view(-1)[constant_action_dim.view(-1)]
                )

            action_error = (
                prediction_actions
                - target_actions
            ).square()

            total_norm_sq += float(
                normalized_error.sum().item()
            )
            total_action_sq += float(
                action_error.sum().item()
            )
            total_values += int(
                action_error.numel()
            )

            per_dim_sq += (
                action_error.double().sum(
                    dim=(0, 1)
                )
            )
            per_dim_count += int(
                action_error.shape[0]
                * action_error.shape[1]
            )

            prediction_min = min(
                prediction_min,
                float(
                    prediction_actions.min().item()
                ),
            )
            prediction_max = max(
                prediction_max,
                float(
                    prediction_actions.max().item()
                ),
            )
            target_min = min(
                target_min,
                float(
                    target_actions.min().item()
                ),
            )
            target_max = max(
                target_max,
                float(
                    target_actions.max().item()
                ),
            )

    if total_values == 0 or per_dim_count == 0:
        raise RuntimeError(
            "Generated-action validation loader is empty"
        )

    return {
        "sample_normalized_mse":
            total_norm_sq / total_values,
        "sample_action_mse":
            total_action_sq / total_values,
        "sample_action_mse_per_dim":
            (
                per_dim_sq
                / per_dim_count
            ).cpu().tolist(),
        "sample_prediction_min":
            prediction_min,
        "sample_prediction_max":
            prediction_max,
        "sample_target_min":
            target_min,
        "sample_target_max":
            target_max,
        "sample_normalized_prediction_min":
            normalized_prediction_min,
        "sample_normalized_prediction_max":
            normalized_prediction_max,
    }


def save_training_checkpoint(
    path,
    step,
    model,
    ema_model,
    optimizer,
    lr_scheduler,
    stats,
    config,
    best_metric,
    best_metric_name,
):
    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint = {
        "step": int(step),
        "model": model.state_dict(),
        "ema": ema_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_scheduler":
            lr_scheduler.state_dict(),
        "stats": {
            key: np.asarray(value).copy()
            for key, value in stats.items()
        },
        "config": config,
        "best_metric": float(best_metric),
        "best_metric_name": str(best_metric_name),
        # Kept for convenient inspection by older local utilities.
        "best_val": float(best_metric),
        "torch_rng_state":
            torch.get_rng_state(),
    }

    if torch.cuda.is_available():
        checkpoint["cuda_rng_state"] = (
            torch.cuda.get_rng_state_all()
        )

    temporary_path = (
        path.with_suffix(
            path.suffix + ".tmp"
        )
    )

    torch.save(
        checkpoint,
        temporary_path,
    )

    temporary_path.replace(path)


def clear_fresh_run_artifacts(output):
    """
    Delete only files created by this training script.

    This is called only when --overwrite-output is explicitly supplied.
    """
    candidates = [
        output / "metrics.jsonl",
        output / "best.pt",
        output / "final.pt",
    ]

    candidates.extend(
        output.glob("step_*.pt")
    )

    for path in candidates:
        if path.is_file():
            path.unlink()


def train(args):
    if (
        args.overfit_one
        and args.overfit_episode
    ):
        raise ValueError(
            "Choose only one overfitting mode"
        )

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

    if args.sample_val_batches < 1:
        raise ValueError(
            "--sample-val-batches must be positive"
        )

    torch.manual_seed(
        args.seed
    )
    np.random.seed(
        args.seed
    )

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

    metrics_path = (
        args.output / "metrics.jsonl"
    )

    if args.resume is None:
        existing_training_files = (
            metrics_path.exists()
            or (args.output / "best.pt").exists()
            or (args.output / "final.pt").exists()
            or any(args.output.glob("step_*.pt"))
        )

        if existing_training_files:
            if not args.overwrite_output:
                raise RuntimeError(
                    f"{args.output} already contains training artifacts. "
                    "Use a new --output, pass --resume, or explicitly pass "
                    "--overwrite-output."
                )

            clear_fresh_run_artifacts(
                args.output
            )

    elif not args.resume.is_file():
        raise FileNotFoundError(
            args.resume
        )

    started = time.monotonic()

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

    if not train_paths:
        raise RuntimeError(
            "Missing training chess episodes"
        )

    if (
        not args.overfit_one
        and not args.overfit_episode
        and not val_paths
    ):
        raise RuntimeError(
            "Missing validation chess episodes"
        )

    train_names = {
        path.name
        for path in train_paths
    }
    val_names = {
        path.name
        for path in val_paths
    }

    if (
        val_paths
        and train_names & val_names
    ):
        raise RuntimeError(
            "Training and validation episodes overlap"
        )

    if args.overfit_episode:
        if not (
            0
            <= args.episode_index
            < len(train_paths)
        ):
            raise ValueError(
                "--episode-index is outside "
                "the training episode list"
            )

        train_paths = [
            train_paths[
                args.episode_index
            ]
        ]

        print(
            f"Overfitting episode: "
            f"{train_paths[0].name}",
            flush=True,
        )

    # Augmentation is part of the real training pipeline, but deliberately
    # disabled for memorization diagnostics.
    train_augmentation = (
        not args.no_augmentation
        and not args.overfit_one
        and not args.overfit_episode
    )

    print(
        f"Training augmentation: "
        f"{train_augmentation}",
        flush=True,
    )

    train_dataset = ChessDataset(
        episode_dirs=train_paths,
        observation_horizon=(
            DEFAULT_OBSERVATION_HORIZON
        ),
        prediction_horizon=(
            DEFAULT_PREDICTION_HORIZON
        ),
        load_images=True,
        image_size=args.image_size,
        augment=train_augmentation,
        validate_videos=True,
    )

    # Overfit diagnostics use the exact training data for validation.
    if args.overfit_one:
        first_episode_dataset_indices = [
            dataset_index
            for dataset_index, (
                episode_idx,
                _
            ) in enumerate(
                train_dataset.indices
            )
            if episode_idx == 0
        ]

        if not first_episode_dataset_indices:
            raise RuntimeError(
                "First episode has no valid full-horizon sample"
            )

        sample_dataset_index = (
            first_episode_dataset_indices[
                len(
                    first_episode_dataset_indices
                )
                // 2
            ]
        )

        episode_idx, timestep = (
            train_dataset.indices[
                sample_dataset_index
            ]
        )

        train_samples = Subset(
            train_dataset,
            [sample_dataset_index],
        )
        val_subset = Subset(
            train_dataset,
            [sample_dataset_index],
        )

        print(
            f"Overfitting dataset sample "
            f"{sample_dataset_index} "
            f"(episode={episode_idx}, timestep={timestep})",
            flush=True,
        )

        val_dataset = None

    elif args.overfit_episode:
        train_samples = train_dataset

        indices = np.linspace(
            0,
            len(train_dataset) - 1,
            min(
                args.val_samples,
                len(train_dataset),
            ),
            dtype=int,
        ).tolist()

        val_subset = Subset(
            train_dataset,
            indices,
        )

        print(
            f"Episode training windows: "
            f"{len(train_dataset)}",
            flush=True,
        )

        val_dataset = None

    else:
        train_samples = train_dataset

        val_dataset = ChessDataset(
            episode_dirs=val_paths,
            observation_horizon=(
                train_dataset.observation_horizon
            ),
            prediction_horizon=(
                train_dataset.prediction_horizon
            ),
            stats=train_dataset.stats,
            load_images=True,
            image_size=args.image_size,
            augment=False,
            validate_videos=True,
        )

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

    loader_generator = torch.Generator()
    loader_generator.manual_seed(
        args.seed
    )

    train_loader = DataLoader(
        train_samples,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(
            device.type == "cuda"
        ),
        generator=loader_generator,
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
        f"Training samples: "
        f"{len(train_samples)}",
        flush=True,
    )

    print(
        f"Fixed validation samples: "
        f"{len(val_subset)}",
        flush=True,
    )

    for key in REQUIRED_STAT_KEYS:
        value = train_dataset.stats[key]
        if not np.isfinite(value).all():
            raise RuntimeError(
                f"Non-finite training statistic: {key}"
            )

    print(
        "Action range per dimension:",
        flush=True,
    )

    if (
        "action_min"
        in train_dataset.stats
        and "action_max"
        in train_dataset.stats
    ):
        for index, (
            low,
            high,
        ) in enumerate(
            zip(
                train_dataset.stats[
                    "action_min"
                ],
                train_dataset.stats[
                    "action_max"
                ],
            )
        ):
            print(
                f"  action[{index:02d}] "
                f"min={low:+.5f} "
                f"max={high:+.5f}",
                flush=True,
            )

    use_pretrained = (
        not args.no_pretrained
        and args.resume is None
    )

    model = ChessDiffusionPolicy(
        observation_horizon=(
            train_dataset.observation_horizon
        ),
        prediction_horizon=(
            train_dataset.prediction_horizon
        ),
        state_dim=DEFAULT_STATE_DIM,
        action_dim=DEFAULT_ACTION_DIM,
        vision_feature_dim=(
            DEFAULT_VISION_FEATURE_DIM
        ),
        pretrained_vision=use_pretrained,
        freeze_backbone=True,
        num_diffusion_steps=(
            DEFAULT_DIFFUSION_STEPS
        ),
        noise_schedule=(
            DEFAULT_NOISE_SCHEDULE
        ),
        diffusion_step_embed_dim=(
            DEFAULT_DIFFUSION_STEP_EMBED_DIM
        ),
        unet_down_dims=(
            DEFAULT_UNET_DOWN_DIMS
        ),
        unet_kernel_size=(
            DEFAULT_UNET_KERNEL_SIZE
        ),
        unet_n_groups=(
            DEFAULT_UNET_N_GROUPS
        ),
        clip_sample=DEFAULT_CLIP_SAMPLE,
        clip_sample_range=(
            DEFAULT_CLIP_SAMPLE_RANGE
        ),
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
        "camera_keys": CAMERA_KEYS,
        "image_size": int(
            args.image_size
        ),
        "observation_horizon": int(
            train_dataset.observation_horizon
        ),
        "prediction_horizon": int(
            train_dataset.prediction_horizon
        ),
        "state_dim": DEFAULT_STATE_DIM,
        "action_dim": DEFAULT_ACTION_DIM,
        "vision_feature_dim":
            DEFAULT_VISION_FEATURE_DIM,
        "freeze_backbone": True,
        "pretrained_vision": (
            not args.no_pretrained
        ),
        "num_diffusion_steps":
            DEFAULT_DIFFUSION_STEPS,
        "noise_schedule":
            DEFAULT_NOISE_SCHEDULE,
        "diffusion_step_embed_dim":
            DEFAULT_DIFFUSION_STEP_EMBED_DIM,
        "unet_down_dims":
            tuple(
                DEFAULT_UNET_DOWN_DIMS
            ),
        "unet_kernel_size":
            DEFAULT_UNET_KERNEL_SIZE,
        "unet_n_groups":
            DEFAULT_UNET_N_GROUPS,
        "action_normalization":
            "limits_-1_1",
        "action_range_eps":
            DEFAULT_ACTION_RANGE_EPS,
        "clip_sample":
            DEFAULT_CLIP_SAMPLE,
        "clip_sample_range":
            DEFAULT_CLIP_SAMPLE_RANGE,
        "checkpoint_selection": (
            "ema_generated_normalized_mse"
            if (args.overfit_one or args.overfit_episode)
            else "val_noise_loss"
        ),
        "batch_size": int(
            args.batch_size
        ),
        "learning_rate": float(
            args.lr
        ),
        "total_steps": int(
            args.steps
        ),
        "ema_decay": float(
            args.ema_decay
        ),
        "seed": int(
            args.seed
        ),
        "val_samples": int(
            args.val_samples
        ),
        "val_seed": int(
            args.val_seed
        ),
        "sample_val_seed": int(
            args.sample_val_seed
        ),
        "train_augmentation": bool(
            train_augmentation
        ),
        "overfit_one": bool(
            args.overfit_one
        ),
        "overfit_episode": bool(
            args.overfit_episode
        ),
        "overfit_episode_name": (
            train_paths[0].name
            if args.overfit_episode
            else None
        ),
        "episode_index": (
            int(args.episode_index)
            if args.overfit_episode
            else None
        ),
    }

    start_step = 0
    best_metric = float("inf")
    best_metric_name = config["checkpoint_selection"]

    if args.resume is not None:
        checkpoint = torch.load(
            args.resume,
            map_location=device,
            weights_only=False,
        )

        saved_config = checkpoint[
            "config"
        ]

        strict_resume_keys = (
            "task",
            "camera_keys",
            "image_size",
            "observation_horizon",
            "prediction_horizon",
            "state_dim",
            "action_dim",
            "vision_feature_dim",
            "freeze_backbone",
            "pretrained_vision",
            "num_diffusion_steps",
            "noise_schedule",
            "diffusion_step_embed_dim",
            "unet_down_dims",
            "unet_kernel_size",
            "unet_n_groups",
            "action_normalization",
            "action_range_eps",
            "clip_sample",
            "clip_sample_range",
            "checkpoint_selection",
            "batch_size",
            "learning_rate",
            "total_steps",
            "ema_decay",
            "seed",
            "val_samples",
            "val_seed",
            "sample_val_seed",
            "train_augmentation",
            "overfit_one",
            "overfit_episode",
            "overfit_episode_name",
            "episode_index",
        )

        for key in strict_resume_keys:
            if (
                key not in saved_config
                or saved_config[key]
                != config[key]
            ):
                raise ValueError(
                    f"Resume configuration mismatch: {key}. "
                    f"saved={saved_config.get(key)!r}, "
                    f"current={config.get(key)!r}"
                )

        for key in REQUIRED_STAT_KEYS:
            original = np.asarray(
                checkpoint["stats"][key],
                dtype=np.float32,
            )
            current = train_dataset.stats[
                key
            ]

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
            checkpoint["model"],
            strict=True,
        )
        ema_model.load_state_dict(
            checkpoint["ema"],
            strict=True,
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
        saved_best_metric_name = checkpoint.get(
            "best_metric_name",
            checkpoint.get(
                "config", {}
            ).get(
                "checkpoint_selection",
                best_metric_name,
            ),
        )
        if saved_best_metric_name != best_metric_name:
            raise ValueError(
                "Resume checkpoint selection metric mismatch: "
                f"saved={saved_best_metric_name!r}, current={best_metric_name!r}"
            )
        best_metric = float(
            checkpoint.get(
                "best_metric",
                checkpoint.get(
                    "best_val",
                    float("inf"),
                ),
            )
        )

        if "torch_rng_state" in checkpoint:
            torch.set_rng_state(
                checkpoint[
                    "torch_rng_state"
                ].cpu()
            )

        if (
            device.type == "cuda"
            and "cuda_rng_state"
            in checkpoint
        ):
            torch.cuda.set_rng_state_all(
                checkpoint[
                    "cuda_rng_state"
                ]
            )

        print(
            f"Resumed from step "
            f"{start_step}",
            flush=True,
        )

    model.train()

    train_iterator = iter(
        train_loader
    )

    running_loss_sum = 0.0
    running_loss_count = 0

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

        optimizer.zero_grad(
            set_to_none=True
        )

        loss = model(
            states,
            images,
            actions,
        )

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Non-finite loss at "
                f"step {step + 1}"
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

        running_loss_sum += float(
            loss.item()
        )
        running_loss_count += 1

        if (
            completed_step == 1
            or completed_step
            % args.log_every
            == 0
        ):
            running_mean = (
                running_loss_sum
                / max(
                    running_loss_count,
                    1,
                )
            )

            print(
                f"Step {completed_step}/{args.steps} | "
                f"Loss {loss.item():.5f} | "
                f"Running {running_mean:.5f} | "
                f"Grad {grad_norm.item():.3f} | "
                f"LR "
                f"{optimizer.param_groups[0]['lr']:.2e}",
                flush=True,
            )

        should_validate = (
            completed_step
            % args.val_every
            == 0
            or completed_step
            == args.steps
        )

        if should_validate:
            mean_train_loss = (
                running_loss_sum
                / max(
                    running_loss_count,
                    1,
                )
            )

            val_loss = validate_noise_loss(
                model=ema_model,
                val_loader=val_loader,
                device=device,
                max_batches=args.val_batches,
                seed=args.val_seed,
            )

            sample_metrics = validate_generated_actions(
                model=ema_model,
                val_loader=val_loader,
                stats=train_dataset.stats,
                device=device,
                max_batches=args.sample_val_batches,
                seed=args.sample_val_seed,
            )

            model_sample_metrics = None
            if args.overfit_one or args.overfit_episode:
                model_sample_metrics = validate_generated_actions(
                    model=model,
                    val_loader=val_loader,
                    stats=train_dataset.stats,
                    device=device,
                    max_batches=args.sample_val_batches,
                    seed=args.sample_val_seed,
                )

            if best_metric_name == "ema_generated_normalized_mse":
                current_metric = float(
                    sample_metrics["sample_normalized_mse"]
                )
            elif best_metric_name == "val_noise_loss":
                current_metric = float(val_loss)
            else:
                raise RuntimeError(
                    f"Unknown checkpoint selection metric: {best_metric_name}"
                )

            improved = current_metric < best_metric

            if improved:
                best_metric = current_metric

                save_training_checkpoint(
                    path=(
                        args.output
                        / "best.pt"
                    ),
                    step=completed_step,
                    model=model,
                    ema_model=ema_model,
                    optimizer=optimizer,
                    lr_scheduler=lr_scheduler,
                    stats=train_dataset.stats,
                    config=config,
                    best_metric=best_metric,
                    best_metric_name=best_metric_name,
                )

            record = {
                "step": completed_step,
                "train_loss_mean": mean_train_loss,
                "val_noise_loss": val_loss,
                "checkpoint_metric_name": best_metric_name,
                "checkpoint_metric": current_metric,
                "best_checkpoint_metric": best_metric,
                **sample_metrics,
                "learning_rate": (
                    optimizer.param_groups[
                        0
                    ]["lr"]
                ),
                "elapsed_seconds": (
                    time.monotonic()
                    - started
                ),
            }
            if model_sample_metrics is not None:
                record.update({
                    f"raw_model_{key}": value
                    for key, value in model_sample_metrics.items()
                })

            with metrics_path.open(
                "a"
            ) as f:
                f.write(
                    json.dumps(record)
                    + "\n"
                )

            print(
                f"Validation noise: {val_loss:.6f} | "
                f"checkpoint metric ({best_metric_name}): "
                f"{current_metric:.6f} | "
                f"best: {best_metric:.6f} | "
                f"new best: {improved}",
                flush=True,
            )

            print(
                f"Generated chunk normalized MSE: "
                f"{sample_metrics['sample_normalized_mse']:.6f} | "
                f"action-unit MSE: "
                f"{sample_metrics['sample_action_mse']:.6f}",
                flush=True,
            )

            print(
                f"Generated normalized range: "
                f"[{sample_metrics['sample_normalized_prediction_min']:+.5f}, "
                f"{sample_metrics['sample_normalized_prediction_max']:+.5f}]",
                flush=True,
            )

            print(
                f"Generated action range: "
                f"[{sample_metrics['sample_prediction_min']:+.5f}, "
                f"{sample_metrics['sample_prediction_max']:+.5f}] | "
                f"target range: "
                f"[{sample_metrics['sample_target_min']:+.5f}, "
                f"{sample_metrics['sample_target_max']:+.5f}]",
                flush=True,
            )

            if model_sample_metrics is not None:
                print(
                    f"Raw-model generated normalized MSE: "
                    f"{model_sample_metrics['sample_normalized_mse']:.6f} | "
                    f"EMA: {sample_metrics['sample_normalized_mse']:.6f}",
                    flush=True,
                )

            if (
                args.overfit_one
                or args.overfit_episode
            ):
                formatted = ", ".join(
                    f"{value:.6g}"
                    for value in sample_metrics[
                        "sample_action_mse_per_dim"
                    ]
                )

                print(
                    f"Per-dimension action MSE: "
                    f"[{formatted}]",
                    flush=True,
                )

            running_loss_sum = 0.0
            running_loss_count = 0

            model.train()

        if (
            completed_step
            % args.save_every
            == 0
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
                best_metric=best_metric,
                best_metric_name=best_metric_name,
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
        best_metric=best_metric,
        best_metric_name=best_metric_name,
    )

    print(
        f"TRAINING COMPLETE: "
        f"{final_path}",
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
        "--sample-val-batches",
        type=int,
        default=1,
        help=(
            "Number of validation batches on which "
            "to run full reverse diffusion"
        ),
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
        "--sample-val-seed",
        type=int,
        default=54321,
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
        "--overwrite-output",
        action="store_true",
        help=(
            "Delete old chess training artifacts "
            "inside --output before a fresh run"
        ),
    )

    parser.add_argument(
        "--overfit-one",
        action="store_true",
    )

    parser.add_argument(
        "--overfit-episode",
        action="store_true",
        help=(
            "Train on all full action windows "
            "from one training demonstration"
        ),
    )

    parser.add_argument(
        "--episode-index",
        type=int,
        default=0,
        help=(
            "Index in the sorted training "
            "episode list"
        ),
    )

    parser.add_argument(
        "--no-augmentation",
        action="store_true",
        help=(
            "Disable image augmentation for normal training. "
            "Augmentation is always disabled automatically "
            "for overfit diagnostics."
        ),
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

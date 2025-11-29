"""
Dataset class for loading Franka arm trajectories with IMAGES from .npz files.

Each .npz file contains:
- obs: array of dicts with keys: image, qpos, text
- actions: (T, 8) array of control commands

This dataset returns:
- obs: dict with 'image' (T, 3, H, W), 'qpos' (T, 7), and 'text' (T,) - ONLY ARM JOINTS
- action: (T, 8) - 7 arm joints + 1 gripper (gripper always open)

LAZY LOADING: Images are loaded on-demand from .npz files to prevent OOM.
"""

from typing import Dict
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import copy

from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import SequenceSampler, get_val_mask, downsample_mask
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.common.normalize_util import get_image_range_normalizer


class FrankaNpzImageDataset(BaseImageDataset):
    """
    Loads Franka arm trajectories with images from .npz files.
    
    Observations:
    - image: (T, 3, H, W) - RGB images from camera
    - qpos: (T, 7) - ARM JOINT POSITIONS ONLY (no gripper, no cube)
    - text: (T,) - Task description text (e.g., "red block")
    
    Actions: (T, 8) - 7 arm joints + 1 gripper (gripper always open)
    """
    
    def __init__(
        self,
        data_dir: str,
        horizon: int = 16,
        pad_before: int = 0,
        pad_after: int = 0,
        seed: int = 42,
        val_ratio: float = 0.1,
        max_train_episodes: int = None,
        image_size: tuple = (240, 320),  # Original image size from collection
    ):
        super().__init__()
        
        data_dir = Path(data_dir)
        npz_files = sorted(data_dir.glob("pick_place_episode_*.npz"))
        
        if len(npz_files) == 0:
            raise ValueError(f"No .npz files found in {data_dir}")
        
        print(f"Found {len(npz_files)} episodes in {data_dir}")
        print("Loading episode metadata (lazy image loading enabled)...")
        
        # Create replay buffer - LAZY LOADING: only load qpos, text, action (not images)
        replay_buffer = ReplayBuffer.create_empty_numpy()
        
        # Store metadata for lazy image loading:
        # - episode_file_idx[i] = which .npz file episode i comes from
        # - episode_start_idx[i] = start index of episode i in the replay buffer
        # - episode_lengths[i] = length of episode i
        self.episode_file_indices = []  # Which file each episode comes from
        self.episode_start_indices = []  # Start index of each episode in replay buffer
        self.episode_lengths = []  # Length of each episode
        self.npz_files = npz_files  # Store file paths
        
        for file_idx, npz_path in enumerate(tqdm(npz_files, desc="Loading episode metadata")):
            data = np.load(npz_path, allow_pickle=True)
            
            # Extract observations and actions
            obs_list = data["obs"]  # array of dicts
            actions = data["actions"].astype(np.float32)  # (T, 8)
            
            # Extract qpos and text (small, can load eagerly)
            # DO NOT load images here - they will be loaded on-demand
            qpos_list = []
            text_list = []
            
            for obs_dict in obs_list:
                # Qpos: Extract 7 arm joints (handle both 7D and 16D formats)
                qpos = obs_dict["qpos"].astype(np.float32)
                if qpos.shape[0] > 7:
                    qpos_arm = qpos[:7]  # (7,) - take first 7 if larger
                else:
                    qpos_arm = qpos  # (7,) - already correct size
                qpos_list.append(qpos_arm)
                
                # Extract text label (if present) - same as dataset_franka_npz.py
                text_label = obs_dict.get("text", "")  # Default to empty string if not present
                text_list.append(text_label)
            
            qpos_array = np.array(qpos_list, dtype=np.float32)  # (T, 7)
            text_array = np.array(text_list, dtype=object)  # (T,) - array of strings
            
            # Store episode metadata (before adding to buffer)
            # Get current buffer length from episode_ends
            if replay_buffer.n_episodes > 0:
                episode_start = replay_buffer.episode_ends[-1]
            else:
                episode_start = 0
            episode_length = len(actions)
            self.episode_file_indices.append(file_idx)
            self.episode_start_indices.append(episode_start)
            self.episode_lengths.append(episode_length)
            
            # Add episode to replay buffer (WITHOUT images)
            episode = {
                "qpos": qpos_array,
                "text": text_array,
                "action": actions,
            }
            replay_buffer.add_episode(episode)
        
        # Create train/val split
        val_mask = get_val_mask(
            n_episodes=replay_buffer.n_episodes,
            val_ratio=val_ratio,
            seed=seed,
        )
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask,
            max_n=max_train_episodes,
            seed=seed,
        )
        
        # Create sequence sampler (no image key since we're lazy loading)
        self.sampler = SequenceSampler(
            replay_buffer=replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask,
            key_first_k={"qpos": 2, "text": 2},  # Only load first 2 timesteps for obs (no image key)
        )
        
        self.replay_buffer = replay_buffer
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.image_size = image_size
        
        # Convert to numpy arrays for efficient indexing
        self.episode_file_indices = np.array(self.episode_file_indices, dtype=np.int32)
        self.episode_start_indices = np.array(self.episode_start_indices, dtype=np.int32)
        self.episode_lengths = np.array(self.episode_lengths, dtype=np.int32)
        
        print(f"Dataset: {len(self)} sequences, {replay_buffer.n_episodes} episodes (lazy image loading enabled)")
    
    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=~self.train_mask,
        )
        val_set.train_mask = ~self.train_mask
        return val_set
    
    def _get_episode_info(self, buffer_idx: int) -> tuple:
        """
        Get episode index, file index, and relative timestep for a given buffer index.
        
        Returns:
            (episode_idx, file_idx, relative_timestep)
        """
        # Find which episode this buffer index belongs to
        # Use episode_ends from replay buffer for accurate lookup
        episode_ends = np.array(self.replay_buffer.episode_ends[:], dtype=np.int64)
        
        # Find the episode that contains this buffer_idx
        # episode_ends[i] is the end index (exclusive) of episode i
        episode_idx = int(np.searchsorted(episode_ends, buffer_idx + 1))
        if episode_idx >= len(self.episode_file_indices):
            episode_idx = len(self.episode_file_indices) - 1
        
        # Get file index and relative timestep within episode
        file_idx = int(self.episode_file_indices[episode_idx])
        episode_start = int(self.episode_start_indices[episode_idx])
        relative_timestep = buffer_idx - episode_start
        
        return episode_idx, file_idx, relative_timestep
    
    def _load_images_from_file(self, file_idx: int, timestep_indices: list) -> np.ndarray:
        """
        Load images from a specific .npz file for given timestep indices.
        
        Args:
            file_idx: Index of the .npz file
            timestep_indices: List of relative timestep indices within the episode
        
        Returns:
            images: (T, 3, H, W) array of normalized images
        """
        npz_path = self.npz_files[file_idx]
        data = np.load(npz_path, allow_pickle=True)
        obs_list = data["obs"]
        
        images = []
        for t_idx in timestep_indices:
            if 0 <= t_idx < len(obs_list):
                # Image: (H, W, 3) -> (3, H, W) and normalize to [0, 1]
                image = obs_list[t_idx]["image"].astype(np.float32)  # (H, W, 3)
                image = np.moveaxis(image, -1, 0)  # (3, H, W)
                image = image / 255.0  # Normalize to [0, 1]
                images.append(image)
            else:
                # Out of bounds - use padding (first or last image)
                if t_idx < 0:
                    image = obs_list[0]["image"].astype(np.float32)
                else:
                    image = obs_list[-1]["image"].astype(np.float32)
                image = np.moveaxis(image, -1, 0)
                image = image / 255.0
                images.append(image)
        
        return np.array(images, dtype=np.float32)  # (T, 3, H, W)
    
    def get_normalizer(self, mode="limits", **kwargs):
        """Create normalizer from replay buffer statistics."""
        # Extract all data
        qpos_all = self.replay_buffer["qpos"]  # (N, 7) - arm joints only
        action_all = self.replay_buffer["action"]  # (N, 8)
        
        # Normalizer.fit expects flat keys (not nested under "obs")
        # The data structure is: obs = {"image": ..., "qpos": ...}
        # So normalizer should have: normalizer["image"], normalizer["qpos"], normalizer["action"]
        data = {
            "qpos": qpos_all,  # Flat key for fitting
            "action": action_all,
        }
        
        # Set range_eps to prevent blowing up scale factors for joints with very small variation
        # Default is 1e-4, but joint 7 has range ~0.00032 which causes scale factor of 6250
        # Using 5e-2 (same as kitchen dataset) to ignore nearly constant joints
        if "range_eps" not in kwargs:
            kwargs["range_eps"] = 5e-2
        
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        
        # Image normalizer (just identity, images already in [0, 1])
        # Add it as a flat key to match the data structure
        normalizer["image"] = get_image_range_normalizer()
        
        return normalizer
    
    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer["action"])
    
    def __len__(self):
        return len(self.sampler)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Get sequence from sampler (qpos, text, action - no images)
        sample = self.sampler.sample_sequence(idx)
        
        # Get buffer indices for this sequence
        buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx = self.sampler.indices[idx]
        
        # Load images lazily from .npz files
        # Get episode info for each timestep in the actual data range (not padding)
        timestep_indices = []
        file_indices = []
        
        for buffer_idx in range(buffer_start_idx, buffer_end_idx):
            episode_idx, file_idx, relative_timestep = self._get_episode_info(buffer_idx)
            timestep_indices.append(relative_timestep)
            file_indices.append(file_idx)
        
        # Group consecutive timesteps from the same file for efficient loading
        images = []
        current_file_idx = None
        current_timesteps = []
        
        for i, (file_idx, t_idx) in enumerate(zip(file_indices, timestep_indices)):
            if file_idx != current_file_idx:
                # Load previous file's images
                if current_file_idx is not None and len(current_timesteps) > 0:
                    file_images = self._load_images_from_file(current_file_idx, current_timesteps)
                    images.extend(file_images)
                # Start new file
                current_file_idx = file_idx
                current_timesteps = [t_idx]
            else:
                current_timesteps.append(t_idx)
        
        # Load last file's images
        if current_file_idx is not None and len(current_timesteps) > 0:
            file_images = self._load_images_from_file(current_file_idx, current_timesteps)
            images.extend(file_images)
        
        # Convert to numpy array
        if len(images) > 0:
            images = np.array(images, dtype=np.float32)  # (T_actual, 3, H, W)
        else:
            # Edge case: no images loaded (shouldn't happen, but handle it)
            images = np.zeros((0, 3, self.image_size[0], self.image_size[1]), dtype=np.float32)
        
        # Handle padding (same logic as SequenceSampler)
        # sample_start_idx and sample_end_idx indicate where the actual data goes in the padded sequence
        if sample_start_idx > 0 or sample_end_idx < self.horizon:
            padded_images = np.zeros(
                shape=(self.horizon, 3, self.image_size[0], self.image_size[1]),
                dtype=np.float32
            )
            if len(images) > 0:
                if sample_start_idx > 0:
                    padded_images[:sample_start_idx] = images[0]  # Repeat first image
                if sample_end_idx < self.horizon:
                    padded_images[sample_end_idx:] = images[-1]  # Repeat last image
                padded_images[sample_start_idx:sample_end_idx] = images
            images = padded_images
        else:
            # No padding needed, but ensure correct shape
            if len(images.shape) == 0 or images.shape[0] != self.horizon:
                # Reshape if needed
                if len(images) == 0:
                    images = np.zeros((self.horizon, 3, self.image_size[0], self.image_size[1]), dtype=np.float32)
                elif images.shape[0] < self.horizon:
                    # Pad if too short
                    padded = np.zeros((self.horizon, 3, self.image_size[0], self.image_size[1]), dtype=np.float32)
                    padded[:len(images)] = images
                    padded[len(images):] = images[-1]
                    images = padded
        
        # Convert text array to list of strings (for CLIP encoder)
        # Text is stored at both obs["text"] (for image-based models) and top-level "text" (for compatibility)
        text_array = sample["text"]  # (T,) numpy array of strings
        text_list = [str(text_array[i]) for i in range(len(text_array))]
        
        torch_data = {
            "obs": {
                "image": torch.from_numpy(images),  # (T, 3, H, W) - loaded lazily
                "qpos": torch.from_numpy(sample["qpos"]),  # (T, 7) - arm joints only
                "text": text_list,  # (T,) list of strings (for image-based models)
            },
            "action": torch.from_numpy(sample["action"]),  # (T, 8)
            "text": text_list,  # (T,) list of strings (top-level for compatibility with training code)
        }
        return torch_data


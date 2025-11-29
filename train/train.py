"""
Training script for Franka arm manipulation using diffusion transformer with IMAGES and CLIP/SigLIP text embeddings.

This script trains a diffusion policy model for a Franka robot arm to pick up objects based on text commands.
It uses:
- CLIP or SigLIP text encoder for semantic text embeddings (frozen, with learnable projection)
- Vision encoder (ResNet) for image observations
- TransformerForDiffusion for action prediction
- Observations: qpos (arm joints), images (RGB camera), and text (task description)

The model architecture:
- Concatenates qpos with projected text embedding → token 1
- Encodes images via ResNet → token 2
- Each timestep has 2 tokens: [qpos+text, image]
- Transformer processes these tokens to predict actions

Data format:
- Loads from data/pick_place_episode_*.npz files collected by collect_trajectories_image.py
- Each episode contains: obs (image, qpos, text), actions
- Text labels are aligned with trajectories using custom_collate_fn

This code is designed for open-source release on GitHub.
"""

from dataclasses import dataclass
from typing import Optional, List
import sys
from pathlib import Path

# Add repo root to path for imports
script_dir = Path(__file__).parent.absolute()
repo_root = script_dir.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from train.dataset_npz import FrankaNpzImageDataset
from diffusion_policy.model.diffusion.transformer_for_diffusion import TransformerForDiffusion
from diffusion_policy.common.pytorch_util import dict_apply
from train.utils import custom_collate_fn, make_scheduler, create_obs_encoder


@dataclass
class FrankaConfig:
    """Configuration for Franka arm training with images and CLIP text embeddings."""
    # Data
    data_dir: str = "data"  # Relative to repo root (where collect_data.py saves data)
    horizon: int = 16  # prediction horizon
    n_obs_steps: int = 2  # observation history
    n_action_steps: int = 8  # action chunk size
    
    # Text prompt (fallback if not in observations)
    task_text: str = "red block"  # Fallback task description if text not in obs
    
    # Model - Text Encoder (CLIP or SigLIP)
    text_encoder_type: str = "clip"  # "clip" (default) or "siglip"
    clip_model_name: str = "ViT-B-32"  # CLIP model name (e.g., "ViT-B-32", "ViT-B-16")
    clip_pretrained: str = "openai"  # CLIP pretrained weights (e.g., "openai", "laion2b_s34b_b79k")
    # For SigLIP: use model_name like "ViT-B-16-SigLIP" and pretrained like "webli"
    clip_text_freeze: bool = True  # Always freeze text encoder (not fine-tuning)
    
    # Model - Vision Encoder (ResNet)
    resnet_name: str = "resnet18"
    resnet_weights: Optional[str] = "IMAGENET1K_V1"  # or None for random init
    resize_shape: Optional[tuple] = (128, 128)  # Resize images to this (reduced for speed)
    crop_shape: Optional[tuple] = None  # Crop to this (if None, no crop)
    random_crop: bool = False
    use_group_norm: bool = False
    share_rgb_model: bool = True
    imagenet_norm: bool = True
    
    # Model - Transformer
    n_layer: int = 6
    n_head: int = 8
    n_emb: int = 512
    n_cond_layers: int = 2
    p_drop_emb: float = 0.1
    p_drop_attn: float = 0.1
    causal_attn: bool = False
    
    # Training
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"
    batch_size: int = 32
    num_epochs: int = 80
    lr: float = 1e-4
    weight_decay: float = 1e-6
    
    # Diffusion
    num_train_timesteps: int = 100
    num_inference_steps: Optional[int] = None
    
    # Data split
    val_ratio: float = 0.1
    max_train_episodes: Optional[int] = None
    
    # Logging
    print_every: int = 100
    save_every: int = 10
    
    # Output directory
    output_dir: str = "outputs/franka_arm_image_training"


def train_loop(cfg: FrankaConfig) -> None:
    """Main training loop for Franka arm with images and CLIP text embeddings."""
    device = torch.device(cfg.device)
    print(f"Using device: {device}")
    
    # Resolve data_dir relative to repo root (where collect_data.py saves data)
    script_dir = Path(__file__).parent.absolute()
    repo_root = script_dir.parent
    data_dir = (repo_root / cfg.data_dir).resolve()
    print(f"Resolved data_dir: {data_dir}")
    
    # Create output directory structure (relative to repo root)
    output_dir = (repo_root / cfg.output_dir).resolve()
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    print(f"Checkpoints will be saved to: {checkpoint_dir}")
    
    # 1) Load dataset
    print(f"\nLoading dataset from {data_dir}...")
    dataset = FrankaNpzImageDataset(
        data_dir=str(data_dir),
        horizon=cfg.horizon,
        pad_before=0,
        pad_after=0,
        seed=42,
        val_ratio=cfg.val_ratio,
        max_train_episodes=cfg.max_train_episodes,
    )
    
    val_dataset = dataset.get_validation_dataset()
    
    train_loader = DataLoader(
        dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=2,
        collate_fn=custom_collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=2,
        collate_fn=custom_collate_fn
    )
    
    # Get normalizer
    normalizer = dataset.get_normalizer()
    
    # Get data dimensions
    sample = dataset[0]
    image_shape = sample["obs"]["image"].shape[1:]  # (3, H, W)
    qpos_dim = sample["obs"]["qpos"].shape[-1]  # 7 (arm joints only)
    action_dim = sample["action"].shape[-1]  # 8
    
    # Define shape_meta for image encoder
    shape_meta = {
        "obs": {
            "image": {
                "shape": image_shape,
                "type": "rgb",
            },
            "qpos": {
                "shape": (qpos_dim,),
                "type": "low_dim",
            },
        },
        "action": {
            "shape": (action_dim,),
        },
    }
    
    # Check if text is in observations
    has_text = "text" in sample
    if has_text:
        print(f"Text found in observations!")
        text_sample = sample["text"]
        if isinstance(text_sample, torch.Tensor):
            print(f"  Text type: tensor, shape: {text_sample.shape}")
        elif isinstance(text_sample, list):
            print(f"  Text type: list, example: {text_sample[0] if len(text_sample) > 0 else 'empty'}")
        else:
            print(f"  Text type: {type(text_sample)}")
    else:
        print(f"Text NOT in observations - will use fallback: '{cfg.task_text}'")
    
    print(f"Image shape: {image_shape}, Qpos dim: {qpos_dim}, Action dim: {action_dim}")
    
    # 2) Create observation encoder (CLIP/SigLIP text embeddings + ResNet images)
    encoder_type_str = cfg.text_encoder_type.upper() if cfg.text_encoder_type.lower() == "siglip" else "CLIP"
    print(f"\nCreating observation encoder ({encoder_type_str} text embeddings + ResNet images)...")
    obs_encoder, clip_text_encoder, image_encoder = create_obs_encoder(
        clip_model_name=cfg.clip_model_name,
        clip_pretrained=cfg.clip_pretrained,
        clip_text_freeze=cfg.clip_text_freeze,
        resnet_name=cfg.resnet_name,
        resnet_weights=cfg.resnet_weights,
        resize_shape=cfg.resize_shape,
        crop_shape=cfg.crop_shape,
        random_crop=cfg.random_crop,
        use_group_norm=cfg.use_group_norm,
        share_rgb_model=cfg.share_rgb_model,
        imagenet_norm=cfg.imagenet_norm,
        shape_meta=shape_meta,
        device=str(device),
        n_emb=cfg.n_emb,
        text_encoder_type=cfg.text_encoder_type,  # "clip" or "siglip"
    )
    
    # Get obs feature dimension (now returns separate tokens)
    with torch.no_grad():
        # Create example observation
        example_obs = {
            "qpos": torch.zeros(1, qpos_dim, device=device),
            "image": torch.zeros(1, *image_shape, device=device),
            "text": ["zebra"],  # Example text label (list format for CLIP)
        }
        
        obs_tokens = obs_encoder(example_obs)  # (1, 2, n_emb) - [qpos+clip_text, image]
        # For transformer, we need to pass tokens already in n_emb space
        # So cond_dim should be n_emb (tokens are pre-projected)
        obs_feature_dim = cfg.n_emb  # Each token is n_emb dimensional
    print(f"Observation tokens: {obs_tokens.shape} (2 tokens per timestep: [qpos+clip_text, image])")
    print(f"Each token dimension: {obs_feature_dim} (n_emb)")
    
    # 3) Create diffusion transformer model
    # For separate tokens, we pass pre-projected tokens (n_emb dim)
    print(f"\nCreating TransformerForDiffusion model...")
    model = TransformerForDiffusion(
        input_dim=action_dim,
        output_dim=action_dim,
        horizon=cfg.horizon,
        n_obs_steps=cfg.n_obs_steps * 2,  # 2 tokens per timestep: qpos+clip_text, image
        cond_dim=obs_feature_dim,  # Each token is n_emb dimensional (pre-projected)
        n_layer=cfg.n_layer,
        n_head=cfg.n_head,
        n_emb=cfg.n_emb,
        p_drop_emb=cfg.p_drop_emb,
        p_drop_attn=cfg.p_drop_attn,
        causal_attn=cfg.causal_attn,
        time_as_cond=True,
        obs_as_cond=True,
        n_cond_layers=cfg.n_cond_layers,
    ).to(device)
    
    num_params_model = sum(p.numel() for p in model.parameters())
    num_params_encoder = sum(p.numel() for p in obs_encoder.parameters())
    print(f"Transformer params: {num_params_model:,}")
    print(f"Observation encoder params: {num_params_encoder:,}")
    print(f"Total params: {num_params_model + num_params_encoder:,}")
    
    # 4) Optimizer
    print(f"\nSetting up optimizer...")
    print(f"  Base LR: {cfg.lr}")
    
    # Get parameter groups
    optim_groups = model.get_optim_groups(weight_decay=cfg.weight_decay)
    
    # Observation encoder params (learnable projection layers; CLIP text encoder is frozen)
    # CLIP text encoder should be frozen (not fine-tuned)
    if clip_text_encoder is not None:
        clip_text_params = list(clip_text_encoder.get_clip_parameters())
        if len(clip_text_params) > 0:
            for p in clip_text_params:
                p.requires_grad = False
            print(f"  CLIP text encoder: FROZEN ({sum(p.numel() for p in clip_text_params):,} params)")
    
    # Only include trainable encoder parameters (learnable projection layers and image encoder)
    encoder_params = [p for p in obs_encoder.parameters() if p.requires_grad]
    # Also include image encoder parameters (ResNet)
    image_encoder_params = [p for p in image_encoder.parameters() if p.requires_grad]
    
    optim_groups.append({
        "params": encoder_params + image_encoder_params,
        "weight_decay": cfg.weight_decay,
    })
    print(f"  Observation encoder params (trainable): {sum(p.numel() for p in encoder_params):,} params")
    print(f"  Image encoder params (trainable): {sum(p.numel() for p in image_encoder_params):,} params")
    
    optimizer = torch.optim.AdamW(
        optim_groups,
        lr=cfg.lr,
        betas=(0.9, 0.95),
    )
    
    # 5) Diffusion scheduler
    scheduler = make_scheduler(cfg.num_train_timesteps)
    if cfg.num_inference_steps is None:
        num_inference_steps = scheduler.config.num_train_timesteps
    else:
        num_inference_steps = cfg.num_inference_steps
    
    # 6) Check for existing checkpoints and resume if available
    start_epoch = 0
    global_step = 0
    
    # Find latest checkpoint
    checkpoint_epochs = []
    for checkpoint_path in checkpoint_dir.iterdir():
        if checkpoint_path.is_dir():
            if checkpoint_path.name.startswith("epoch_"):
                try:
                    epoch_num = int(checkpoint_path.name.split("_")[1])
                    training_info_path = checkpoint_path / "training_info.pth"
                    if training_info_path.exists():
                        checkpoint_epochs.append(epoch_num)
                except (ValueError, IndexError):
                    pass
    
    if checkpoint_epochs:
        latest_epoch = max(checkpoint_epochs)
        latest_checkpoint_dir = checkpoint_dir / f"epoch_{latest_epoch:04d}"
        training_info_path = latest_checkpoint_dir / "training_info.pth"
        
        if training_info_path.exists():
            print(f"\nFound existing checkpoint at epoch {latest_epoch}")
            print(f"Loading from: {latest_checkpoint_dir}")
            
            # Load training info
            training_info = torch.load(training_info_path, map_location=device, weights_only=False)
            start_epoch = training_info.get("epoch", latest_epoch)
            
            # Load model state
            model_path = latest_checkpoint_dir / "model.pth"
            if model_path.exists():
                model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
                print(f"  Loaded model weights")
            
            # Load obs encoder state
            obs_encoder_path = latest_checkpoint_dir / "obs_encoder.pth"
            if obs_encoder_path.exists():
                obs_encoder.load_state_dict(torch.load(obs_encoder_path, map_location=device, weights_only=False))
                print(f"  Loaded observation encoder weights")
            
            # Load image encoder state (if separate)
            image_encoder_path = latest_checkpoint_dir / "image_encoder.pth"
            if image_encoder_path.exists():
                image_encoder.load_state_dict(torch.load(image_encoder_path, map_location=device, weights_only=False))
                print(f"  Loaded image encoder weights")
            
            # Load optimizer state
            optimizer_path = latest_checkpoint_dir / "optimizer.pth"
            if optimizer_path.exists():
                optimizer.load_state_dict(torch.load(optimizer_path, map_location=device, weights_only=False))
                print(f"  Loaded optimizer state")
            
            # Load normalizer state (if available)
            if "normalizer" in training_info:
                normalizer.load_state_dict(training_info["normalizer"])
                print(f"  Loaded normalizer state")
            
            print(f"  Resuming from epoch {start_epoch}/{cfg.num_epochs}")
            print(f"  Previous avg train loss: {training_info.get('avg_train_loss', 'N/A')}")
            if "avg_val_loss" in training_info and training_info["avg_val_loss"] is not None:
                print(f"  Previous avg val loss: {training_info['avg_val_loss']}")
        else:
            print(f"\nNo valid checkpoint found, starting from scratch")
    else:
        print(f"\nNo existing checkpoints found, starting from scratch")
    
    # 7) Training loop
    print(f"\nStarting training for {cfg.num_epochs} epochs...")
    if start_epoch > 0:
        print(f"Resuming from epoch {start_epoch}")
    
    # Track last epoch losses for final checkpoint
    last_avg_loss = None
    last_avg_val_loss = None
    
    for epoch in range(start_epoch, cfg.num_epochs):
        model.train()
        obs_encoder.train()
        image_encoder.train()
        epoch_losses = []
        
        for batch_idx, batch in enumerate(train_loader):
            # Move observations to device
            obs = dict_apply(batch["obs"], lambda x: x.to(device) if isinstance(x, torch.Tensor) else x)
            action = batch["action"].to(device)  # (B, T, 8)
            B = action.shape[0]
            
            # Get text labels from batch (aligned by custom_collate_fn)
            text_obs = batch.get("text", None)
            
            # Normalize (normalizer may return CPU tensors, so move to device)
            nobs = normalizer.normalize(obs)
            # Ensure normalized obs are on device
            nobs = dict_apply(nobs, lambda x: x.to(device) if isinstance(x, torch.Tensor) else torch.from_numpy(x).to(device) if isinstance(x, np.ndarray) else x)
            nactions = normalizer["action"].normalize(action)
            if not isinstance(nactions, torch.Tensor):
                nactions = torch.from_numpy(nactions) if isinstance(nactions, np.ndarray) else torch.tensor(nactions)
            nactions = nactions.to(device)
            
            # Prepare observations for encoder (first n_obs_steps)
            # Reshape: (B, T, ...) -> (B*T, ...)
            this_nobs = {
                "qpos": nobs["qpos"][:, : cfg.n_obs_steps, :].reshape(-1, qpos_dim),  # (B*n_obs_steps, 7)
                "image": nobs["image"][:, : cfg.n_obs_steps, ...].reshape(-1, *image_shape),  # (B*n_obs_steps, 3, H, W)
            }
            
            # Pass text directly to encoder (encoder will use CLIP internally)
            if text_obs is not None:
                # Prepare text for encoder: need one text per sample (B * n_obs_steps)
                # The encoder expects text in a format it can process
                if isinstance(text_obs, str):
                    # Single text label applies to all samples
                    text_for_encoder = [text_obs] * (B * cfg.n_obs_steps)
                elif isinstance(text_obs, list):
                    # List of text labels - MUST have length B (one per batch item)
                    # The custom_collate_fn ensures this, so if it doesn't match, something is wrong
                    if len(text_obs) != B:
                        raise ValueError(
                            f"CRITICAL ERROR: text_obs length ({len(text_obs)}) != batch size B ({B})! "
                            f"This means text labels are not aligned with trajectories. "
                            f"Check the custom_collate_fn to ensure it preserves text correctly."
                        )
                    
                    # Expand text to match (B * n_obs_steps) samples
                    text_for_encoder = []
                    for b_idx in range(B):
                        # Each text_obs[b_idx] is a list of T strings (all same within episode)
                        # The custom_collate_fn ensures batch[i] corresponds to text_obs[i]
                        text_for_batch = text_obs[b_idx]
                        
                        if text_for_batch is None:
                            # Default to empty string
                            for t_idx in range(cfg.n_obs_steps):
                                text_for_encoder.append("")
                        elif isinstance(text_for_batch, str):
                            # Single text label for entire sequence
                            for t_idx in range(cfg.n_obs_steps):
                                text_for_encoder.append(text_for_batch)
                        elif isinstance(text_for_batch, (list, tuple)):
                            # List or tuple of text labels per timestep (from dataset: all strings are the same per episode)
                            # Use first string since text is constant within an episode
                            if len(text_for_batch) > 0:
                                text_str = str(text_for_batch[0])
                            else:
                                text_str = ""
                            for t_idx in range(cfg.n_obs_steps):
                                text_for_encoder.append(text_str)
                        else:
                            # Fallback: empty string
                            for t_idx in range(cfg.n_obs_steps):
                                text_for_encoder.append("")
                else:
                    # Fallback: empty strings if text format not recognized
                    text_for_encoder = [""] * (B * cfg.n_obs_steps)
                
                this_nobs["text"] = text_for_encoder
            else:
                # Fallback: empty strings if text missing
                this_nobs["text"] = [""] * (B * cfg.n_obs_steps)
            
            # Encode observations - returns separate tokens
            nobs_tokens = obs_encoder(this_nobs)  # (B*n_obs_steps, 2, n_emb) - [qpos+clip_text, image] per sample
            
            # Print L2 and cosine distance between CLIP embeddings after 100 batches
            if batch_idx == 100 and text_obs is not None:
                # Get unique text labels in this batch
                unique_texts = []
                if isinstance(text_obs, list):
                    for text_item in text_obs:
                        if isinstance(text_item, str):
                            if text_item not in unique_texts:
                                unique_texts.append(text_item)
                        elif isinstance(text_item, (list, tuple)) and len(text_item) > 0:
                            text_str = str(text_item[0])
                            if text_str not in unique_texts:
                                unique_texts.append(text_str)
                
                # Get CLIP embeddings for unique texts
                if len(unique_texts) >= 2:
                    with torch.no_grad():
                        clip_embs = obs_encoder.clip_text_encoder.forward_text(unique_texts)  # (num_texts, text_embed_dim)
                        # Apply learnable projection to see how it affects distances
                        projected_clip_embs = obs_encoder.clip_text_proj(clip_embs)  # (num_texts, n_emb)
                        # Compute distances between all pairs
                        for i in range(len(unique_texts)):
                            for j in range(i + 1, len(unique_texts)):
                                # Raw CLIP embedding distances (frozen, constant)
                                raw_l2_dist = torch.norm(clip_embs[i] - clip_embs[j]).item()
                                raw_cosine_dist = 1 - F.cosine_similarity(
                                    clip_embs[i].unsqueeze(0), clip_embs[j].unsqueeze(0)
                                ).item()
                                
                                # Projected embedding distances (changes as projection learns)
                                proj_l2_dist = torch.norm(projected_clip_embs[i] - projected_clip_embs[j]).item()
                                proj_cosine_dist = 1 - F.cosine_similarity(
                                    projected_clip_embs[i].unsqueeze(0), projected_clip_embs[j].unsqueeze(0)
                                ).item()
                                
                                print(f"\n[Batch {batch_idx}] '{unique_texts[i]}' vs '{unique_texts[j]}':")
                                print(f"  Raw CLIP: L2={raw_l2_dist:.4f}, Cosine={raw_cosine_dist:.4f}")
                                print(f"  Projected: L2={proj_l2_dist:.4f}, Cosine={proj_cosine_dist:.4f}")
            
            # Reshape: (B*n_obs_steps, 2, n_emb) -> (B, n_obs_steps*2, n_emb)
            # Flatten the 2 tokens per timestep into sequence
            B_total = nobs_tokens.shape[0]  # B * n_obs_steps
            cond = nobs_tokens.reshape(B, cfg.n_obs_steps * 2, cfg.n_emb)  # (B, n_obs_steps*2, n_emb)
            
            # Forward diffusion: add noise to actions
            noise = torch.randn_like(nactions)
            timesteps = torch.randint(
                0, scheduler.config.num_train_timesteps, (B,), device=device, dtype=torch.long
            )
            noisy_actions = scheduler.add_noise(nactions, noise, timesteps)
            
            # Predict noise
            pred_noise = model(noisy_actions, timesteps, cond=cond)
            
            # Loss
            loss = F.mse_loss(pred_noise, noise)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            
            optimizer.step()
            
            epoch_losses.append(loss.item())
            global_step += 1
            
            if batch_idx % cfg.print_every == 0:
                print(
                    f"Epoch {epoch+1}/{cfg.num_epochs}, "
                    f"Batch {batch_idx}/{len(train_loader)}, "
                    f"Loss: {loss.item():.6f}"
                )
        
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        last_avg_loss = avg_loss  # Track for final checkpoint
        print(f"\nEpoch {epoch+1}/{cfg.num_epochs} - Avg Loss: {avg_loss:.6f}")
        
        # Validation
        avg_val_loss = None
        if len(val_dataset) > 0:
            model.eval()
            obs_encoder.eval()
            image_encoder.eval()
            val_losses = []
            with torch.no_grad():
                for batch in val_loader:
                    # Move observations to device
                    obs = dict_apply(batch["obs"], lambda x: x.to(device) if isinstance(x, torch.Tensor) else x)
                    action = batch["action"].to(device)
                    B = action.shape[0]
                    
                    # Get text labels
                    text_obs = batch.get("text", None)
                    
                    # Normalize
                    nobs = normalizer.normalize(obs)
                    nobs = dict_apply(nobs, lambda x: x.to(device) if isinstance(x, torch.Tensor) else torch.from_numpy(x).to(device) if isinstance(x, np.ndarray) else x)
                    nactions = normalizer["action"].normalize(action)
                    if not isinstance(nactions, torch.Tensor):
                        nactions = torch.from_numpy(nactions) if isinstance(nactions, np.ndarray) else torch.tensor(nactions)
                    nactions = nactions.to(device)
                    
                    # Prepare observations for encoder
                    this_nobs = {
                        "qpos": nobs["qpos"][:, : cfg.n_obs_steps, :].reshape(-1, qpos_dim),
                        "image": nobs["image"][:, : cfg.n_obs_steps, ...].reshape(-1, *image_shape),
                    }
                    
                    # Pass text directly to encoder
                    if text_obs is not None:
                        # Prepare text for encoder: need one text per sample (B * n_obs_steps)
                        if isinstance(text_obs, str):
                            text_for_encoder = [text_obs] * (B * cfg.n_obs_steps)
                        elif isinstance(text_obs, list):
                            if len(text_obs) != B:
                                raise ValueError(
                                    f"CRITICAL ERROR: text_obs length ({len(text_obs)}) != batch size B ({B})! "
                                    f"This means text labels are not aligned with trajectories. "
                                    f"Check the custom_collate_fn to ensure it preserves text correctly."
                                )
                            
                            # Expand text to match (B * n_obs_steps) samples
                            text_for_encoder = []
                            for b_idx in range(B):
                                text_for_batch = text_obs[b_idx]
                                
                                if text_for_batch is None:
                                    for t_idx in range(cfg.n_obs_steps):
                                        text_for_encoder.append("")
                                elif isinstance(text_for_batch, str):
                                    for t_idx in range(cfg.n_obs_steps):
                                        text_for_encoder.append(text_for_batch)
                                elif isinstance(text_for_batch, (list, tuple)):
                                    if len(text_for_batch) > 0:
                                        text_str = str(text_for_batch[0])
                                    else:
                                        text_str = ""
                                    for t_idx in range(cfg.n_obs_steps):
                                        text_for_encoder.append(text_str)
                                else:
                                    for t_idx in range(cfg.n_obs_steps):
                                        text_for_encoder.append("")
                        else:
                            text_for_encoder = [""] * (B * cfg.n_obs_steps)
                        
                        this_nobs["text"] = text_for_encoder
                    else:
                        this_nobs["text"] = [""] * (B * cfg.n_obs_steps)
                    
                    # Encode observations - returns separate tokens
                    nobs_tokens = obs_encoder(this_nobs)  # (B*n_obs_steps, 2, n_emb)
                    # Reshape: (B*n_obs_steps, 2, n_emb) -> (B, n_obs_steps*2, n_emb)
                    cond = nobs_tokens.reshape(B, cfg.n_obs_steps * 2, cfg.n_emb)
                    
                    noise = torch.randn_like(nactions)
                    timesteps = torch.randint(
                        0, scheduler.config.num_train_timesteps, (B,), device=device, dtype=torch.long
                    )
                    noisy_actions = scheduler.add_noise(nactions, noise, timesteps)
                    pred_noise = model(noisy_actions, timesteps, cond=cond)
                    val_loss = F.mse_loss(pred_noise, noise)
                    val_losses.append(val_loss.item())
            
            avg_val_loss = sum(val_losses) / len(val_losses)
            last_avg_val_loss = avg_val_loss  # Track for final checkpoint
            print(f"Validation Loss: {avg_val_loss:.6f}")
        
        # Save checkpoint
        if (epoch + 1) % cfg.save_every == 0:
            epoch_checkpoint_dir = checkpoint_dir / f"epoch_{epoch+1:04d}"
            epoch_checkpoint_dir.mkdir(exist_ok=True)
            
            # Save model
            torch.save(
                model.state_dict(),
                epoch_checkpoint_dir / "model.pth"
            )
            
            # Save optimizer
            torch.save(
                optimizer.state_dict(),
                epoch_checkpoint_dir / "optimizer.pth"
            )
            
            # Save scheduler (just the config, since DDPMScheduler doesn't have state)
            torch.save(
                {
                    "num_train_timesteps": scheduler.config.num_train_timesteps,
                    "beta_start": scheduler.config.beta_start,
                    "beta_end": scheduler.config.beta_end,
                    "beta_schedule": scheduler.config.beta_schedule,
                    "prediction_type": scheduler.config.prediction_type,
                },
                epoch_checkpoint_dir / "scheduler.pth"
            )
            
            # Save obs encoder
            torch.save(
                obs_encoder.state_dict(),
                epoch_checkpoint_dir / "obs_encoder.pth"
            )
            
            # Save image encoder (if separate)
            torch.save(
                image_encoder.state_dict(),
                epoch_checkpoint_dir / "image_encoder.pth"
            )
            
            # Save training info
            torch.save(
                {
                    "epoch": epoch + 1,
                    "normalizer": normalizer.state_dict(),
                    "config": cfg,
                    "avg_train_loss": avg_loss,
                    "avg_val_loss": avg_val_loss if len(val_dataset) > 0 else None,
                    "task_text": cfg.task_text,
                    "shape_meta": shape_meta,
                },
                epoch_checkpoint_dir / "training_info.pth"
            )
            
            print(f"Saved checkpoint to {epoch_checkpoint_dir}")
    
    # 8) Final sampling demo
    print("\n=== Sampling demo ===")
    model.eval()
    obs_encoder.eval()
    image_encoder.eval()
    with torch.no_grad():
        batch = next(iter(train_loader))
        obs = dict_apply(batch["obs"], lambda x: x[:4].to(device) if isinstance(x, torch.Tensor) else x[:4])
        action_true = batch["action"][:4].to(device)  # (4, T, 8)
        # Get text labels
        text_obs = batch.get("text", None)
        
        # Normalize
        nobs = normalizer.normalize(obs)
        nobs = dict_apply(nobs, lambda x: x.to(device) if isinstance(x, torch.Tensor) else torch.from_numpy(x).to(device) if isinstance(x, np.ndarray) else x)
        
        # Prepare observations for encoder
        this_nobs = {
            "qpos": nobs["qpos"][:, : cfg.n_obs_steps, :].reshape(-1, qpos_dim),
            "image": nobs["image"][:, : cfg.n_obs_steps, ...].reshape(-1, *image_shape),
        }
        
        # Pass text directly to encoder
        if text_obs is not None:
            # Prepare text for encoder: need one text per sample (4 * n_obs_steps)
            demo_batch_size = 4
            if isinstance(text_obs, str):
                text_for_encoder = [text_obs] * (demo_batch_size * cfg.n_obs_steps)
            elif isinstance(text_obs, list):
                if len(text_obs) < demo_batch_size:
                    # Pad with last element if needed
                    text_obs = text_obs + [text_obs[-1]] * (demo_batch_size - len(text_obs))
                text_obs = text_obs[:demo_batch_size]  # Take first 4
                
                text_for_encoder = []
                for b_idx in range(demo_batch_size):
                    text_for_batch = text_obs[b_idx]
                    
                    if text_for_batch is None:
                        for t_idx in range(cfg.n_obs_steps):
                            text_for_encoder.append("")
                    elif isinstance(text_for_batch, str):
                        for t_idx in range(cfg.n_obs_steps):
                            text_for_encoder.append(text_for_batch)
                    elif isinstance(text_for_batch, (list, tuple)):
                        if len(text_for_batch) > 0:
                            text_str = str(text_for_batch[0])
                        else:
                            text_str = ""
                        for t_idx in range(cfg.n_obs_steps):
                            text_for_encoder.append(text_str)
                    else:
                        for t_idx in range(cfg.n_obs_steps):
                            text_for_encoder.append("")
            else:
                text_for_encoder = [""] * (demo_batch_size * cfg.n_obs_steps)
            
            this_nobs["text"] = text_for_encoder
        else:
            this_nobs["text"] = [""] * (4 * cfg.n_obs_steps)
        
        # Encode observations - returns separate tokens
        nobs_tokens = obs_encoder(this_nobs)  # (4*n_obs_steps, 2, n_emb)
        # Reshape: (4*n_obs_steps, 2, n_emb) -> (4, n_obs_steps*2, n_emb)
        cond = nobs_tokens.reshape(4, cfg.n_obs_steps * 2, cfg.n_emb)
        
        # Sample from diffusion model
        nactions_sampled = torch.randn_like(action_true)  # start from noise
        nactions_sampled = nactions_sampled.to(device)
        scheduler.set_timesteps(num_inference_steps)
        
        for t in scheduler.timesteps:
            noise_pred = model(nactions_sampled, t, cond=cond)
            out = scheduler.step(noise_pred, t, nactions_sampled)
            nactions_sampled = out.prev_sample
        
        # Unnormalize
        actions_sampled = normalizer["action"].unnormalize(nactions_sampled)
        
        # Get action chunk (first n_action_steps)
        action_chunk = actions_sampled[:, : cfg.n_action_steps, :]
        
        print(f"True action (first step): {action_true[0, 0].cpu().numpy()}")
        print(f"Sampled action (first step): {actions_sampled[0, 0].cpu().numpy()}")
        print(f"Action chunk (first {cfg.n_action_steps} steps) shape: {action_chunk.shape}")
    
    # 9) Save final model checkpoint (only if training completed all epochs)
    if start_epoch < cfg.num_epochs:
        final_checkpoint_dir = checkpoint_dir / "final"
        final_checkpoint_dir.mkdir(exist_ok=True)
        
        # Save model
        torch.save(
            model.state_dict(),
            final_checkpoint_dir / "model.pth"
        )
        
        # Save obs encoder
        torch.save(
            obs_encoder.state_dict(),
            final_checkpoint_dir / "obs_encoder.pth"
        )
        
        # Save image encoder
        torch.save(
            image_encoder.state_dict(),
            final_checkpoint_dir / "image_encoder.pth"
        )
        
        # Save optimizer
        torch.save(
            optimizer.state_dict(),
            final_checkpoint_dir / "optimizer.pth"
        )
        
        # Save scheduler config
        torch.save(
            {
                "num_train_timesteps": scheduler.config.num_train_timesteps,
                "beta_start": scheduler.config.beta_start,
                "beta_end": scheduler.config.beta_end,
                "beta_schedule": scheduler.config.beta_schedule,
                "prediction_type": scheduler.config.prediction_type,
            },
            final_checkpoint_dir / "scheduler.pth"
        )
        
        # Get final epoch loss for saving (from last epoch)
        final_avg_loss = last_avg_loss
        final_val_loss = last_avg_val_loss if len(val_dataset) > 0 else None
        
        # Save training info
        torch.save(
            {
                "epoch": cfg.num_epochs,
                "normalizer": normalizer.state_dict(),
                "config": cfg,
                "task_text": cfg.task_text,
                "avg_train_loss": final_avg_loss,
                "avg_val_loss": final_val_loss,
                "shape_meta": shape_meta,
            },
            final_checkpoint_dir / "training_info.pth"
        )
        
        print(f"\nFinal model saved to: {final_checkpoint_dir}")
    
    print("\nTraining complete!")


if __name__ == "__main__":
    cfg = FrankaConfig()
    train_loop(cfg)

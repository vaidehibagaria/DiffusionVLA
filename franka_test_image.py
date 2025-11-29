#!/usr/bin/env python3
"""
Test script for Franka arm using trained diffusion policy with IMAGE observations and CLIP/SigLIP text embeddings.

This script loads a model trained by train/train.py and runs closed-loop control:
- Uses image observations: RGB camera (320x240) + qpos (7 arm joints) + text (CLIP/SigLIP embeddings)
- Vision encoder (ResNet) processes images
- CLIP or SigLIP text encoder processes text commands
- Starts from home position
- Uses diffusion denoising to predict action sequences
- Executes action chunks (receding horizon control)
- Replans every n_action_steps

Architecture:
- Token 1: qpos + projected text embedding (concatenated, then projected)
- Token 2: image embedding (via ResNet, then projected)
- Each timestep has 2 tokens: [qpos+text, image]

This code is designed for open-source release on GitHub.
"""

from __future__ import annotations

import argparse
import sys
import os
import types
from pathlib import Path

# Add parent directory to path so we can import diffusion_policy
script_dir = Path(__file__).parent.absolute()
repo_root = script_dir.parent
sys.path.insert(0, str(repo_root))

import numpy as np
import torch
import time
import mujoco
import mujoco.viewer as viewer
from typing import Optional

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

# Import from dataset folder (avoiding setuptools conflict)
import importlib.util
franka_env_path = repo_root / "dataset" / "franka_env.py"
spec = importlib.util.spec_from_file_location("franka_env_module", franka_env_path)
franka_env_module = importlib.util.module_from_spec(spec)
sys.modules["franka_env_module"] = franka_env_module
spec.loader.exec_module(franka_env_module)
FrankaTaskEnv = franka_env_module.FrankaTaskEnv
StepResult = franka_env_module.StepResult

from diffusion_policy.model.diffusion.transformer_for_diffusion import TransformerForDiffusion
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.common.pytorch_util import dict_apply

# Import from refactored training modules
from train.models.image_text_obs_encoder import ImageTextObsEncoder
from train.utils import make_scheduler, create_obs_encoder
from train.train import FrankaConfig


def load_model_and_normalizer(checkpoint_dir: str | Path, device: str = "cuda:0"):
    """
    Load trained model, normalizer, and observation encoder from checkpoint.
    
    The checkpoint should be from train/train.py which uses:
    - ImageTextObsEncoder (images + CLIP text embeddings)
    - ResNet for image encoding
    - CLIP for text encoding
    """
    checkpoint_dir = Path(checkpoint_dir)
    
    # Load training info (contains config and normalizer)
    training_info = torch.load(checkpoint_dir / "training_info.pth", map_location=device, weights_only=False)
    cfg = training_info["config"]
    
    # Recreate normalizer
    normalizer = LinearNormalizer()
    normalizer.load_state_dict(training_info["normalizer"])
    
    # Infer action dimension from checkpoint
    model_state = torch.load(checkpoint_dir / "model.pth", map_location=device, weights_only=True)
    if "head.weight" in model_state:
        action_dim = model_state["head.weight"].shape[0]
    elif "input_emb.weight" in model_state:
        action_dim = model_state["input_emb.weight"].shape[1]
    else:
        # Fallback: try to get from normalizer
        if "action" in normalizer.params_dict:
            action_stats = normalizer.params_dict["action"].get("input_stats", {})
            if "shape" in action_stats:
                action_dim = action_stats["shape"][-1] if isinstance(action_stats["shape"], (list, tuple)) else action_stats["shape"]
            else:
                scale = normalizer.params_dict["action"].get("scale")
                if scale is not None:
                    scale_np = scale.cpu().numpy() if isinstance(scale, torch.Tensor) else scale
                    action_dim = scale_np.shape[-1] if len(scale_np.shape) > 0 else 1
                else:
                    action_dim = 8  # Default: 7 arm joints + 1 gripper
        else:
            action_dim = 8
    
    print(f"Inferred action_dim from checkpoint: {action_dim}")
    
    # Observation dimensions (from training)
    qpos_dim = 7  # Arm joints only
    image_shape = (3, 240, 320)  # (C, H, W) - matches training data format
    
    # Define shape_meta for image encoder (must match training)
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
    
    # Create observation encoder (CLIP/SigLIP text embeddings + ResNet images)
    text_encoder_type = getattr(cfg, 'text_encoder_type', 'clip')  # Default to CLIP for backward compatibility
    encoder_type_str = text_encoder_type.upper() if text_encoder_type.lower() == "siglip" else "CLIP"
    print(f"Creating observation encoder ({encoder_type_str} text embeddings + ResNet images)...")
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
        text_encoder_type=text_encoder_type,  # "clip" or "siglip"
    )
    
    tokens_per_timestep = 2  # [qpos+clip_text, image]
    token_description = "2 tokens: [qpos+clip_text, image]"
    
    # Load observation encoder weights if available
    obs_encoder_path = checkpoint_dir / "obs_encoder.pth"
    if obs_encoder_path.exists():
        obs_encoder.load_state_dict(torch.load(obs_encoder_path, map_location=device, weights_only=False))
        print(f"Loaded observation encoder from checkpoint")
    else:
        print(f"Warning: obs_encoder.pth not found, using initialized encoder")
    
    # Load image encoder weights if separate
    image_encoder_path = checkpoint_dir / "image_encoder.pth"
    if image_encoder_path.exists():
        image_encoder.load_state_dict(torch.load(image_encoder_path, map_location=device, weights_only=False))
        print(f"Loaded image encoder from checkpoint")
    
    obs_encoder.eval()
    image_encoder.eval()
    
    # Get observation feature dimension
    with torch.no_grad():
        example_obs = {
            "qpos": torch.zeros(1, qpos_dim, device=device),
            "image": torch.zeros(1, *image_shape, device=device),
            "text": ["zebra"],  # List format for CLIP encoder
        }
        obs_tokens = obs_encoder(example_obs)
        obs_feature_dim = cfg.n_emb  # Each token is n_emb dimensional
    
    print(f"Observation tokens: {obs_tokens.shape} ({token_description})")
    print(f"Each token dimension: {obs_feature_dim} (n_emb)")
    
    # Recreate model (must match training architecture)
    model = TransformerForDiffusion(
        input_dim=action_dim,
        output_dim=action_dim,
        horizon=cfg.horizon,
        n_obs_steps=cfg.n_obs_steps * tokens_per_timestep,  # 2 tokens per timestep
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
    
    # Load model weights
    model.load_state_dict(torch.load(checkpoint_dir / "model.pth", map_location=device, weights_only=False))
    model.eval()
    
    # Recreate scheduler
    scheduler_config = torch.load(checkpoint_dir / "scheduler.pth", map_location=device, weights_only=False)
    if isinstance(scheduler_config, dict):
        scheduler = DDPMScheduler(
            num_train_timesteps=scheduler_config["num_train_timesteps"],
            beta_start=scheduler_config["beta_start"],
            beta_end=scheduler_config["beta_end"],
            beta_schedule=scheduler_config["beta_schedule"],
            prediction_type=scheduler_config["prediction_type"],
        )
    else:
        scheduler = scheduler_config
    
    num_inference_steps = getattr(cfg, 'num_inference_steps', None)
    if num_inference_steps is None:
        num_inference_steps = getattr(scheduler.config, 'num_train_timesteps', 100)
    
    return model, normalizer, scheduler, cfg, num_inference_steps, action_dim, obs_encoder, qpos_dim, image_shape, tokens_per_timestep


def capture_image(env: FrankaTaskEnv, width: int = 320, height: int = 240) -> np.ndarray:
    """
    Capture RGB image from environment.
    
    Args:
        env: FrankaTaskEnv instance
        width: Image width (default 320, matches training)
        height: Image height (default 240, matches training)
    
    Returns:
        image: (H, W, 3) uint8 RGB image
    """
    rgb = env.render(width=width, height=height, camera_name="overhead")
    return rgb  # (H, W, 3) uint8


def image_to_tensor(image: np.ndarray, device: str) -> torch.Tensor:
    """
    Convert RGB image to normalized tensor format expected by model.
    
    Args:
        image: (H, W, 3) uint8 RGB image
        device: Device to place tensor on
    
    Returns:
        image_tensor: (3, H, W) float32 tensor in [0, 1] range
    """
    # Convert to float and normalize to [0, 1]
    image_float = image.astype(np.float32) / 255.0  # (H, W, 3)
    # Convert to (C, H, W) format
    image_tensor = torch.from_numpy(image_float).permute(2, 0, 1)  # (3, H, W)
    return image_tensor.to(device)


def obs_to_dict(state: dict, image: np.ndarray, device: str) -> dict:
    """
    Convert state dict and image to observation dict matching training format.
    
    Args:
        state: State dict from env.get_full_state() with qpos
        image: (H, W, 3) uint8 RGB image
        device: Device to place tensors on
    
    Returns:
        obs_dict: Dict with keys:
            - "qpos": (7,) tensor - arm joint positions
            - "image": (3, H, W) tensor - normalized RGB image
    """
    qpos = state["qpos"].astype(np.float32)  # (16,)
    qpos_arm = qpos[:7]  # (7,) - only arm joints
    
    # Convert to tensors
    qpos_tensor = torch.from_numpy(qpos_arm).to(device).float()  # (7,)
    image_tensor = image_to_tensor(image, device)  # (3, H, W)
    
    return {
        "qpos": qpos_tensor,
        "image": image_tensor,
    }


def set_home_position(env: FrankaTaskEnv):
    """Set robot to home position (same as training)."""
    home_q = np.array([
        0.0,
        0.0,
        0.0,
        -1.57079,
        0.0,
        1.57079,
        -0.7853,
    ])
    # Position control: desired position = home_q
    env.data.qpos[:7] = home_q
    env.data.qvel[:7] = 0.0
    env.data.ctrl[:7] = home_q
    
    # Gripper open (ctrl range 0–255)
    env.data.ctrl[7] = 255.0
    
    # Recompute derived quantities
    mujoco.mj_forward(env.model, env.data)


def run_episode(
    env: FrankaTaskEnv,
    model: TransformerForDiffusion,
    normalizer: LinearNormalizer,
    scheduler: DDPMScheduler,
    cfg: FrankaConfig,
    num_inference_steps: int,
    device: str,
    action_dim: int,
    obs_encoder: ImageTextObsEncoder,
    qpos_dim: int,
    image_shape: tuple,
    tokens_per_timestep: int = 2,
    text_label: str = "blue block",  # Text instruction for this episode
    max_steps: int = 200,
    mj_viewer: Optional[viewer.Viewer] = None,
    verbose: bool = True,
) -> dict:
    """
    Run a single episode using the diffusion policy with image observations.
    
    Closed-loop control:
    1. Maintain observation history (n_obs_steps) with images and qpos
    2. Predict full action horizon using diffusion
    3. Execute first n_action_steps
    4. Replan and repeat
    
    Args:
        env: FrankaTaskEnv instance
        model: Trained TransformerForDiffusion model
        normalizer: LinearNormalizer for observations and actions
        scheduler: DDPMScheduler for diffusion denoising
        cfg: Training configuration
        num_inference_steps: Number of diffusion denoising steps
        device: Device to run model on
        action_dim: Action dimension (8: 7 joints + 1 gripper)
        obs_encoder: ImageTextObsEncoder for processing observations
        qpos_dim: Qpos dimension (7: arm joints only)
        image_shape: Image shape tuple (C, H, W)
        tokens_per_timestep: Number of tokens per timestep (2: [qpos+clip_text, image])
        text_label: Text command (e.g., "zebra" for red, "blue block" for blue)
        max_steps: Maximum steps per episode
        mj_viewer: Optional MuJoCo viewer for visualization
        verbose: Print progress messages
    
    Returns:
        result: Dict with keys: steps, success, done
    """
    # Reset environment
    env.reset(task="pick_place")
    
    # Set to home position
    set_home_position(env)
    
    # Observation history buffers (separate for images and qpos)
    image_history = []
    qpos_history = []
    
    # Episode statistics
    step_count = 0
    done = False
    success = False
    
    # Latency tracking
    encode_times = []
    diffusion_times = []
    total_inference_times = []
    
    if verbose:
        print(f"Starting episode (max_steps={max_steps})")
        print(f"Horizon={cfg.horizon}, n_obs_steps={cfg.n_obs_steps}, n_action_steps={cfg.n_action_steps}")
        print(f"Text label: '{text_label}' (encoder will use CLIP embeddings)")
    
    while step_count < max_steps:
        # Get current observation (image + qpos)
        state = env.get_full_state()
        image = capture_image(env)  # (H, W, 3) uint8
        
        # Convert to observation dict
        obs_dict = obs_to_dict(state, image, device)
        
        # Add to history
        image_history.append(obs_dict["image"])  # (3, H, W)
        qpos_history.append(obs_dict["qpos"])  # (7,)
        
        # Keep only last n_obs_steps
        if len(image_history) > cfg.n_obs_steps:
            image_history = image_history[-cfg.n_obs_steps:]
            qpos_history = qpos_history[-cfg.n_obs_steps:]
        
        # Need at least n_obs_steps to condition on
        if len(image_history) < cfg.n_obs_steps:
            # Pad with current observation
            while len(image_history) < cfg.n_obs_steps:
                image_history.insert(0, obs_dict["image"])
                qpos_history.insert(0, obs_dict["qpos"])
        
        # Stack into batch format
        qpos_batch = torch.stack(qpos_history, dim=0).unsqueeze(0)  # (1, n_obs_steps, 7)
        image_batch = torch.stack(image_history, dim=0).unsqueeze(0)  # (1, n_obs_steps, 3, H, W)
        
        # Normalize observations (pass tensors directly, matching training script)
        obs = {
            "qpos": qpos_batch,  # Keep as tensor
            "image": image_batch,  # Keep as tensor
        }
        nobs = normalizer.normalize(obs)
        # Ensure normalized obs are on device (matching training script)
        nobs = dict_apply(nobs, lambda x: x.to(device) if isinstance(x, torch.Tensor) else torch.from_numpy(x).to(device) if isinstance(x, np.ndarray) else x)
        
        # Prepare observations for encoder (reshape to batch)
        # Reshape: (1, n_obs_steps, ...) -> (n_obs_steps, ...)
        this_nobs = {
            "qpos": nobs["qpos"].reshape(-1, qpos_dim),  # (n_obs_steps, 7)
            "image": nobs["image"].reshape(-1, *image_shape),  # (n_obs_steps, 3, H, W)
            "text": [text_label] * cfg.n_obs_steps,  # Pass text, encoder uses CLIP embeddings
        }
        
        # Measure inference latency
        inference_start = time.time()
        
        # Encode observations using ImageTextObsEncoder (returns separate tokens)
        with torch.no_grad():
            # Measure encoding time
            encode_start = time.time()
            nobs_tokens = obs_encoder(this_nobs)  # (n_obs_steps, tokens_per_timestep, n_emb)
            encode_time = time.time() - encode_start
            
            # Reshape: (n_obs_steps, tokens_per_timestep, n_emb) -> (1, n_obs_steps*tokens_per_timestep, n_emb)
            cond = nobs_tokens.reshape(1, cfg.n_obs_steps * tokens_per_timestep, cfg.n_emb)
        
        # Sample action sequence using diffusion
        with torch.no_grad():
            # Start from noise
            nactions = torch.randn(1, cfg.horizon, action_dim, device=device)
            scheduler.set_timesteps(num_inference_steps)
            
            # Measure diffusion time
            diffusion_start = time.time()
            # Denoising loop
            for t in scheduler.timesteps:
                t_int = int(t.item()) if isinstance(t, torch.Tensor) else int(t)
                noise_pred = model(nactions, t_int, cond=cond)
                out = scheduler.step(noise_pred, t_int, nactions)
                nactions = out.prev_sample
            diffusion_time = time.time() - diffusion_start
            
            # Unnormalize actions
            actions = normalizer["action"].unnormalize(nactions)
            actions = actions.squeeze(0).cpu().numpy()  # (horizon, action_dim)
        
        # Total inference time
        total_inference_time = time.time() - inference_start
        
        # Track latency statistics
        encode_times.append(encode_time)
        diffusion_times.append(diffusion_time)
        total_inference_times.append(total_inference_time)
        
        # Print latency information (every 10 steps to avoid spam)
        if step_count % 10 == 0 and verbose:
            print(f"[Step {step_count}] Inference Latency:")
            print(f"  Encoding: {encode_time*1000:.2f} ms")
            print(f"  Diffusion ({num_inference_steps} steps): {diffusion_time*1000:.2f} ms ({diffusion_time/num_inference_steps*1000:.2f} ms/step)")
            print(f"  Total: {total_inference_time*1000:.2f} ms")
        
        # Execute action chunk (first n_action_steps)
        action_chunk = actions[:cfg.n_action_steps]  # (n_action_steps, action_dim)
        
        # Execute each action in the chunk
        for action in action_chunk:
            # Get current qpos before applying action
            current_qpos = env.data.qpos[:action_dim].copy()
            
            # Compute action delta
            action_delta = action[:7] - current_qpos[:7]
            action_scale = 1.0  # Default: no scaling
            
            # Apply scaling
            scaled_action = current_qpos[:7] + action_delta * action_scale
            
            # Actions are absolute target positions (next qpos)
            # Pad with gripper control if action_dim=7
            if action_dim == 7:
                ctrl = np.zeros(8)
                ctrl[:7] = scaled_action  # 7 arm joints
                ctrl[7] = 255.0  # Gripper open
            else:
                ctrl = np.zeros(8)
                ctrl[:7] = scaled_action[:7]  # 7 arm joints
                ctrl[7] = action[7] if len(action) > 7 else 255.0  # Gripper from action or default open
            
            # Step environment with action
            step_result: StepResult = env.step(ctrl=ctrl)
            done = False  # Always continue until max_steps
            success = env.success_flag
            
            step_count += 1
            
            if mj_viewer is not None:
                mj_viewer.sync()
            
            if step_count >= max_steps:
                break
        
        if verbose and step_count % 50 == 0:
            print(f"Step {step_count}/{max_steps}, Success: {success}")
    
    if verbose:
        print(f"\nEpisode finished:")
        print(f"  Steps: {step_count}/{max_steps}")
        print(f"  Success: {success}")
        print(f"  Done: {done}")
        
        # Print latency statistics summary
        if len(total_inference_times) > 0:
            avg_encode = np.mean(encode_times) * 1000
            avg_diffusion = np.mean(diffusion_times) * 1000
            avg_total = np.mean(total_inference_times) * 1000
            min_total = np.min(total_inference_times) * 1000
            max_total = np.max(total_inference_times) * 1000
            
            print(f"\nLatency Statistics (over {len(total_inference_times)} inference calls):")
            print(f"  Encoding: {avg_encode:.2f} ms (avg)")
            print(f"  Diffusion: {avg_diffusion:.2f} ms (avg)")
            print(f"  Total Inference: {avg_total:.2f} ms (avg), {min_total:.2f} ms (min), {max_total:.2f} ms (max)")
            print(f"  Throughput: {1000/avg_total:.2f} Hz (inferences per second)")
    
    return {
        "steps": step_count,
        "success": success,
        "done": done,
        "latency_stats": {
            "encode_times": encode_times,
            "diffusion_times": diffusion_times,
            "total_inference_times": total_inference_times,
        } if len(total_inference_times) > 0 else None,
    }


def main():
    """Main entry point for testing Franka arm with image observations."""
    parser = argparse.ArgumentParser(
        description="Test Franka arm with image observations and CLIP text embeddings"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="outputs/franka_arm_image_training/checkpoints/final",
        help="Path to checkpoint directory (from train/train.py)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Device to run model on",
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=10,
        help="Number of test episodes to run",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=200,
        help="Maximum steps per episode",
    )
    parser.add_argument(
        "--view",
        action="store_true",
        help="Open MuJoCo viewer during testing",
    )
    args = parser.parse_args()
    
    # Load model, normalizer, scheduler, and observation encoder
    print(f"Loading model from {args.checkpoint}...")
    model, normalizer, scheduler, cfg, num_inference_steps, action_dim, obs_encoder, qpos_dim, image_shape, tokens_per_timestep = load_model_and_normalizer(
        args.checkpoint, args.device
    )
    print(f"Model loaded. Using {num_inference_steps} inference steps.")
    print(f"Action dimension: {action_dim}")
    print(f"Qpos dim: {qpos_dim}, Image shape: {image_shape}")
    
    # Create environment
    xml_path = script_dir / "franka_emika_panda.xml"
    if not xml_path.exists():
        xml_path = repo_root / "setup" / "franka_emika_panda.xml"
    env = FrankaTaskEnv(xml_path=str(xml_path), max_steps=args.max_steps)
    
    # Run episodes
    results = []
    text_label = "zebra"  # Change this to test different commands: "zebra" (red), "blue block" (blue)

    if args.view:
        with viewer.launch_passive(env.model, env.data) as v:
            for i in range(args.num_episodes):
                print(f"\n{'='*50}")
                print(f"Episode {i+1}/{args.num_episodes}")
                print(f"{'='*50}")
                
                result = run_episode(
                    env, model, normalizer, scheduler, cfg, num_inference_steps,
                    args.device, action_dim, obs_encoder, qpos_dim, image_shape,
                    tokens_per_timestep=tokens_per_timestep,
                    text_label=text_label, max_steps=args.max_steps, mj_viewer=v, verbose=True
                )
                results.append(result)
    else:
        for i in range(args.num_episodes):
            print(f"\n{'='*50}")
            print(f"Episode {i+1}/{args.num_episodes}")
            print(f"{'='*50}")
            
            # Get text label for this episode
            text_label = input(f"Enter command for episode {i+1} (or press Enter for default 'blue block'): ").strip()
            if not text_label:
                text_label = "blue block"
                print(f"Using default: '{text_label}'")
            else:
                print(f"Using command: '{text_label}'")
            
            result = run_episode(
                env, model, normalizer, scheduler, cfg, num_inference_steps,
                args.device, action_dim, obs_encoder, qpos_dim, image_shape,
                tokens_per_timestep=tokens_per_timestep,
                text_label=text_label, max_steps=args.max_steps, mj_viewer=None, verbose=True
            )
            results.append(result)
    
    # Print summary
    print(f"\n{'='*50}")
    print("Summary")
    print(f"{'='*50}")
    successes = sum(r["success"] for r in results)
    avg_steps = np.mean([r["steps"] for r in results])
    print(f"Success rate: {successes}/{args.num_episodes} ({100*successes/args.num_episodes:.1f}%)")
    print(f"Average steps: {avg_steps:.1f}")


if __name__ == "__main__":
    main()

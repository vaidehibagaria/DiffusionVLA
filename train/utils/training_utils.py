"""
Training utility functions for Franka arm with images and CLIP/SigLIP.
"""

from typing import Optional
import torch
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusion_policy.model.vision.model_getter import get_clip_encoder, get_siglip_encoder, get_resnet
from diffusion_policy.model.vision.multi_image_obs_encoder import MultiImageObsEncoder
from diffusion_policy_test.models.image_text_obs_encoder import ImageTextObsEncoder


def make_scheduler(num_train_timesteps: int = 100) -> DDPMScheduler:
    """Create DDPM scheduler for diffusion training."""
    return DDPMScheduler(
        num_train_timesteps=num_train_timesteps,
        beta_start=1e-4,
        beta_end=2e-2,
        beta_schedule="squaredcos_cap_v2",
        prediction_type="epsilon",
    )


def create_obs_encoder(
    clip_model_name: str,
    clip_pretrained: str,
    clip_text_freeze: bool,
    resnet_name: str,
    resnet_weights: Optional[str],
    resize_shape: Optional[tuple],
    crop_shape: Optional[tuple],
    random_crop: bool,
    use_group_norm: bool,
    share_rgb_model: bool,
    imagenet_norm: bool,
    shape_meta: dict,
    device: str,
    n_emb: int = 512,
    text_encoder_type: str = "clip",  # "clip" or "siglip"
):
    """
    Create observation encoder: CLIP/SigLIP text embeddings + image encoder (ResNet).
    
    Args:
        text_encoder_type: "clip" (default) or "siglip" - which text encoder to use
    
    Returns:
        obs_encoder: ImageTextObsEncoder that processes qpos, images, and text
        text_encoder: CLIPEncoder or SigLIPEncoder for text embeddings (frozen)
        image_encoder: MultiImageObsEncoder for image encoding (ResNet)
    """
    # Create text encoder (CLIP or SigLIP)
    if text_encoder_type.lower() == "siglip":
        text_encoder = get_siglip_encoder(
            model_name=clip_model_name,  # Reuse clip_model_name for SigLIP model name
            pretrained=clip_pretrained,  # Reuse clip_pretrained for SigLIP pretrained weights
            device=device,
            freeze=clip_text_freeze,  # Freeze text encoder
        )
        print(f"Using SigLIP text encoder: {clip_model_name}, pretrained: {clip_pretrained}")
    else:
        # Default: CLIP
        text_encoder = get_clip_encoder(
            model_name=clip_model_name,
            pretrained=clip_pretrained,
            device=device,
            freeze=clip_text_freeze,  # Freeze CLIP text encoder
        )
        print(f"Using CLIP text encoder: {clip_model_name}, pretrained: {clip_pretrained}")
    
    # Create ResNet model for image encoding
    rgb_model = get_resnet(
        name=resnet_name,
        weights=resnet_weights,
    )
    
    # Create MultiImageObsEncoder for images
    image_encoder = MultiImageObsEncoder(
        shape_meta=shape_meta,
        rgb_model=rgb_model,
        resize_shape=resize_shape,
        crop_shape=crop_shape,
        random_crop=random_crop,
        use_group_norm=use_group_norm,
        share_rgb_model=share_rgb_model,
        imagenet_norm=imagenet_norm,
    ).to(device)
    
    # Create observation encoder with text encoder (CLIP or SigLIP) and image encoder
    obs_encoder = ImageTextObsEncoder(
        clip_text_encoder=text_encoder,  # Can be CLIP or SigLIP (both have same interface)
        image_encoder=image_encoder,
        qpos_dim=7,  # Arm joints only
        n_emb=n_emb,  # Project to transformer embedding dimension
    ).to(device)
    
    return obs_encoder, text_encoder, image_encoder


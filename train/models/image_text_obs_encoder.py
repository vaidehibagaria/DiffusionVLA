"""
Observation encoder for image-based Franka arm training with CLIP/SigLIP text embeddings.

This module contains the ImageTextObsEncoder class that processes:
- qpos (arm joint positions)
- images (RGB camera observations)
- text (task descriptions via CLIP or SigLIP)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusion_policy.model.vision.model_getter import CLIPEncoder
from diffusion_policy.model.vision.multi_image_obs_encoder import MultiImageObsEncoder


class ImageTextObsEncoder(nn.Module):
    """
    Observation encoder that processes images, qpos, and CLIP/SigLIP text embeddings.
    
    Architecture:
    - Token 1: qpos + projected text embedding (concatenated, then projected)
    - Token 2: image embedding (via ResNet, then projected)
    
    Each modality is projected separately to n_emb dimension.
    Returns: (B, 2, n_emb) where 2 = [qpos+text, image] per timestep
    
    The text encoder (CLIP or SigLIP) is frozen, but a learnable projection layer allows
    the model to adapt text embeddings to the task during training.
    """
    def __init__(
        self,
        clip_text_encoder: CLIPEncoder,
        image_encoder: MultiImageObsEncoder,
        qpos_dim: int = 7,
        n_emb: int = 512,
    ):
        super().__init__()
        self.clip_text_encoder = clip_text_encoder
        self.image_encoder = image_encoder
        self.qpos_dim = qpos_dim
        self.n_emb = n_emb
        
        # CLIP text embedding dimension
        self.text_embed_dim = clip_text_encoder.text_embed_dim
        
        # Learnable projection layer for CLIP text embeddings
        # This allows the model to adapt CLIP embeddings during training
        self.clip_text_proj = nn.Linear(self.text_embed_dim, n_emb)
        
        # Get image feature dimension from image encoder
        # The image encoder outputs features that we'll project to n_emb
        with torch.no_grad():
            # Create dummy image to get output dimension
            dummy_image = torch.zeros(1, 3, 240, 320)  # (B, C, H, W)
            dummy_obs = {"image": dummy_image}
            dummy_features = image_encoder(dummy_obs)  # (B, D_image)
            self.image_feature_dim = dummy_features.shape[-1]
        
        # Projection layers:
        # - qpos+clip_text: concatenated input (qpos_dim + n_emb) -> n_emb
        # - image: image_feature_dim -> n_emb
        self.qpos_text_proj = nn.Linear(qpos_dim + n_emb, n_emb)
        self.image_proj = nn.Linear(self.image_feature_dim, n_emb)
        
        # Total output dim (for compatibility, but we return separate tokens)
        self.output_dim = n_emb  # Each token is n_emb dimensional
        
        encoder_type = "CLIP/SigLIP"  # Generic name since both work
        print(f"ImageTextObsEncoder initialized ({encoder_type} text embeddings + ResNet images):")
        print(f"  Text embed dim: {self.text_embed_dim} → {n_emb} (learnable projection)")
        print(f"  Image feature dim: {self.image_feature_dim} → {n_emb} (projected)")
        print(f"  Qpos dim: {qpos_dim} + projected_text: {n_emb} → {n_emb} (projected)")
        print(f"  Tokens per timestep: 2 [qpos+text, image]")
    
    def forward(self, obs_dict: dict) -> torch.Tensor:
        """
        Args:
            obs_dict: dict with keys:
                - "text": str, list[str], or list[list[str]] - text labels (e.g., "zebra", "blue block")
                - "qpos": (B, 7) tensor - arm joint positions
                - "image": (B, 3, H, W) tensor - RGB images
        
        Returns:
            (B, 2, n_emb) tensor
            Sequence: [qpos+clip_text_1, image_1, qpos+clip_text_2, image_2, ...]
        """
        # Extract text input
        # The training loop already formats text correctly as a list of strings
        # with length B (or B * n_obs_steps), where each string corresponds to the correct trajectory
        text_input = obs_dict.get("text", None)
        if text_input is None:
            raise ValueError("text is required in observations")
        
        # Convert text to list of strings for CLIP encoder
        # The training loop already formats text correctly as a list of strings
        # with length (B * n_obs_steps), where each string corresponds to the correct trajectory
        # After reshape, qpos has shape (B * n_obs_steps, qpos_dim), so B here is (B * n_obs_steps)
        B = obs_dict["qpos"].shape[0]
        if isinstance(text_input, str):
            # Single text string for all samples (shouldn't happen in training, but handle for compatibility)
            texts = [text_input] * B
        elif isinstance(text_input, list):
            # List of text strings - already correctly formatted by training loop
            # Training loop expands text_obs[b_idx] to n_obs_steps copies, so text_input has length (B * n_obs_steps)
            # Each element corresponds to one sample in the reshaped batch
            texts = []
            for text_item in text_input:
                if isinstance(text_item, str):
                    texts.append(text_item)
                elif isinstance(text_item, (list, tuple)):
                    # List of strings per timestep (all same within episode, use first)
                    # This should not happen if training loop formats correctly, but handle for safety
                    if len(text_item) > 0:
                        texts.append(str(text_item[0]))
                    else:
                        texts.append("")  # Fallback to empty string
                else:
                    texts.append(str(text_item))
            
            # CRITICAL: Do NOT extend or truncate - text_input should already have correct length
            # The custom_collate_fn and training loop ensure alignment: text_input length = B (which is B * n_obs_steps after reshape)
            if len(texts) != B:
                raise ValueError(
                    f"CRITICAL ERROR: text_input length ({len(texts)}) != batch size B ({B})! "
                    f"This means text labels are not aligned with trajectories. "
                    f"Text should be pre-formatted by training loop to have length B (B * n_obs_steps after reshape)."
                )
        else:
            raise TypeError(f"Unsupported text type: {type(text_input)}")
        
        # Get text embeddings (frozen text encoder - CLIP or SigLIP)
        with torch.no_grad() if self.clip_text_encoder.freeze else torch.enable_grad():
            text_emb = self.clip_text_encoder.forward_text(texts)  # (B, text_embed_dim)
        
        # Apply learnable projection to text embeddings
        projected_text = self.clip_text_proj(text_emb)  # (B, n_emb)
        
        # Extract pose and image components
        qpos = obs_dict.get("qpos", None)  # (B, 7)
        images = obs_dict.get("image", None)  # (B, 3, H, W)
        
        if qpos is None or images is None:
            raise ValueError("Both 'qpos' and 'image' must be provided")
        
        # Encode images using ResNet encoder
        image_obs = {"image": images}
        image_features = self.image_encoder(image_obs)  # (B, image_feature_dim)
        
        # Project image features to n_emb
        image_tokens = self.image_proj(image_features)  # (B, n_emb)
        
        # Concatenate qpos with projected text embedding, then project
        qpos_text_concat = torch.cat([qpos, projected_text], dim=-1)  # (B, 7 + n_emb)
        qpos_text_tokens = self.qpos_text_proj(qpos_text_concat)  # (B, n_emb)
        
        # Apply LayerNorm to normalize tokens to similar scales
        qpos_text_tokens = F.layer_norm(qpos_text_tokens, (self.n_emb,))
        image_tokens = F.layer_norm(image_tokens, (self.n_emb,))
        
        # Stack tokens: [qpos+text, image] for each sample
        # Shape: (B, 2, n_emb)
        tokens = torch.stack([qpos_text_tokens, image_tokens], dim=1)
        
        return tokens  # (B, 2, n_emb) - will be reshaped in training loop


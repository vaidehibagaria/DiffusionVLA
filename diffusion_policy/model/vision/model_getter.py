import torch
import torch.nn as nn
import torchvision
from typing import List

def get_resnet(name, weights=None, **kwargs):
    """
    name: resnet18, resnet34, resnet50
    weights: "IMAGENET1K_V1", "r3m"
    """
    # load r3m weights
    if (weights == "r3m") or (weights == "R3M"):
        return get_r3m(name=name, **kwargs)

    func = getattr(torchvision.models, name)
    resnet = func(weights=weights, **kwargs)
    resnet.fc = torch.nn.Identity()
    return resnet

def get_r3m(name, **kwargs):
    """
    name: resnet18, resnet34, resnet50
    """
    import r3m
    r3m.device = 'cpu'
    model = r3m.load_r3m(name)
    r3m_model = model.module
    resnet_model = r3m_model.convnet
    resnet_model = resnet_model.to('cpu')
    return resnet_model


class CLIPEncoder(nn.Module):
    """
    CLIP encoder that provides image and text embeddings.
    
    Uses open_clip library with ViT for image embeddings.
    Provides forward_image and forward_text methods that return embeddings
    which can be concatenated for multimodal conditioning.
    """
    
    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "openai",
        device: str = "cpu",
        freeze: bool = True,
    ):
        """
        Args:
            model_name: CLIP model name (e.g., "ViT-B-32", "ViT-B-16", "ViT-L-14")
            pretrained: Pretrained weights (e.g., "openai", "laion2b_s34b_b79k")
            device: Device to load model on
            freeze: If True, freeze CLIP parameters (no training). If False, allow fine-tuning.
        """
        super().__init__()
        
        try:
            import open_clip
        except ImportError:
            raise ImportError(
                "open_clip is not installed. Install it with: pip install open-clip-torch"
            )
        
        self.device = device
        self.model_name = model_name
        self.pretrained = pretrained
        self.freeze = freeze
        
        # Load CLIP model
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name=model_name,
            pretrained=pretrained,
            device=device,
        )
        
        # Set freeze state
        if freeze:
            # Freeze CLIP - always in eval mode, no gradients
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False
        else:
            # Allow fine-tuning - parameters can have gradients
            # Model will respect train/eval mode from parent
            for param in self.model.parameters():
                param.requires_grad = True
        
        # Get embedding dimensions
        with torch.no_grad():
            # Get image embedding dim
            dummy_img = torch.zeros(1, 3, 224, 224, device=device)
            img_emb = self.model.encode_image(dummy_img)
            self.image_embed_dim = img_emb.shape[-1]
            
            # Get text embedding dim
            dummy_text = open_clip.tokenize(["dummy"])
            txt_emb = self.model.encode_text(dummy_text.to(device))
            self.text_embed_dim = txt_emb.shape[-1]
        
        # Total fused embedding dim (image + text)
        self.fused_embed_dim = self.image_embed_dim + self.text_embed_dim
        
        freeze_status = "FROZEN - no training" if freeze else "FINE-TUNABLE - use smaller LR"
        print(f"CLIPEncoder initialized ({freeze_status}):")
        print(f"  Model: {model_name}, Pretrained: {pretrained}")
        print(f"  Image embedding dim: {self.image_embed_dim}")
        print(f"  Text embedding dim: {self.text_embed_dim}")
        print(f"  Fused embedding dim: {self.fused_embed_dim}")
    
    def train(self, mode: bool = True):
        """Override train() to handle frozen vs fine-tunable CLIP."""
        super().train(mode)
        if self.freeze:
            # If frozen, always keep CLIP in eval mode
            self.model.eval()
        else:
            # If fine-tunable, respect training mode
            if mode:
                self.model.train()
            else:
                self.model.eval()
        return self
    
    def eval(self):
        """Override eval()."""
        super().eval()
        self.model.eval()
        return self
    
    def get_clip_parameters(self):
        """
        Get CLIP model parameters. Useful for setting different learning rates.
        
        Returns:
            Generator of CLIP model parameters
        """
        return self.model.parameters()
    
    def get_image_encoder_parameters(self):
        """
        Get only the image encoder (visual) parameters.
        Useful for fine-tuning only the image encoder.
        
        Returns:
            Generator of image encoder parameters
        """
        return self.model.visual.parameters()
    
    def get_text_encoder_parameters(self):
        """
        Get only the text encoder parameters.
        
        Returns:
            Generator of text encoder parameters
        """
        # Text encoder includes transformer, token_embedding, ln_final, text_projection
        for name, param in self.model.named_parameters():
            if 'visual' not in name:  # Everything except visual is text encoder
                yield param
    
    def forward_image(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encode images to embeddings.
        
        Args:
            images: (B, C, H, W) tensor, images in [0, 1] range (normalized)
                   or (B, C, H, W) tensor in [0, 255] range (will be normalized)
        
        Returns:
            (B, D) tensor where D is image embedding dimension
        """
        # CLIP expects images in [0, 1] range
        # If images are in [0, 255], normalize them
        if images.max() > 1.0:
            images = images / 255.0
        
        # CLIP expects images to be 224x224
        # Resize if needed (using interpolation)
        if images.shape[-2:] != (224, 224):
            images = torch.nn.functional.interpolate(
                images, size=(224, 224), mode='bilinear', align_corners=False
            )
        
        # Use no_grad only if frozen
        if self.freeze:
            with torch.no_grad():
                image_features = self.model.encode_image(images)
        else:
            image_features = self.model.encode_image(images)
        
        # Normalize embeddings (CLIP uses normalized embeddings)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        return image_features
    
    def forward_text(self, texts: List[str]) -> torch.Tensor:
        """
        Encode text prompts to embeddings.
        
        Args:
            texts: List of strings, length B (batch size)
        
        Returns:
            (B, D) tensor where D is text embedding dimension
        """
        import open_clip
        
        # Tokenize texts
        text_tokens = open_clip.tokenize(texts).to(self.device)
        
        # Use no_grad only if frozen
        if self.freeze:
            with torch.no_grad():
                text_features = self.model.encode_text(text_tokens)
        else:
            text_features = self.model.encode_text(text_tokens)
        
        # Normalize embeddings (CLIP uses normalized embeddings)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        return text_features
    
    def forward_fused(self, images: torch.Tensor, texts: List[str]) -> torch.Tensor:
        """
        Encode both images and texts, then concatenate embeddings.
        
        Args:
            images: (B, C, H, W) tensor
            texts: List of strings, length B
        
        Returns:
            (B, D_fused) tensor where D_fused = image_embed_dim + text_embed_dim
        """
        img_emb = self.forward_image(images)  # (B, D_img)
        txt_emb = self.forward_text(texts)    # (B, D_txt)
        
        # Concatenate image and text embeddings
        fused = torch.cat([img_emb, txt_emb], dim=-1)  # (B, D_img + D_txt)
        return fused


def get_clip_encoder(
    model_name: str = "ViT-B-32",
    pretrained: str = "openai",
    device: str = "cpu",
    freeze: bool = True,
) -> CLIPEncoder:
    """
    Get a CLIP encoder instance.
    
    Args:
        model_name: CLIP model name (e.g., "ViT-B-32", "ViT-B-16", "ViT-L-14")
        pretrained: Pretrained weights (e.g., "openai", "laion2b_s34b_b79k")
        device: Device to load model on
        freeze: If True, freeze CLIP parameters (no training). If False, allow fine-tuning.
                When fine-tuning, use a smaller learning rate (e.g., 1e-5 vs 1e-4) for CLIP.
    
    Returns:
        CLIPEncoder instance
    
    Example for fine-tuning with different learning rates:
        clip_encoder = get_clip_encoder(freeze=False)
        # In optimizer setup:
        clip_params = list(clip_encoder.get_clip_parameters())
        other_params = [p for p in model.parameters() if p not in clip_params]
        optimizer = torch.optim.AdamW([
            {'params': clip_params, 'lr': 1e-5},      # Smaller LR for CLIP
            {'params': other_params, 'lr': 1e-4}      # Normal LR for rest
        ])
    """
    return CLIPEncoder(model_name=model_name, pretrained=pretrained, device=device, freeze=freeze)

# Franka Arm Manipulation with Image Observations and Text Conditioning

This repository contains code for training and testing a diffusion policy for Franka robot arm manipulation using image observations and text commands (CLIP or SigLIP embeddings).

## Overview

The system enables a Franka robot arm to pick and place objects based on natural language commands. It uses:
- **Image observations**: RGB camera views (320x240) of the workspace
- **Text commands**: Natural language descriptions (e.g., "red cube", "blue block")
- **Diffusion policy**: Transformer-based action prediction using diffusion models

## Files

- **`dataset/collect_data.py`**: Collects demonstration trajectories with images, joint positions, and text labels
- **`train/train.py`**: Trains the diffusion policy model on collected data
- **`franka_test_image.py`**: Tests the trained model in simulation

## How It Works

### 1. Data Collection (`dataset/collect_data.py`)

Collects demonstration trajectories by:
- Capturing RGB images from an overhead camera
- Recording arm joint positions (7D)
- Saving text labels describing the task (e.g., "red cube", "blue block")
- Executing scripted pick-and-place motions

**Usage:**
```bash
python dataset/collect_data.py --collect --num_episodes 100
```

Data is automatically saved to `data/` directory at the repo root.

### 2. Training (`train/train.py`)

Trains a diffusion transformer model that:
- Encodes images using ResNet
- Encodes text using CLIP (default) or SigLIP
- Predicts action sequences using diffusion denoising
- Learns from demonstration trajectories

**Usage:**
```bash
python train/train.py
```

**Configuration:**
- Default uses CLIP text encoder (`text_encoder_type="clip"`)
- To use SigLIP, set `text_encoder_type="siglip"` in the config
- For SigLIP, use model names like `"ViT-B-16-SigLIP"` and pretrained weights like `"webli"`

### 3. Testing (`franka_test_image.py`)

Runs the trained model in closed-loop control:
- Captures current image and joint positions
- Processes text command
- Predicts action sequence using diffusion
- Executes actions and replans

**Usage:**
```bash
python franka_test_image.py --checkpoint outputs/franka_arm_image_training/checkpoints/final --view
```

## Architecture

The model processes observations as follows:
- **Token 1**: Concatenated arm joint positions (7D) + projected text embedding
- **Token 2**: Image embedding from ResNet
- Each timestep has 2 tokens: `[qpos+text, image]`
- Transformer processes these tokens to predict action sequences

## Text Encoders

### CLIP (Default)
- Model: `ViT-B-32` (default)
- Pretrained: `openai` (default)
- Install: `pip install open-clip-torch`

### SigLIP (Optional)
- Model: `ViT-B-16-SigLIP` (example)
- Pretrained: `webli` (example)
- Install: `pip install open-clip-torch` (same as CLIP)

To use SigLIP, modify the config in `train/train.py`:
```python
text_encoder_type: str = "siglip"
clip_model_name: str = "ViT-B-16-SigLIP"
clip_pretrained: str = "webli"
```

## Example Results

[Add your results video here]

The trained model successfully:
- Interprets natural language commands
- Uses visual observations to locate objects
- Executes pick-and-place actions
- Generalizes to different object positions

## Requirements

Install all dependencies using: `pip install -r requirements.txt`

- Python 3.8+
- PyTorch
- MuJoCo
- open-clip-torch (for CLIP/SigLIP)
- diffusers (for diffusion scheduler)

## Notes

- The text encoder (CLIP or SigLIP) is frozen during training
- A learnable projection layer adapts text embeddings to the task
- The model uses receding horizon control: predicts full sequence, executes first few steps, then replans


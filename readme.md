
# Multi-Skip CNN + Fusion Multi-Task Model ( AgriFusion-MTNet )

## Overview

This project implements a **Multi-Skip Convolutional Neural Network (CNN) with Fusion Layers** for multi-task learning in agricultural or phenology-related image analysis. The model simultaneously performs:

- **Segmentation** (e.g., identifying regions of interest in input images)
- **Phenology Regression** (predicting growth stage metrics)
- **Yield Regression** (predicting expected crop yield)

The architecture is designed to handle **multispectral input data** with 12 to 16+ channels and integrates flexible fusion strategies to combine intermediate features effectively.

---

## Pipeline Architecture

### 1. Flexible CNN Encoder
- Accepts input images with 12+ spectral channels.
- Extracts hierarchical feature maps at multiple spatial resolutions.
- Uses multiple convolutional layers and max-pooling.
- Outputs skip connections for rich spatial information retention.

### 2. Fusion Modules
- **Early Fusion**: Concatenates input channels before encoding (not the main fusion here but possible).
- **Late Fusion**: Combines high-level features from parallel network branches by averaging.
- **Gated Fusion**: Learns attention-like gating to weight the importance of different feature maps dynamically.

### 3. Multi-Task Heads
- **Segmentation Head**: Performs pixel-wise classification to segment relevant regions.
- **Phenology Head**: Uses global feature pooling followed by fully connected layers to regress phenology-related continuous variables.
- **Yield Head**: Similar regression head predicting crop yield values.

---

## Usage

1. **Prepare your multispectral input data** with 12 or more channels.
2. **Train the model** with labeled data for segmentation masks, phenology measurements, and yield values.
3. The model outputs:
   - Segmentation maps matching input spatial dimensions.
   - Phenology regression predictions.
   - Yield regression predictions.

---

## File Structure

- `base_encoder.py` — Defines the CNN encoder with multi-scale skip connections.
- `fusion_modules.py` — Implements fusion layers: early, late, gated fusion.
- `task_heads.py` — Contains segmentation and regression heads for multi-task learning.
- `multitask_model.py` — Combines encoder, fusion, and task heads into a unified Keras model.

---

## Requirements

Use the provided `requirements.txt` or create a Conda environment with the necessary packages, e.g., TensorFlow 2.11+, NumPy, etc.

---

## License

 will be updated later

---

## Contact

For questions or collaboration, please contact: `chawthiri177@gmail.com`

---



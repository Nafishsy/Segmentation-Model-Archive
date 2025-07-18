# Deep Learning Segmentation Models Archive

Personal archive of deep learning segmentation models implemented as part of ongoing research in biomedical image segmentation.

## Models Implemented

### Classical Architectures
- **U-Net** - Encoder-decoder with skip connections
- **SegNet** - Encoder-decoder with pooling indices  
- **Residual U-Net** - U-Net with residual connections
- **Dense U-Net** - Dense blocks for feature reuse
- **Attention U-Net** - Attention gates for feature selection
- **UNet++** - Nested U-Net with dense skip pathways

### Advanced Architectures
- **ASPP-enhanced models** - Multi-scale context with atrous convolutions
- **Transformer-based variants** - Self-attention mechanisms
- **Hybrid Models** - UNet++ with ASPP and attention, SegNet with transformer blocks

## Repository Structure

- **Notebooks** - `Models.ipynb` contains code, training, evaluation, and visualization
- **Python Scripts** - Each model available as standalone file
- **File naming** - `<architecture>_<alteration>_model.py`

## Usage

Browse notebooks or Python scripts to select a model. Training, evaluation, and visualization routines are included.

## Requirements

```bash
pip install tensorflow keras numpy opencv-python scikit-image matplotlib pandas patchify
```

## Evaluation Metrics

Models evaluated using Dice, Jaccard, Accuracy, Precision, and Recall.

---

*Research archive documenting progress in computer vision and biomedical image segmentation.*

"""Emoji provider classification: shared code for the notebook and the scripts.

Modules
-------
data        : locate/extract the Kaggle archive, scan native PNG metadata, preprocess and cache pixels,
              deduplicate, build/load the validation split, PyTorch datasets with signature-preserving augmentation
models      : compact CNNs trained from scratch, optional metadata late fusion, timm backbones (ConvNeXt, ResNet)
train       : training loop (AdamW + OneCycle, AMP on CUDA), checkpointing, prediction with optional TTA
evaluate    : metrics, confusion matrix, per-class F1, learning curves, failure grids, Grad-CAM
metadata_model : the metadata-only diagnostic (file format, no pixels)
submit      : submission writer with integrity checks and optional post-processing rules
"""
__version__ = "1.0.0"

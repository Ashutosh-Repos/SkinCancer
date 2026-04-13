"""
Skin Cancer Detection System

A production-grade deep learning system for automated skin lesion classification.
Supports from-scratch CNN models and transfer learning with EfficientNet, 
ResNet50, DenseNet121, and Vision Transformers.
"""

__version__ = '2.0.0'
__author__ = 'Skin Cancer Detection Team'

from . import config
from . import data_loader
from . import models
from . import inference
from . import gradcam

__all__ = [
    'config',
    'data_loader',
    'models',
    'inference',
    'gradcam',
]

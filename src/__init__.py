"""
Skin Cancer Detection System

A production-grade deep learning system for automated skin lesion classification.
Supports from-scratch CNN models and transfer learning with EfficientNet, 
ResNet50, DenseNet121, and Vision Transformers.
"""

__version__ = '2.0.0'
__author__ = 'Skin Cancer Detection Team'

# Lazy imports — TensorFlow-heavy modules are only loaded when accessed,
# not on `import src` or `from src import config`.
__all__ = [
    'config',
    'data_loader',
    'models',
    'inference',
    'gradcam',
]


def __getattr__(name):
    """Lazy-load submodules to avoid importing TensorFlow unnecessarily."""
    if name in __all__:
        import importlib
        return importlib.import_module(f'.{name}', __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

"""
Data loading and preprocessing module.
Handles dataset loading, preprocessing, and data augmentation.
"""

import os
import gc
import json
import numpy as np
import pandas as pd
from PIL import Image
from typing import Tuple
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from config import (
    DATASET_CONFIG, 
    LESION_CLASSES, 
    CLASS_TO_INDEX,
    INDEX_TO_CLASS,
    TRAINING_CONFIG,
    AUGMENTATION_CONFIG
)


class DataLoader:
    """Handles loading and preprocessing of skin lesion dataset."""
    
    def __init__(self, metadata_path: str, images_dir: str, image_size: tuple = None):
        """
        Initialize the data loader.
        
        Args:
            metadata_path: Path to the metadata CSV file
            images_dir: Directory containing the images
            image_size: Optional override for image size (height, width).
                        If None, uses DATASET_CONFIG default.
        """
        self.metadata_path = metadata_path
        self.images_dir = images_dir
        self.image_size = image_size or DATASET_CONFIG['image_size']
        self.num_classes = DATASET_CONFIG['num_classes']
        
        self.metadata_df = None
        
    def load_metadata(self) -> pd.DataFrame:
        """Load and preprocess metadata."""
        print("Loading metadata...")
        self.metadata_df = pd.read_csv(self.metadata_path)
        
        # Handle missing values in age
        age_mean = self.metadata_df['age'].mean()
        self.metadata_df['age'] = self.metadata_df['age'].fillna(age_mean)
        
        # Create cell type mapping using explicit CLASS_TO_INDEX
        # (pd.Categorical assigns alphabetically which mismatches INDEX_TO_CLASS)
        self.metadata_df['cell_type'] = self.metadata_df['dx'].map(LESION_CLASSES)
        self.metadata_df['cell_type_idx'] = self.metadata_df['cell_type'].map(CLASS_TO_INDEX)
        
        # Map image paths
        image_paths = self._get_image_paths()
        self.metadata_df['path'] = self.metadata_df['image_id'].map(image_paths.get)
        
        # Validate: check how many images were found
        found = self.metadata_df['path'].notna().sum()
        total = len(self.metadata_df)
        print(f"Loaded {total} samples, found {found}/{total} image files")
        
        if found < total * 0.5:
            # Most images missing — show diagnostic info
            print(f"\nERROR: Only {found}/{total} images found!")
            print(f"  Expected images in: {self.images_dir}")
            data_dir = os.path.dirname(self.images_dir) or '.'
            print(f"  Contents of {data_dir}/:")
            for item in sorted(os.listdir(data_dir)):
                full = os.path.join(data_dir, item)
                if os.path.isdir(full):
                    n = len(os.listdir(full))
                    print(f"    {item}/ ({n} files)")
                else:
                    print(f"    {item}")
            raise FileNotFoundError(
                f"Only {found}/{total} images found in {self.images_dir}. "
                f"Re-run: python scripts/download_dataset.py"
            )
        
        return self.metadata_df
    
    def _get_image_paths(self) -> dict:
        """Create a mapping of image IDs to file paths.
        
        Searches the images_dir first, then falls back to searching
        the entire parent data directory to handle different Kaggle
        extraction layouts.
        """
        image_paths = {}
        
        # Primary search: images_dir (data/images/)
        search_dirs = [self.images_dir]
        
        # Fallback: also search parent data directory
        data_dir = os.path.dirname(self.images_dir)
        if data_dir and data_dir != self.images_dir:
            search_dirs.append(data_dir)
        
        for search_dir in search_dirs:
            if not os.path.exists(search_dir):
                continue
            for root, _, files in os.walk(search_dir):
                for file in files:
                    if file.endswith('.jpg'):
                        image_id = os.path.splitext(file)[0]
                        if image_id not in image_paths:
                            image_paths[image_id] = os.path.join(root, file)
        
        return image_paths
    
    def load_images(self) -> None:
        """Load and preprocess all images."""
        print(f"Loading images (resizing to {self.image_size[1]}x{self.image_size[0]})...")
        images = []
        errors = 0
        for i, path in enumerate(self.metadata_df['path']):
            try:
                img = Image.open(path).resize(
                    (self.image_size[1], self.image_size[0])  # (width, height)
                )
                images.append(np.array(img))
            except Exception as e:
                errors += 1
                if errors <= 5:  # Only print first 5 errors
                    print(f"Error loading image {path}: {e}")
                # Add a blank image as placeholder
                images.append(np.zeros((*self.image_size, 3), dtype=np.uint8))
            
            if (i + 1) % 2000 == 0:
                print(f"  Loaded {i + 1}/{len(self.metadata_df)} images...")
        
        if errors > 5:
            print(f"  ... and {errors - 5} more errors (total: {errors})")
        
        self.metadata_df['image'] = images
        print(f"Loaded {len(images)} images ({errors} errors)")
    
    def prepare_data(self, normalize: object = True) -> Tuple:
        """
        Prepare train, validation, and test datasets.
        
        Args:
            normalize: Controls image normalization.
                - True: Custom mean/std normalization (for Sequential/ResNet)
                - False: Keep raw [0, 255] pixels (for EfficientNet, ResNet50, DenseNet)
                - 'rescale': Scale to [0, 1] range (for ViT)
        
        Returns:
            Tuple of (X_train, y_train, X_val, y_val, X_test, y_test)
        """
        if self.metadata_df is None:
            raise ValueError("Metadata not loaded. Call load_metadata() first.")
        
        if 'image' not in self.metadata_df.columns:
            self.load_images()
        
        print("Preparing datasets...")
        
        # Extract features as uint8 first (4× less memory than float32).
        # For 224×224 images, uint8 array = ~1.5 GB vs float32 = ~6 GB.
        # We split as uint8, then convert each split to float32 separately.
        X = np.array(self.metadata_df['image'].tolist(), dtype=np.uint8)
        y = self.metadata_df['cell_type_idx'].values
        
        # Free the DataFrame images to reclaim memory
        if 'image' in self.metadata_df.columns:
            self.metadata_df = self.metadata_df.drop(columns=['image'])
            gc.collect()
            print("  Freed DataFrame image memory")
        
        # Split into train+val and test (still uint8, low memory)
        test_split = TRAINING_CONFIG['test_split']
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, 
            test_size=test_split, 
            random_state=42,
            stratify=y
        )
        
        # Free the full X array (no longer needed after split)
        del X
        gc.collect()
        
        # Split train into train and validation (still uint8)
        val_split = TRAINING_CONFIG['validation_split']
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val,
            test_size=val_split,
            random_state=42,
            stratify=y_train_val
        )
        
        # Free train_val arrays
        del X_train_val, y_train_val
        gc.collect()
        
        # NOW convert splits to float32 one at a time (keeps peak memory low)
        X_train = X_train.astype(np.float32)
        X_val = X_val.astype(np.float32)
        X_test = X_test.astype(np.float32)
        gc.collect()
        print("  Converted splits to float32")
        
        # Apply normalization based on mode
        if normalize is True:
            # Custom mean/std normalization (for from-scratch models)
            train_mean = X_train.mean()
            train_std = X_train.std()
            
            # Safety: if std is ~0, images are all black/placeholders
            if train_std < 1.0:
                raise ValueError(
                    f"Training data has near-zero variance (mean={train_mean:.2f}, "
                    f"std={train_std:.2f}). This means most images are blank. "
                    f"Check that images exist in data/images/."
                )
            
            X_train = (X_train - train_mean) / train_std
            X_val = (X_val - train_mean) / train_std
            X_test = (X_test - train_mean) / train_std
            
            self.train_mean = train_mean
            self.train_std = train_std
            
            # Save normalization stats for inference
            self._save_norm_stats(train_mean, train_std)
            print(f"  Normalized with mean={train_mean:.2f}, std={train_std:.2f}")
            
        elif normalize == 'rescale':
            # Scale to [0, 1] range (for ViT)
            X_train = X_train / 255.0
            X_val = X_val / 255.0
            X_test = X_test / 255.0
            print("  Rescaled to [0, 1] range (ViT)")
            
        else:
            # Keep raw [0, 255] pixels (for EfficientNet, ResNet50, DenseNet)
            print("  No normalization applied (model handles preprocessing)")
        
        # Convert labels to categorical
        y_train = to_categorical(y_train, num_classes=self.num_classes)
        y_val = to_categorical(y_val, num_classes=self.num_classes)
        y_test = to_categorical(y_test, num_classes=self.num_classes)
        
        print(f"Train samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        print(f"Test samples: {len(X_test)}")
        
        return X_train, y_train, X_val, y_val, X_test, y_test
    
    def _save_norm_stats(self, mean: float, std: float):
        """Save normalization statistics for inference."""
        stats = {'mean': float(mean), 'std': float(std)}
        stats_path = os.path.join(
            os.path.dirname(self.metadata_path), 'norm_stats.json'
        )
        try:
            with open(stats_path, 'w') as f:
                json.dump(stats, f, indent=2)
            print(f"  Normalization stats saved to {stats_path}")
        except Exception as e:
            print(f"  Warning: Could not save norm stats: {e}")
    
    def get_data_generator(self) -> ImageDataGenerator:
        """
        Create a data augmentation generator.
        
        Returns:
            Configured ImageDataGenerator
        """
        return ImageDataGenerator(**AUGMENTATION_CONFIG)
    
    def get_class_weights(self, y_train_categorical: np.ndarray) -> dict:
        """
        Compute balanced class weights to handle class imbalance.
        
        Args:
            y_train_categorical: One-hot encoded training labels
            
        Returns:
            Dictionary mapping class index to weight
        """
        y_integers = np.argmax(y_train_categorical, axis=1)
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(y_integers),
            y=y_integers
        )
        weights_dict = dict(enumerate(class_weights))
        print("\nClass weights computed:")
        for idx, weight in weights_dict.items():
            class_code = INDEX_TO_CLASS[idx]
            class_name = LESION_CLASSES[class_code]
            print(f"  {class_name}: {weight:.3f}")
        return weights_dict
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess a single image for inference.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Preprocessed image
        """
        # Resize image
        img = Image.fromarray(image)
        img = img.resize((self.image_size[1], self.image_size[0]))
        img_array = np.array(img, dtype=np.float32)
        
        # Normalize using training statistics
        if hasattr(self, 'train_mean') and hasattr(self, 'train_std'):
            img_array = (img_array - self.train_mean) / self.train_std
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    
    def get_class_distribution(self) -> pd.Series:
        """Get the distribution of classes in the dataset."""
        if self.metadata_df is None:
            raise ValueError("Metadata not loaded.")
        return self.metadata_df['dx'].value_counts()


def load_dataset(image_size: tuple = None, normalize: object = True) -> Tuple:
    """
    Convenience function to load the complete dataset.
    
    Args:
        image_size: Optional override for image size (height, width)
        normalize: Normalization mode (True, False, or 'rescale')
    
    Returns:
        Tuple of (X_train, y_train, X_val, y_val, X_test, y_test, data_loader)
    """
    data_loader = DataLoader(
        metadata_path=DATASET_CONFIG['metadata_file'],
        images_dir=DATASET_CONFIG['images_dir'],
        image_size=image_size
    )
    
    data_loader.load_metadata()
    X_train, y_train, X_val, y_val, X_test, y_test = data_loader.prepare_data(
        normalize=normalize
    )
    
    return X_train, y_train, X_val, y_val, X_test, y_test, data_loader


def load_test_only(image_size: tuple = None, normalize: object = True) -> Tuple:
    """
    Load ONLY the test split, immediately freeing train/val from memory.
    
    Uses the same random_state=42 split as load_dataset(), so the test set
    is identical. This is ~6× more memory-efficient for evaluation.
    
    Args:
        image_size: Optional override for image size (height, width)
        normalize: Normalization mode (True, False, or 'rescale')
    
    Returns:
        Tuple of (X_test, y_test)
    """
    data_loader = DataLoader(
        metadata_path=DATASET_CONFIG['metadata_file'],
        images_dir=DATASET_CONFIG['images_dir'],
        image_size=image_size
    )
    
    data_loader.load_metadata()
    X_train, y_train, X_val, y_val, X_test, y_test = data_loader.prepare_data(
        normalize=normalize
    )
    
    # Immediately free train/val to save memory
    del X_train, y_train, X_val, y_val
    gc.collect()
    print(f"  Freed train/val memory (keeping {len(X_test)} test samples only)")
    
    return X_test, y_test


"""
Grad-CAM (Gradient-weighted Class Activation Mapping) visualization module.
Generates visual explanations showing which image regions the model focuses on.
Critical for medical imaging trust and interpretability.
"""

import os
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image

from config import INDEX_TO_CLASS, LESION_CLASSES, DATASET_CONFIG


class GradCAM:
    """Generates Grad-CAM heatmaps for model predictions."""
    
    def __init__(self, model_path: str):
        """
        Initialize Grad-CAM with a trained model.
        
        Args:
            model_path: Path to saved model file
        """
        print(f"Loading model for Grad-CAM from {model_path}...")
        self.model = load_model(model_path)
        
        # Auto-detect settings from metadata
        self._load_settings(model_path)
        
        # Will be set by _find_last_conv_layer if model has a nested backbone
        self._backbone_name = None
        
        # Find the last convolutional layer for Grad-CAM
        self.last_conv_layer = self._find_last_conv_layer()
        print(f"  Last conv layer: {self.last_conv_layer}")
        print(f"  Image size: {self.image_size}")
        
        # Force the model to build/trace if it's a Sequential model
        # This fixes "layer has never been called" error
        try:
            dummy_input = tf.zeros((1, *self.image_size, 3))
            _ = self.model(dummy_input)
            print("  Model warmed up successfully")
        except Exception as e:
            print(f"  Note: Model warmup skipped ({e})")
            
        print("Model loaded for Grad-CAM!")
    
    def _load_settings(self, model_path: str):
        """Load model settings from metadata."""
        metadata_path = model_path.replace('.h5', '_metadata.json')
        
        if os.path.exists(metadata_path):
            with open(metadata_path) as f:
                metadata = json.load(f)
            self.image_size = tuple(metadata.get('image_size', [90, 120]))
            self.normalize_mode = metadata.get('normalize', True)
        else:
            input_shape = self.model.input_shape
            if isinstance(input_shape, list):
                input_shape = input_shape[0]
            h, w = input_shape[1], input_shape[2]
            if h and w:
                self.image_size = (h, w)
                self.normalize_mode = False if (h == 224) else True
            else:
                self.image_size = DATASET_CONFIG['image_size']
                self.normalize_mode = True
        
        # Load norm stats if needed
        self.train_mean, self.train_std = 160.0, 46.7
        if self.normalize_mode is True:
            stats_path = os.path.join('data', 'norm_stats.json')
            if os.path.exists(stats_path):
                with open(stats_path) as f:
                    stats = json.load(f)
                self.train_mean = stats['mean']
                self.train_std = stats['std']
    
    def _find_last_conv_layer(self) -> str:
        """Find the name of the last convolutional layer with spatial dimensions."""
        last_conv_name = None
        
        # Iterate backwards to find the last Conv2D that has spatial extent > 1
        for layer in reversed(self.model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                # Try to get shape from output or input
                try:
                    shape = layer.output.shape
                    h, w = shape[1], shape[2]
                except Exception:
                    # Fallback for models where output isn't connected
                    h, w = 0, 0
                
                if h is not None and w is not None and h > 1 and w > 1:
                    last_conv_name = layer.name
                    break
        
        # If not found at top level, search inside nested models
        if last_conv_name is None:
            for layer in reversed(self.model.layers):
                if hasattr(layer, 'layers'):
                    for sublayer in reversed(layer.layers):
                        if isinstance(sublayer, tf.keras.layers.Conv2D):
                            try:
                                shape = sublayer.output.shape
                                h, w = shape[1], shape[2]
                            except Exception:
                                h, w = 0, 0
                                
                            if h is not None and w is not None and h > 1 and w > 1:
                                last_conv_name = sublayer.name
                                self._backbone_name = layer.name
                                break
                    if last_conv_name: break
        
        # Fallback to absolute last if none with spatial found
        if last_conv_name is None:
            for layer in reversed(self.model.layers):
                if isinstance(layer, tf.keras.layers.Conv2D):
                    last_conv_name = layer.name
                    break
                    
        if last_conv_name is None:
            raise ValueError("No Conv2D layer found in model")
        
        return last_conv_name
    
    def preprocess_image(self, image_path: str) -> tuple:
        """
        Load and preprocess image for Grad-CAM.
        
        Returns:
            (preprocessed_tensor, original_image)
        """
        # Load original image
        img = Image.open(image_path).convert('RGB')
        original = img.resize((self.image_size[1], self.image_size[0]))
        original_array = np.array(original)
        
        # Preprocess for model
        img_array = np.array(original, dtype=np.float32)
        
        if self.normalize_mode is True:
            img_array = (img_array - self.train_mean) / self.train_std
        elif self.normalize_mode == 'rescale':
            img_array = img_array / 255.0
        # else: keep [0, 255]
        
        img_tensor = np.expand_dims(img_array, axis=0)
        
        return img_tensor, original_array
    
    def compute_heatmap(self, img_tensor: np.ndarray, pred_index: int = None) -> np.ndarray:
        """
        Compute Grad-CAM heatmap.
        """
        # 1. Try to find the target layer
        target_layer = None
        for layer in self.model.layers:
            if layer.name == self.last_conv_layer:
                target_layer = layer
                break
        
        # 2. Build the grad_model
        grad_model = None
        try:
            # Try functional/direct way first
            grad_model = tf.keras.Model(
                inputs=self.model.input,
                outputs=[target_layer.output, self.model.output]
            )
        except Exception:
            # Fallback for Sequential models: Reconstruct Functional graph
            try:
                img_input = tf.keras.Input(shape=(*self.image_size, 3))
                x = img_input
                target_out = None
                for layer in self.model.layers:
                    x = layer(x)
                    if layer.name == self.last_conv_layer:
                        target_out = x
                
                if target_out is not None:
                    grad_model = tf.keras.Model(inputs=img_input, outputs=[target_out, x])
            except Exception as e:
                print(f"  Warning: Reconstruction fallback failed: {e}")

        if grad_model is None:
            print("  Warning: Could not define gradient graph. Returning empty heatmap.")
            return np.zeros(self.image_size, dtype=np.float32)

        # 3. Compute gradients with Tape
        try:
            with tf.GradientTape() as tape:
                conv_outputs, predictions = grad_model(img_tensor)
                
                # Unwrap list if necessary
                if isinstance(predictions, list):
                    predictions = predictions[0]
                
                # Ensure predictions is 2D (batch_size, num_classes)
                if hasattr(predictions, 'shape'):
                    if len(predictions.shape) == 1:
                        predictions = tf.expand_dims(predictions, axis=0)
                else:
                    predictions = tf.convert_to_tensor(predictions)
                
                if pred_index is None:
                    pred_index = tf.argmax(predictions[0])
                
                # Extract probability for target class
                class_output = predictions[:, pred_index]

            # Get gradients
            grads = tape.gradient(class_output, conv_outputs)
            
            if grads is None:
                print("  Warning: Gradients are None.")
                return np.zeros(self.image_size, dtype=np.float32)

            # Pooling and heatmap generation
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            
            # Use simple channel-wise multiplication for stability
            heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs[0]), axis=-1)
            
            heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
            return heatmap.numpy()
            
        except Exception as e:
            import traceback
            print(f"  Warning: Grad-CAM tape failed: {e}")
            traceback.print_exc()
            return np.zeros(self.image_size, dtype=np.float32)
    
    def overlay_heatmap(
        self, 
        heatmap: np.ndarray, 
        original_image: np.ndarray, 
        alpha: float = 0.4,
        colormap: str = 'jet'
    ) -> np.ndarray:
        """
        Overlay heatmap on original image.
        
        Args:
            heatmap: Grad-CAM heatmap (H', W')
            original_image: Original image (H, W, 3)
            alpha: Heatmap opacity
            colormap: Matplotlib colormap name
        
        Returns:
            Overlay image as numpy array (H, W, 3)
        """
        # Resize heatmap to match original image
        heatmap_resized = np.array(
            Image.fromarray((heatmap * 255).astype(np.uint8)).resize(
                (original_image.shape[1], original_image.shape[0])
            )
        ).astype(np.float32) / 255.0
        
        # Apply colormap
        cmap = plt.get_cmap(colormap)
        heatmap_colored = cmap(heatmap_resized)[:, :, :3]  # Drop alpha channel
        heatmap_colored = (heatmap_colored * 255).astype(np.uint8)
        
        # Overlay
        overlay = np.uint8(alpha * heatmap_colored + (1 - alpha) * original_image)
        
        return overlay
    
    def visualize(
        self, 
        image_path: str, 
        save_path: str = None,
        show_top_k: int = 3
    ) -> str:
        """
        Generate and save Grad-CAM visualization.
        
        Args:
            image_path: Path to input image
            save_path: Path to save output (auto-generated if None)
            show_top_k: Number of top predictions to show
        
        Returns:
            Path to saved visualization
        """
        # Preprocess
        img_tensor, original = self.preprocess_image(image_path)
        
        # Get prediction
        predictions = self.model.predict(img_tensor, verbose=0)
        pred_idx = np.argmax(predictions[0])
        confidence = predictions[0][pred_idx]
        class_code = INDEX_TO_CLASS[pred_idx]
        class_name = LESION_CLASSES[class_code]
        
        # Compute heatmap
        try:
            heatmap = self.compute_heatmap(img_tensor, pred_idx)
            overlay = self.overlay_heatmap(heatmap, original)
            has_heatmap = True
        except Exception as e:
            print(f"Warning: Could not generate heatmap: {e}")
            has_heatmap = False
        
        # Create visualization
        if has_heatmap:
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            
            axes[0].imshow(original)
            axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
            axes[0].axis('off')
            
            axes[1].imshow(heatmap, cmap='jet')
            axes[1].set_title('Grad-CAM Heatmap', fontsize=14, fontweight='bold')
            axes[1].axis('off')
            
            axes[2].imshow(overlay)
            axes[2].set_title('Overlay', fontsize=14, fontweight='bold')
            axes[2].axis('off')
        else:
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            ax.imshow(original)
            ax.set_title('Original Image', fontsize=14, fontweight='bold')
            ax.axis('off')
        
        # Add prediction text
        pred_text = f"Prediction: {class_name} ({confidence*100:.1f}%)"
        fig.suptitle(pred_text, fontsize=16, fontweight='bold', y=1.02)
        
        # Add top-k predictions as text
        sorted_preds = sorted(enumerate(predictions[0]), key=lambda x: x[1], reverse=True)
        top_k_text = "Top predictions: "
        for i, (idx, prob) in enumerate(sorted_preds[:show_top_k]):
            code = INDEX_TO_CLASS[idx]
            name = LESION_CLASSES[code]
            top_k_text += f"{name}: {prob*100:.1f}%  |  "
        
        fig.text(0.5, -0.02, top_k_text, ha='center', fontsize=10, style='italic')
        
        plt.tight_layout()
        
        # Save
        if save_path is None:
            os.makedirs('results/gradcam', exist_ok=True)
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            save_path = f'results/gradcam/gradcam_{base_name}.png'
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\nGrad-CAM visualization saved to {save_path}")
        print(f"Prediction: {class_name} ({confidence*100:.1f}%)")
        
        return save_path


def main():
    """Main Grad-CAM function."""
    parser = argparse.ArgumentParser(
        description='Generate Grad-CAM visualization for skin lesion predictions'
    )
    parser.add_argument(
        '--model', type=str, required=True,
        help='Path to trained model file (.h5)'
    )
    parser.add_argument(
        '--image', type=str, required=True,
        help='Path to input image'
    )
    parser.add_argument(
        '--output', type=str, default=None,
        help='Path to save Grad-CAM visualization'
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        return
    if not os.path.exists(args.image):
        print(f"Error: Image file not found: {args.image}")
        return
    
    cam = GradCAM(model_path=args.model)
    cam.visualize(image_path=args.image, save_path=args.output)


if __name__ == '__main__':
    main()

"""
Training script for skin cancer detection models.
Handles model training with proper logging, checkpointing, and metrics tracking.
Supports from-scratch and transfer learning models with two-stage fine-tuning.
"""

import os
import gc
import json
import argparse
from datetime import datetime
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

from tensorflow.keras.callbacks import (
    ModelCheckpoint, 
    EarlyStopping, 
    ReduceLROnPlateau,
    CSVLogger,
    TensorBoard
)
from tensorflow.keras.optimizers import Adam

from config import (
    TRAINING_CONFIG,
    CLR_CONFIG,
    MODEL_CONFIG,
    PATHS,
    TRANSFER_LEARNING_MODELS,
    ensure_directories
)
from data_loader import load_dataset
from models import get_model, CyclicLR


class ModelTrainer:
    """Handles model training and evaluation."""
    
    def __init__(self, model_type: str = 'sequential', use_class_weights: bool = False):
        """
        Initialize the trainer.
        
        Args:
            model_type: Type of model to train
            use_class_weights: Whether to use class weights for imbalance
        """
        self.model_type = model_type
        self.use_class_weights = use_class_weights
        self.model = None
        self.history = None
        self.data_loader = None
        self.class_weights = None
        
        ensure_directories()
        
        # Create timestamped run directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.run_name = f"{model_type}_{timestamp}"
        self.run_dir = os.path.join(PATHS['logs_dir'], self.run_name)
        os.makedirs(self.run_dir, exist_ok=True)
        
        print(f"Training run: {self.run_name}")
    
    def load_data(self):
        """Load and prepare dataset with correct preprocessing for model type."""
        print("\n" + "="*60)
        print("LOADING DATASET")
        print("="*60)
        
        # Determine image size and normalization based on model type
        if self.model_type in TRANSFER_LEARNING_MODELS:
            config = MODEL_CONFIG[self.model_type]
            image_size = config['image_size']
            normalize = config['normalize']
            print(f"Using image size: {image_size} (transfer learning)")
            print(f"Normalization mode: {normalize}")
        else:
            image_size = None  # Use default (90, 120)
            normalize = True   # Custom mean/std normalization
            print(f"Using default image size (90, 120)")
        
        (self.X_train, self.y_train, 
         self.X_val, self.y_val, 
         self.X_test, self.y_test, 
         self.data_loader) = load_dataset(
             image_size=image_size,
             normalize=normalize
         )
        
        # Compute class weights if requested
        if self.use_class_weights:
            self.class_weights = self.data_loader.get_class_weights(self.y_train)
        
        print("\nDataset loaded successfully!")
        print(f"Training set shape: {self.X_train.shape}")
        print(f"Validation set shape: {self.X_val.shape}")
        print(f"Test set shape: {self.X_test.shape}")
    
    def build_model(self):
        """Build and compile model."""
        print("\n" + "="*60)
        print("BUILDING MODEL")
        print("="*60)
        
        self.model = get_model(self.model_type)
        
        # Save model architecture (may fail for some models)
        try:
            model_json = self.model.to_json()
            json_path = os.path.join(self.run_dir, 'model_architecture.json')
            with open(json_path, 'w') as f:
                json.dump(json.loads(model_json), f, indent=2)
            print(f"\nModel architecture saved to {json_path}")
        except Exception as e:
            print(f"\nNote: Could not save model JSON ({e})")
        
        # Save model summary
        summary_path = os.path.join(self.run_dir, 'model_summary.txt')
        with open(summary_path, 'w') as f:
            self.model.summary(print_fn=lambda x: f.write(x + '\n'))
    
    def get_callbacks(self, stage: str = ''):
        """Create training callbacks."""
        callbacks = []
        
        # Model checkpoint
        checkpoint_path = os.path.join(
            PATHS['checkpoints_dir'], 
            f'{self.model_type}_best.h5'
        )
        checkpoint = ModelCheckpoint(
            checkpoint_path,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        )
        callbacks.append(checkpoint)
        
        # CSV logger (append for stage 2)
        csv_path = os.path.join(self.run_dir, 'training_log.csv')
        csv_logger = CSVLogger(csv_path, append=(stage == 'stage2'))
        callbacks.append(csv_logger)
        
        # TensorBoard
        tensorboard_dir = os.path.join(self.run_dir, 'tensorboard')
        tensorboard = TensorBoard(
            log_dir=tensorboard_dir,
            histogram_freq=1,
            write_graph=True
        )
        callbacks.append(tensorboard)
        
        if self.model_type == 'sequential':
            # Learning rate reduction for sequential model
            reduce_lr = ReduceLROnPlateau(
                monitor='val_loss',
                patience=TRAINING_CONFIG['reduce_lr_patience'],
                verbose=1,
                factor=TRAINING_CONFIG['reduce_lr_factor'],
                min_lr=TRAINING_CONFIG['min_lr']
            )
            callbacks.append(reduce_lr)
            
            # Early stopping
            early_stop = EarlyStopping(
                monitor='val_loss',
                patience=TRAINING_CONFIG['early_stopping_patience'],
                verbose=1,
                restore_best_weights=True
            )
            callbacks.append(early_stop)
        
        elif self.model_type == 'resnet':
            # Cyclic LR for custom ResNet
            steps_per_epoch = len(self.X_train) // TRAINING_CONFIG['batch_size']
            step_size = CLR_CONFIG['step_size_multiplier'] * steps_per_epoch
            
            clr = CyclicLR(
                base_lr=CLR_CONFIG['base_lr'],
                max_lr=CLR_CONFIG['max_lr'],
                step_size=step_size,
                mode=CLR_CONFIG['mode']
            )
            callbacks.append(clr)
        
        elif self.model_type in TRANSFER_LEARNING_MODELS:
            # Early stopping for transfer learning
            early_stop = EarlyStopping(
                monitor='val_accuracy',
                patience=5,
                verbose=1,
                restore_best_weights=True,
                mode='max'
            )
            callbacks.append(early_stop)
        
        return callbacks
    
    def _save_model_metadata(self):
        """Save model metadata for inference/evaluation auto-detection."""
        metadata = {
            'model_type': self.model_type,
            'image_size': list(self.data_loader.image_size),
            'normalize': MODEL_CONFIG.get(self.model_type, {}).get('normalize', True),
            'class_weights_used': self.use_class_weights,
            'timestamp': datetime.now().isoformat(),
        }
        
        # Save alongside the best checkpoint
        metadata_path = os.path.join(
            PATHS['checkpoints_dir'],
            f'{self.model_type}_best_metadata.json'
        )
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Also save in the run directory
        run_metadata_path = os.path.join(self.run_dir, 'model_metadata.json')
        with open(run_metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Model metadata saved")
    
    def train(self, epochs: int = None, batch_size: int = None):
        """
        Train the model.
        
        For transfer learning models, implements two-stage training:
          Stage 1: Train only the classification head (base frozen)
          Stage 2: Fine-tune top layers of the base model
        
        Args:
            epochs: Number of epochs (uses config default if None)
            batch_size: Batch size (uses config default if None)
        """
        if epochs is None:
            epochs = TRAINING_CONFIG['epochs']
        if batch_size is None:
            batch_size = TRAINING_CONFIG['batch_size']
        
        # Get data augmentation generator
        # NOTE: datagen.fit() is NOT needed — our augmentation config only uses
        # rotation, zoom, shift, flip, which don't require dataset statistics.
        # Calling fit() on 224×224 data wastes ~1 GB of temporary memory.
        datagen = self.data_loader.get_data_generator()
        
        # Calculate steps per epoch
        steps_per_epoch = len(self.X_train) // batch_size
        
        if self.model_type in TRANSFER_LEARNING_MODELS:
            self._train_transfer_learning(datagen, batch_size, steps_per_epoch)
        else:
            self._train_from_scratch(datagen, epochs, batch_size, steps_per_epoch)
    
    def _train_from_scratch(self, datagen, epochs, batch_size, steps_per_epoch):
        """Train from-scratch models (Sequential CNN, Custom ResNet)."""
        print("\n" + "="*60)
        print("TRAINING MODEL")
        print("="*60)
        print(f"Epochs: {epochs}")
        print(f"Batch size: {batch_size}")
        if self.use_class_weights:
            print("Class weights: ENABLED")
        
        callbacks = self.get_callbacks()
        
        self.history = self.model.fit(
            datagen.flow(self.X_train, self.y_train, batch_size=batch_size),
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            validation_data=(self.X_val, self.y_val),
            callbacks=callbacks,
            class_weight=self.class_weights,
            verbose=1
        )
        
        print("\nTraining completed!")
    
    def _train_transfer_learning(self, datagen, batch_size, steps_per_epoch):
        """
        Two-stage transfer learning training.
        
        Stage 1: Train only the classification head (base model frozen)
        Stage 2: Unfreeze top layers and fine-tune with smaller learning rate
        """
        config = MODEL_CONFIG[self.model_type]
        stage1_epochs = config['stage1_epochs']
        stage2_epochs = config['stage2_epochs']
        
        # ── STAGE 1: Train classification head only ──
        print("\n" + "="*60)
        print(f"STAGE 1: Training classification head ({stage1_epochs} epochs)")
        print("="*60)
        print(f"  Base model: FROZEN")
        print(f"  Learning rate: {config['base_learning_rate']}")
        print(f"  Batch size: {batch_size}")
        if self.use_class_weights:
            print("  Class weights: ENABLED")
        
        callbacks_s1 = self.get_callbacks(stage='stage1')
        
        history_s1 = self.model.fit(
            datagen.flow(self.X_train, self.y_train, batch_size=batch_size),
            steps_per_epoch=steps_per_epoch,
            epochs=stage1_epochs,
            validation_data=(self.X_val, self.y_val),
            callbacks=callbacks_s1,
            class_weight=self.class_weights,
            verbose=1
        )
        
        s1_val_acc = max(history_s1.history.get('val_accuracy', [0]))
        print(f"\nStage 1 complete! Best val accuracy: {s1_val_acc*100:.2f}%")
        
        # ── STAGE 2: Fine-tune top layers ──
        print("\n" + "="*60)
        print(f"STAGE 2: Fine-tuning top layers ({stage2_epochs} epochs)")
        print("="*60)
        
        # Unfreeze the base model for fine-tuning
        if self.model_type != 'vit':
            base_model = self.model.layers[1]  # The backbone layer
            base_model.trainable = True
            
            # Freeze bottom layers, keep top layers trainable
            fine_tune_at = config['fine_tune_at_layer']
            for layer in base_model.layers[:fine_tune_at]:
                layer.trainable = False
            
            trainable = sum(1 for l in base_model.layers if l.trainable)
            frozen = sum(1 for l in base_model.layers if not l.trainable)
            print(f"  Base model: {trainable} trainable layers, {frozen} frozen layers")
        else:
            # ViT: unfreeze the hub layer (it was frozen in Stage 1)
            vit_layer = self.model.layers[1]  # The hub KerasLayer
            vit_layer.trainable = True
            print(f"  ViT: Hub layer unfrozen for fine-tuning")
        
        # Recompile with smaller learning rate
        fine_tune_lr = config['fine_tune_learning_rate']
        print(f"  Learning rate: {fine_tune_lr}")
        
        self.model.compile(
            optimizer=Adam(learning_rate=fine_tune_lr),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        callbacks_s2 = self.get_callbacks(stage='stage2')
        
        history_s2 = self.model.fit(
            datagen.flow(self.X_train, self.y_train, batch_size=batch_size),
            steps_per_epoch=steps_per_epoch,
            epochs=stage2_epochs,
            validation_data=(self.X_val, self.y_val),
            callbacks=callbacks_s2,
            class_weight=self.class_weights,
            verbose=1
        )
        
        # Merge both histories for plotting
        merged_history = {}
        for key in history_s1.history:
            merged_history[key] = history_s1.history[key] + history_s2.history.get(key, [])
        
        # Create a simple namespace to hold merged history
        class MergedHistory:
            def __init__(self, h):
                self.history = h
        
        self.history = MergedHistory(merged_history)
        
        s2_val_acc = max(history_s2.history.get('val_accuracy', [0]))
        print(f"\nStage 2 complete! Best val accuracy: {s2_val_acc*100:.2f}%")
        print(f"Improvement: {s1_val_acc*100:.2f}% → {s2_val_acc*100:.2f}%")
    
    def evaluate(self):
        """Evaluate model on test set."""
        print("\n" + "="*60)
        print("EVALUATING MODEL")
        print("="*60)
        
        test_loss, test_accuracy = self.model.evaluate(
            self.X_test, 
            self.y_test, 
            verbose=0
        )
        
        print(f"\nTest Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_accuracy*100:.2f}%")
        
        # Save metrics
        metrics = {
            'test_loss': float(test_loss),
            'test_accuracy': float(test_accuracy),
            'model_type': self.model_type,
            'class_weights_used': self.use_class_weights,
        }
        
        if self.history and hasattr(self.history, 'history'):
            metrics['max_val_accuracy'] = float(max(self.history.history.get('val_accuracy', [0])))
            metrics['max_train_accuracy'] = float(max(self.history.history.get('accuracy', [0])))
        
        metrics_path = os.path.join(self.run_dir, 'metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"\nMetrics saved to {metrics_path}")
        
        return test_loss, test_accuracy
    
    def plot_history(self):
        """Plot and save training history."""
        if not self.history or not hasattr(self.history, 'history'):
            print("No training history to plot.")
            return
        
        print("\nPlotting training history...")
        
        acc = self.history.history.get('accuracy', [])
        val_acc = self.history.history.get('val_accuracy', [])
        loss = self.history.history.get('loss', [])
        val_loss = self.history.history.get('val_loss', [])
        
        if not acc:
            print("No accuracy data in history.")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Accuracy plot
        ax1.plot(range(len(acc)), acc, label='Training Accuracy', linewidth=2)
        ax1.plot(range(len(val_acc)), val_acc, label='Validation Accuracy', linewidth=2)
        
        # Mark stage boundary for transfer learning
        if self.model_type in TRANSFER_LEARNING_MODELS:
            stage1_epochs = MODEL_CONFIG[self.model_type]['stage1_epochs']
            if stage1_epochs < len(acc):
                ax1.axvline(x=stage1_epochs, color='red', linestyle='--', 
                          alpha=0.7, label='Stage 1→2 boundary')
                ax2.axvline(x=stage1_epochs, color='red', linestyle='--',
                          alpha=0.7, label='Stage 1→2 boundary')
        
        ax1.set_title(f'{self.model_type.upper()} - Training vs Validation Accuracy', 
                     fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Loss plot
        ax2.plot(range(len(loss)), loss, label='Training Loss', linewidth=2)
        ax2.plot(range(len(val_loss)), val_loss, label='Validation Loss', linewidth=2)
        ax2.set_title(f'{self.model_type.upper()} - Training vs Validation Loss', 
                     fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        plot_path = os.path.join(self.run_dir, 'training_history.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Training history plot saved to {plot_path}")
    
    def save_final_model(self):
        """Save the final trained model."""
        model_path = os.path.join(PATHS['models_dir'], f'{self.model_type}_final.h5')
        self.model.save(model_path)
        print(f"\nFinal model saved to {model_path}")
    
    def run_full_training(self, epochs: int = None, batch_size: int = None):
        """
        Run complete training pipeline.
        
        Args:
            epochs: Number of epochs
            batch_size: Batch size
        """
        self.load_data()
        self.build_model()
        
        # Free test data during training to save memory (~0.6 GB for 224×224)
        # Will reload from disk for evaluation afterwards.
        X_test_backup = None
        y_test_backup = None
        if self.model_type in TRANSFER_LEARNING_MODELS:
            X_test_backup = self.X_test
            y_test_backup = self.y_test
            del self.X_test, self.y_test
            gc.collect()
            print("Freed test data to save memory during training")
        
        self.train(epochs, batch_size)
        
        # Restore test data
        if X_test_backup is not None:
            self.X_test = X_test_backup
            self.y_test = y_test_backup
            del X_test_backup, y_test_backup
        
        # Reload the best checkpoint so evaluate() uses the true best model
        # (EarlyStopping may have restored a Stage 2 best that is worse
        #  than an earlier checkpoint saved by ModelCheckpoint)
        best_checkpoint = os.path.join(
            PATHS['checkpoints_dir'], f'{self.model_type}_best.h5'
        )
        if os.path.exists(best_checkpoint):
            from tensorflow.keras.models import load_model
            print(f"\nReloading best checkpoint from {best_checkpoint}")
            self.model = load_model(best_checkpoint)
        
        self.evaluate()
        self.plot_history()
        self.save_final_model()
        self._save_model_metadata()
        
        print("\n" + "="*60)
        print("TRAINING COMPLETE")
        print("="*60)
        print(f"All outputs saved to: {self.run_dir}")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(
        description='Train skin cancer detection model'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='sequential',
        choices=['sequential', 'resnet', 'efficientnet', 'resnet50', 'densenet', 'vit'],
        help='Model architecture to train'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='Number of training epochs (for from-scratch models)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='Batch size for training'
    )
    parser.add_argument(
        '--class-weights',
        action='store_true',
        default=False,
        help='Use class weights to handle class imbalance'
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("SKIN CANCER DETECTION MODEL TRAINING")
    print("="*60)
    print(f"Model: {args.model}")
    if args.class_weights:
        print("Class weights: ENABLED")
    
    trainer = ModelTrainer(
        model_type=args.model,
        use_class_weights=args.class_weights
    )
    trainer.run_full_training(epochs=args.epochs, batch_size=args.batch_size)


if __name__ == '__main__':
    main()

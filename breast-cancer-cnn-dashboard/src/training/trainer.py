"""
Training module for Breast Cancer CNN Classifier.
Implements training loop with callbacks, early stopping, learning rate scheduling, and mixed precision.
"""

import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from collections import defaultdict
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, OneCycleLR

import numpy as np
from tqdm import tqdm

from src.utils.config import config
from src.utils.logging_utils import get_logger, TrainingLogger
from src.utils.reproducibility import set_seed
from src.utils.helpers import (
    AverageMeter, ProgressMeter, save_checkpoint, load_checkpoint,
    ensure_dir, save_json
)

logger = get_logger(__name__)


@dataclass
class TrainingState:
    """Training state container."""
    epoch: int = 0
    best_metric: float = float('inf')
    best_epoch: int = 0
    patience_counter: int = 0
    history: Dict[str, List[float]] = field(default_factory=lambda: defaultdict(list))
    stopped: bool = False


class EarlyStopping:
    """Early stopping callback with patience and metric monitoring."""
    
    def __init__(
        self,
        patience: int = 15,
        min_delta: float = 1e-4,
        monitor: str = 'val_loss',
        mode: str = 'min',
        restore_best_weights: bool = True,
        verbose: bool = True
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        self.verbose = verbose
        
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        self.stopped = False
        
        # Determine comparison function
        if mode == 'min':
            self.is_better = lambda current, best: current < best - min_delta
        else:
            self.is_better = lambda current, best: current > best + min_delta
    
    def __call__(self, metric_value: float, model: nn.Module, epoch: int) -> bool:
        """
        Check if training should stop.
        
        Returns:
            True if training should stop
        """
        if self.best_score is None:
            self.best_score = metric_value
            self.best_weights = model.state_dict().copy()
            return False
        
        if self.is_better(metric_value, self.best_score):
            self.best_score = metric_value
            self.counter = 0
            self.best_weights = model.state_dict().copy()
            if self.verbose:
                logger.info(f"  EarlyStopping: New best {self.monitor} = {metric_value:.6f} at epoch {epoch}")
        else:
            self.counter += 1
            if self.verbose:
                logger.info(f"  EarlyStopping: No improvement for {self.counter}/{self.patience} epochs")
            
            if self.counter >= self.patience:
                self.stopped = True
                if self.verbose:
                    logger.warning(f"Early stopping triggered after {epoch} epochs")
                
                if self.restore_best_weights and self.best_weights is not None:
                    model.load_state_dict(self.best_weights)
                    if self.verbose:
                        logger.info("Restored best model weights")
        
        return self.stopped
    
    def reset(self):
        """Reset early stopping state."""
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        self.stopped = False


class ModelCheckpoint:
    """Model checkpoint callback."""
    
    def __init__(
        self,
        filepath: str,
        monitor: str = 'val_loss',
        mode: str = 'min',
        save_best_only: bool = True,
        save_last: bool = True,
        verbose: bool = True
    ):
        self.filepath = Path(filepath)
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.save_last = save_last
        self.verbose = verbose
        
        self.best_score = None
        self.best_epoch = 0
        
        ensure_dir(self.filepath.parent)
        
        if mode == 'min':
            self.is_better = lambda current, best: current < best
        else:
            self.is_better = lambda current, best: current > best
    
    def __call__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: Any,
        epoch: int,
        metrics: Dict[str, float],
        is_last: bool = False
    ):
        """Save checkpoint if conditions met."""
        current_score = metrics.get(self.monitor)
        
        if current_score is None:
            if self.verbose:
                logger.warning(f"Monitor metric '{self.monitor}' not found in metrics")
            return
        
        # Save best model
        if self.save_best_only:
            if self.best_score is None or self.is_better(current_score, self.best_score):
                self.best_score = current_score
                self.best_epoch = epoch
                
                checkpoint_path = self.filepath.parent / "best_model.pth"
                save_checkpoint(
                    model, optimizer, scheduler, epoch, metrics,
                    checkpoint_path, is_best=True
                )
                
                if self.verbose:
                    logger.info(f"  Checkpoint: Saved best model at epoch {epoch} ({self.monitor}={current_score:.6f})")
        
        # Save last model
        if is_last and self.save_last:
            checkpoint_path = self.filepath.parent / "last_model.pth"
            save_checkpoint(
                model, optimizer, scheduler, epoch, metrics,
                checkpoint_path, is_best=False
            )
            
            if self.verbose:
                logger.info(f"  Checkpoint: Saved last model at epoch {epoch}")


class LearningRateScheduler:
    """Learning rate scheduler wrapper."""
    
    def __init__(
        self,
        optimizer: optim.Optimizer,
        scheduler_type: str = 'ReduceLROnPlateau',
        **kwargs
    ):
        self.scheduler_type = scheduler_type
        
        if scheduler_type == 'ReduceLROnPlateau':
            self.scheduler = ReduceLROnPlateau(
                optimizer,
                mode=kwargs.get('mode', 'min'),
                factor=kwargs.get('factor', 0.5),
                patience=kwargs.get('patience', 10),
                min_lr=kwargs.get('min_lr', 1e-6),
                verbose=kwargs.get('verbose', True)
            )
        elif scheduler_type == 'CosineAnnealingLR':
            self.scheduler = CosineAnnealingLR(
                optimizer,
                T_max=kwargs.get('T_max', 100),
                eta_min=kwargs.get('eta_min', 1e-6)
            )
        elif scheduler_type == 'OneCycleLR':
            self.scheduler = OneCycleLR(
                optimizer,
                max_lr=kwargs.get('max_lr', 0.01),
                epochs=kwargs.get('epochs', 100),
                steps_per_epoch=kwargs.get('steps_per_epoch', 100)
            )
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")
    
    def step(self, metric: float = None):
        """Step the scheduler."""
        if self.scheduler_type == 'ReduceLROnPlateau':
            if metric is not None:
                self.scheduler.step(metric)
        else:
            self.scheduler.step()
    
    def get_last_lr(self) -> List[float]:
        """Get current learning rates."""
        return self.scheduler.get_last_lr()


class Trainer:
    """
    Main training class with full training loop, callbacks, and logging.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        device: torch.device,
        scheduler: Optional[LearningRateScheduler] = None,
        early_stopping: Optional[EarlyStopping] = None,
        checkpoint: Optional[ModelCheckpoint] = None,
        mixed_precision: bool = True,
        gradient_accumulation_steps: int = 1,
        log_interval: int = 10
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler
        self.early_stopping = early_stopping
        self.checkpoint = checkpoint
        self.mixed_precision = mixed_precision
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.log_interval = log_interval
        
        # Mixed precision scaler
        self.scaler = GradScaler() if mixed_precision and device.type == 'cuda' else None
        
        # Training state
        self.state = TrainingState()
        self.train_logger = TrainingLogger(logger, log_interval)
        
        # Move model to device
        self.model.to(device)
        
        logger.info(f"Trainer initialized on {device}")
        logger.info(f"Mixed precision: {mixed_precision and device.type == 'cuda'}")
        logger.info(f"Gradient accumulation steps: {gradient_accumulation_steps}")
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        losses = AverageMeter('Loss', ':.4f')
        accuracies = AverageMeter('Acc', ':6.2f')
        
        progress = ProgressMeter(
            len(self.train_loader),
            [losses, accuracies],
            prefix=f"Epoch [{self.state.epoch}] "
        )
        
        self.optimizer.zero_grad()
        
        for i, (images, targets) in enumerate(self.train_loader):
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            
            # Forward pass with mixed precision
            if self.scaler is not None:
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, targets)
                
                # Scale loss and backpropagate
                self.scaler.scale(loss).backward()
                
                # Gradient accumulation
                if (i + 1) % self.gradient_accumulation_steps == 0:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)
                
                loss.backward()
                
                if (i + 1) % self.gradient_accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            
            # Compute accuracy
            _, predicted = outputs.max(1)
            correct = predicted.eq(targets).sum().item()
            accuracy = 100. * correct / targets.size(0)
            
            losses.update(loss.item(), images.size(0))
            accuracies.update(accuracy, images.size(0))
            
            # Log progress
            if i % self.log_interval == 0:
                progress.display(i)
        
        return {'loss': losses.avg, 'accuracy': accuracies.avg}
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate model on validation set."""
        self.model.eval()
        
        losses = AverageMeter('Loss', ':.4f')
        accuracies = AverageMeter('Acc', ':6.2f')
        
        all_preds = []
        all_targets = []
        all_probs = []
        
        for images, targets in tqdm(self.val_loader, desc="Validating", leave=False):
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            
            if self.scaler is not None:
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, targets)
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)
            
            # Compute accuracy
            _, predicted = outputs.max(1)
            correct = predicted.eq(targets).sum().item()
            accuracy = 100. * correct / targets.size(0)
            
            losses.update(loss.item(), images.size(0))
            accuracies.update(accuracy, images.size(0))
            
            # Store predictions for metrics
            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            all_probs.extend(probs)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
        
        return {
            'loss': losses.avg,
            'accuracy': accuracies.avg,
            'predictions': np.array(all_preds),
            'targets': np.array(all_targets),
            'probabilities': np.array(all_probs)
        }
    
    def fit(
        self,
        epochs: int,
        start_epoch: int = 0
    ) -> Dict[str, List[float]]:
        """
        Full training loop.
        
        Args:
            epochs: Number of epochs to train
            start_epoch: Starting epoch (for resuming)
            
        Returns:
            Training history
        """
        logger.info(f"Starting training for {epochs} epochs")
        start_time = time.time()
        
        for epoch in range(start_epoch, epochs):
            self.state.epoch = epoch
            epoch_start = time.time()
            
            # Train
            self.train_logger.log_epoch_start(epoch + 1, epochs)
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Log epoch results
            self.train_logger.log_epoch_end(train_metrics, {
                'loss': val_metrics['loss'],
                'accuracy': val_metrics['accuracy']
            })
            
            # Update history
            for key, value in train_metrics.items():
                self.state.history[f'train_{key}'].append(value)
            for key, value in val_metrics.items():
                if key not in ['predictions', 'targets', 'probabilities']:
                    self.state.history[f'val_{key}'].append(value)
            
            # Learning rate scheduling
            if self.scheduler:
                if isinstance(self.scheduler.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['loss'])
                else:
                    self.scheduler.step()
                
                current_lr = self.scheduler.get_last_lr()[0]
                logger.info(f"  Learning rate: {current_lr:.2e}")
            
            # Checkpoint
            if self.checkpoint:
                is_last = (epoch == epochs - 1)
                self.checkpoint(
                    self.model, self.optimizer, 
                    self.scheduler.scheduler if self.scheduler else None,
                    epoch, {**train_metrics, **{f'val_{k}': v for k, v in val_metrics.items() if k not in ['predictions', 'targets', 'probabilities']}},
                    is_last
                )
            
            # Early stopping
            if self.early_stopping:
                if self.early_stopping(val_metrics['loss'], self.model, epoch):
                    self.state.stopped = True
                    break
            
            epoch_time = time.time() - epoch_start
            logger.info(f"  Epoch time: {epoch_time:.1f}s")
        
        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time/60:.1f} minutes")
        logger.info(f"Best validation loss: {self.early_stopping.best_score:.6f} at epoch {self.early_stopping.best_epoch + 1}" if self.early_stopping else "")
        
        return dict(self.state.history)
    
    def get_training_history(self) -> Dict[str, List[float]]:
        """Get training history."""
        return dict(self.state.history)


def create_optimizer(model: nn.Module, config_dict: Dict[str, Any] = None) -> optim.Optimizer:
    """Create optimizer from configuration."""
    config_dict = config_dict or config.training
    
    optimizer_name = config_dict.get('optimizer', 'AdamW')
    lr = config_dict.get('learning_rate', 0.001)
    weight_decay = config_dict.get('weight_decay', 1e-4)
    
    # Filter only trainable parameters
    params = filter(lambda p: p.requires_grad, model.parameters())
    
    if optimizer_name == 'AdamW':
        optimizer = optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'Adam':
        optimizer = optim.Adam(params, lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'SGD':
        momentum = config_dict.get('momentum', 0.9)
        optimizer = optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    logger.info(f"Created optimizer: {optimizer_name} (lr={lr}, weight_decay={weight_decay})")
    return optimizer


def create_scheduler(optimizer: optim.Optimizer, config_dict: Dict[str, Any] = None) -> LearningRateScheduler:
    """Create learning rate scheduler from configuration."""
    config_dict = config_dict or config.training
    
    scheduler_type = config_dict.get('scheduler', 'ReduceLROnPlateau')
    scheduler_params = config_dict.get('scheduler_params', {}).get(scheduler_type, {})
    
    scheduler = LearningRateScheduler(optimizer, scheduler_type, **scheduler_params)
    logger.info(f"Created scheduler: {scheduler_type}")
    return scheduler


def create_criterion(class_weights: Optional[torch.Tensor] = None) -> nn.Module:
    """Create loss criterion."""
    loss_name = config.training.get('loss_function', 'CrossEntropyLoss')
    
    if loss_name == 'CrossEntropyLoss':
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    elif loss_name == 'LabelSmoothingCrossEntropy':
        # Custom label smoothing
        class LabelSmoothingCrossEntropy(nn.Module):
            def __init__(self, smoothing=0.1):
                super().__init__()
                self.smoothing = smoothing
            
            def forward(self, x, target):
                confidence = 1. - self.smoothing
                logprobs = F.log_softmax(x, dim=-1)
                nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)
                smooth_loss = -logprobs.mean(dim=-1)
                loss = confidence * nll_loss + self.smoothing * smooth_loss
                return loss.mean()
        
        criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
    else:
        raise ValueError(f"Unknown loss function: {loss_name}")
    
    logger.info(f"Created criterion: {loss_name}")
    return criterion


def create_trainer(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    class_weights: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = None,
    **kwargs
) -> Trainer:
    """
    Factory function to create Trainer with full configuration.
    
    Args:
        model: PyTorch model
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        class_weights: Class weights for loss function
        device: Training device
        **kwargs: Additional trainer parameters
        
    Returns:
        Configured Trainer instance
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create criterion
    criterion = create_criterion(class_weights)
    criterion = criterion.to(device)
    
    # Create optimizer
    optimizer = create_optimizer(model)
    
    # Create scheduler
    scheduler = create_scheduler(optimizer)
    
    # Create early stopping
    early_stopping_config = config.training.get('early_stopping', {})
    if early_stopping_config.get('enabled', True):
        early_stopping = EarlyStopping(
            patience=early_stopping_config.get('patience', 15),
            min_delta=early_stopping_config.get('min_delta', 1e-4),
            monitor=early_stopping_config.get('monitor', 'val_loss'),
            mode=early_stopping_config.get('mode', 'min'),
            restore_best_weights=early_stopping_config.get('restore_best_weights', True)
        )
    else:
        early_stopping = None
    
    # Create checkpoint
    checkpoint_config = config.training
    checkpoint_dir = checkpoint_config.get('checkpoint_dir', 'assets/models/checkpoints')
    checkpoint = ModelCheckpoint(
        filepath=str(Path(checkpoint_dir) / "checkpoint.pth"),
        monitor=early_stopping_config.get('monitor', 'val_loss') if early_stopping_config.get('enabled') else 'val_loss',
        mode=early_stopping_config.get('mode', 'min') if early_stopping_config.get('enabled') else 'min',
        save_best_only=checkpoint_config.get('save_best_only', True),
        save_last=checkpoint_config.get('save_last', True)
    )
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        scheduler=scheduler,
        early_stopping=early_stopping,
        checkpoint=checkpoint,
        mixed_precision=checkpoint_config.get('mixed_precision', True),
        gradient_accumulation_steps=checkpoint_config.get('gradient_accumulation_steps', 1),
        **kwargs
    )
    
    return trainer


if __name__ == "__main__":
    # Example usage
    from src.models import create_model
    from src.data import create_preprocessor
    
    # Create model
    model = create_model('CustomCNN')
    
    # Create data
    preprocessor = create_preprocessor()
    dataloaders = preprocessor.run_full_pipeline()
    
    # Create trainer
    trainer = create_trainer(
        model,
        dataloaders['train'],
        dataloaders['val'],
        preprocessor.class_weights
    )
    
    # Train
    history = trainer.fit(epochs=5)
    print("Training history:", history)
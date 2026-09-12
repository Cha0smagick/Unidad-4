"""
Utility functions for the Breast Cancer CNN Classifier project.
"""

import os
import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
import torch
from PIL import Image


def ensure_dir(path: Union[str, Path]) -> Path:
    """Ensure directory exists, create if necessary."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data: Any, path: Union[str, Path], indent: int = 2) -> None:
    """Save data as JSON file."""
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def load_json(path: Union[str, Path]) -> Any:
    """Load data from JSON file."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_pickle(data: Any, path: Union[str, Path]) -> None:
    """Save data as pickle file."""
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(path: Union[str, Path]) -> Any:
    """Load data from pickle file."""
    with open(path, 'rb') as f:
        return pickle.load(f)


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """
    Count model parameters.
    
    Returns:
        Dictionary with total, trainable, and non-trainable parameter counts
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable = total - trainable
    
    return {
        'total': total,
        'trainable': trainable,
        'non_trainable': non_trainable
    }


def format_number(num: int) -> str:
    """Format large numbers with commas."""
    return f"{num:,}"


def get_model_summary(model: torch.nn.Module, input_size: tuple = (1, 3, 50, 50)) -> str:
    """
    Generate model summary string.
    
    Args:
        model: PyTorch model
        input_size: Input tensor size (batch, channels, height, width)
        
    Returns:
        Formatted model summary string
    """
    param_counts = count_parameters(model)
    
    summary = []
    summary.append("=" * 60)
    summary.append("MODEL SUMMARY")
    summary.append("=" * 60)
    summary.append(f"Total Parameters: {format_number(param_counts['total'])}")
    summary.append(f"Trainable Parameters: {format_number(param_counts['trainable'])}")
    summary.append(f"Non-trainable Parameters: {format_number(param_counts['non_trainable'])}")
    summary.append("-" * 60)
    
    # Model architecture
    summary.append(str(model))
    summary.append("=" * 60)
    
    return "\n".join(summary)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    epoch: int,
    metrics: Dict[str, float],
    path: Union[str, Path],
    is_best: bool = False
) -> None:
    """
    Save model checkpoint.
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        scheduler: Learning rate scheduler (optional)
        epoch: Current epoch
        metrics: Dictionary of metrics
        path: Save path
        is_best: Whether this is the best model
    """
    path = Path(path)
    ensure_dir(path.parent)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'metrics': metrics,
        'is_best': is_best
    }
    
    torch.save(checkpoint, path)
    
    if is_best:
        best_path = path.parent / "best_model.pth"
        torch.save(checkpoint, best_path)


def load_checkpoint(
    path: Union[str, Path],
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    """
    Load model checkpoint.
    
    Args:
        path: Checkpoint path
        model: PyTorch model
        optimizer: Optimizer (optional)
        scheduler: Learning rate scheduler (optional)
        device: Target device
        
    Returns:
        Dictionary with checkpoint info
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    checkpoint = torch.load(path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return {
        'epoch': checkpoint.get('epoch', 0),
        'metrics': checkpoint.get('metrics', {}),
        'is_best': checkpoint.get('is_best', False)
    }


def compute_class_weights(labels: np.ndarray, num_classes: int = 2) -> torch.Tensor:
    """
    Compute class weights for imbalanced datasets.
    
    Args:
        labels: Array of class labels
        num_classes: Number of classes
        
    Returns:
        Tensor of class weights
    """
    from sklearn.utils.class_weight import compute_class_weight
    
    classes = np.arange(num_classes)
    weights = compute_class_weight('balanced', classes=classes, y=labels)
    return torch.FloatTensor(weights)


def get_file_list(directory: Union[str, Path], extensions: List[str] = None) -> List[Path]:
    """
    Get list of files in directory with optional extension filter.
    
    Args:
        directory: Directory to search
        extensions: List of file extensions (e.g., ['.png', '.jpg'])
        
    Returns:
        List of file paths
    """
    directory = Path(directory)
    if not directory.exists():
        return []
    
    if extensions is None:
        return list(directory.rglob('*'))
    
    files = []
    for ext in extensions:
        files.extend(directory.rglob(f'*{ext}'))
    return files


def validate_image_file(filepath: Union[str, Path]) -> bool:
    """Validate that a file is a readable image."""
    try:
        with Image.open(filepath) as img:
            img.verify()
        return True
    except Exception:
        return False


def create_experiment_dir(base_dir: Union[str, Path], experiment_name: str) -> Path:
    """
    Create timestamped experiment directory.
    
    Args:
        base_dir: Base directory for experiments
        experiment_name: Name of the experiment
        
    Returns:
        Path to created experiment directory
    """
    from datetime import datetime
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = Path(base_dir) / f"{experiment_name}_{timestamp}"
    ensure_dir(exp_dir)
    return exp_dir


class AverageMeter:
    """Computes and stores average and current value."""
    
    def __init__(self, name: str = "", fmt: str = ":.4f"):
        self.name = name
        self.fmt = fmt
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def __str__(self):
        return f"{self.name} {self.val{self.fmt}} ({self.avg{self.fmt}})"


class ProgressMeter:
    """Displays progress with multiple meters."""
    
    def __init__(self, num_batches: int, meters: List[AverageMeter], prefix: str = ""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
    
    def _get_batch_fmtstr(self, num_batches: int) -> str:
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'
    
    def display(self, batch: int):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))
    
    def get_str(self, batch: int) -> str:
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        return '\t'.join(entries)
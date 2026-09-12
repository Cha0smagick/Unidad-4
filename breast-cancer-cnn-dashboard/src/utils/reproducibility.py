"""
Reproducibility utilities for setting random seeds and ensuring deterministic behavior.
"""

import random
import os
import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = True, benchmark: bool = False) -> None:
    """
    Set random seeds for reproducibility across Python, NumPy, and PyTorch.
    
    Args:
        seed: Random seed value
        deterministic: Enable deterministic algorithms (may impact performance)
        benchmark: Enable cuDNN benchmarking (faster but non-deterministic)
    """
    # Python built-in
    random.seed(seed)
    
    # Environment variable for hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU
    
    # Deterministic operations
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Enable deterministic algorithms (PyTorch 1.8+)
        try:
            torch.use_deterministic_algorithms(True)
        except AttributeError:
            pass  # Older PyTorch versions
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = benchmark
    
    # Print confirmation
    print(f"Random seeds set to {seed}")
    print(f"Deterministic mode: {deterministic}")
    print(f"Benchmark mode: {benchmark}")


def get_device() -> torch.device:
    """
    Get the optimal device for training (CUDA, MPS, or CPU).
    
    Returns:
        torch.device: The selected device
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple MPS device")
    else:
        device = torch.device("cpu")
        print("Using CPU device")
    
    return device


def print_system_info() -> None:
    """Print system and environment information for reproducibility."""
    print("=" * 60)
    print("SYSTEM INFORMATION")
    print("=" * 60)
    print(f"PyTorch version: {torch.__version__}")
    print(f"NumPy version: {np.__version__}")
    print(f"Python version: {__import__('sys').version}")
    
    if torch.cuda.is_available():
        print(f"CUDA available: True")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"cuDNN version: {torch.backends.cudnn.version()}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {props.name} ({props.total_memory / 1e9:.1f} GB)")
    else:
        print("CUDA available: False")
    
    print("=" * 60)


class ReproducibilityContext:
    """Context manager for temporary seed setting."""
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.prev_states = {}
    
    def __enter__(self):
        # Save current states
        self.prev_states = {
            'python': random.getstate(),
            'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'torch_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }
        # Set new seed
        set_seed(self.seed)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore previous states
        random.setstate(self.prev_states['python'])
        np.random.set_state(self.prev_states['numpy'])
        torch.set_rng_state(self.prev_states['torch'])
        if self.prev_states['torch_cuda'] is not None:
            torch.cuda.set_rng_state_all(self.prev_states['torch_cuda'])
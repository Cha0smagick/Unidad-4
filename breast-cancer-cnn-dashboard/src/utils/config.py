"""
Configuration management module for Breast Cancer CNN Classifier.
Handles loading, validation, and access to configuration parameters.
"""

import yaml
from pathlib import Path
from typing import Any, Dict, Optional
from dataclasses import dataclass, field


@dataclass
class DataConfig:
    """Data configuration parameters."""
    raw_data_path: str
    processed_data_path: str
    image_size: list
    channels: int
    num_classes: int
    class_names: list
    train_split: float
    val_split: float
    test_split: float
    random_seed: int
    augmentation: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PreprocessingConfig:
    """Preprocessing configuration parameters."""
    normalize: Dict[str, list]
    batch_size: int
    num_workers: int
    pin_memory: bool
    persistent_workers: bool


@dataclass
class ModelConfig:
    """Model configuration parameters."""
    architecture: str
    custom_cnn: Dict[str, Any]
    transfer_learning: Dict[str, Any]


@dataclass
class TrainingConfig:
    """Training configuration parameters."""
    epochs: int
    learning_rate: float
    weight_decay: float
    optimizer: str
    scheduler: str
    scheduler_params: Dict[str, Any]
    early_stopping: Dict[str, Any]
    loss_function: str
    class_weights: Optional[list]
    mixed_precision: bool
    gradient_accumulation_steps: int
    checkpoint_dir: str
    save_best_only: bool
    save_last: bool


@dataclass
class EvaluationConfig:
    """Evaluation configuration parameters."""
    metrics: list
    threshold: float
    bootstrap_ci: Dict[str, Any]


@dataclass
class DashboardConfig:
    """Dashboard configuration parameters."""
    host: str
    port: int
    theme: Dict[str, str]
    plots: Dict[str, Any]


@dataclass
class LoggingConfig:
    """Logging configuration parameters."""
    mlflow: Dict[str, Any]
    tensorboard: Dict[str, Any]
    console: Dict[str, Any]


@dataclass
class ReproducibilityConfig:
    """Reproducibility configuration parameters."""
    seeds: Dict[str, int]
    deterministic: bool
    benchmark: bool


class Config:
    """Main configuration class for the project."""
    
    _instance: Optional['Config'] = None
    _config_data: Dict[str, Any] = {}
    
    def __new__(cls, config_path: Optional[str] = None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            if config_path:
                cls._instance.load(config_path)
        return cls._instance
    
    def load(self, config_path: str) -> None:
        """Load configuration from YAML file."""
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(path, 'r', encoding='utf-8') as f:
            self._config_data = yaml.safe_load(f)
        
        # Parse into typed config objects
        self.data = DataConfig(**self._config_data.get('data', {}))
        self.preprocessing = PreprocessingConfig(**self._config_data.get('preprocessing', {}))
        self.model = ModelConfig(**self._config_data.get('model', {}))
        self.training = TrainingConfig(**self._config_data.get('training', {}))
        self.evaluation = EvaluationConfig(**self._config_data.get('evaluation', {}))
        self.dashboard = DashboardConfig(**self._config_data.get('dashboard', {}))
        self.logging = LoggingConfig(**self._config_data.get('logging', {}))
        self.reproducibility = ReproducibilityConfig(**self._config_data.get('reproducibility', {}))
        
        # Project metadata
        self.project = self._config_data.get('project', {})
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by dot-notation key."""
        keys = key.split('.')
        value = self._config_data
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
            if value is None:
                return default
        return value
    
    def update(self, key: str, value: Any) -> None:
        """Update configuration value."""
        keys = key.split('.')
        config = self._config_data
        for k in keys[:-1]:
            config = config.setdefault(k, {})
        config[keys[-1]] = value
    
    def save(self, config_path: str) -> None:
        """Save configuration to YAML file."""
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(self._config_data, f, default_flow_style=False, sort_keys=False)
    
    @property
    def raw_config(self) -> Dict[str, Any]:
        """Return raw configuration dictionary."""
        return self._config_data.copy()


def load_config(config_path: Optional[str] = None) -> Config:
    """Factory function to load configuration."""
    if config_path is None:
        config_path = Path(__file__).parent.parent.parent / "configs" / "config.yaml"
    return Config(str(config_path))


# Global config instance
config = load_config()
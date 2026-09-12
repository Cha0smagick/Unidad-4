"""
Model architectures for Breast Cancer CNN Classifier.
Implements custom CNN with BatchNorm, Dropout, and transfer learning options.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Dict, Any
import timm
from torchvision import models

from src.utils.config import config
from src.utils.logging_utils import get_logger
from src.utils.helpers import count_parameters, get_model_summary

logger = get_logger(__name__)


class ConvBlock(nn.Module):
    """Convolutional block with BatchNorm, ReLU, and optional Dropout."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        use_batch_norm: bool = True,
        dropout_rate: float = 0.0
    ):
        super().__init__()
        
        layers = []
        
        # Convolution
        layers.append(nn.Conv2d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=not use_batch_norm
        ))
        
        # Batch Normalization
        if use_batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        
        # Activation
        layers.append(nn.ReLU(inplace=True))
        
        # Dropout
        if dropout_rate > 0:
            layers.append(nn.Dropout2d(dropout_rate))
        
        self.block = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CustomCNN(nn.Module):
    """
    Custom CNN architecture for breast cancer histopathology classification.
    
    Features:
    - Configurable depth and filter progression
    - Batch Normalization after each conv layer
    - Dropout for regularization
    - Global Average Pooling before classifier
    - Fully connected layers with Dropout
    """
    
    def __init__(
        self,
        input_channels: int = 3,
        num_classes: int = 2,
        base_filters: int = 32,
        filter_progression: List[int] = None,
        kernel_size: int = 3,
        pool_size: int = 2,
        fc_hidden: List[int] = None,
        dropout_rate: float = 0.5,
        use_batch_norm: bool = True,
        use_global_pool: bool = True
    ):
        super().__init__()
        
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.use_global_pool = use_global_pool
        
        # Default filter progression
        if filter_progression is None:
            filter_progression = [base_filters * (2 ** i) for i in range(4)]
        
        # Default FC hidden layers
        if fc_hidden is None:
            fc_hidden = [512, 256]
        
        self.filter_progression = filter_progression
        self.fc_hidden = fc_hidden
        
        # Build convolutional backbone
        self.conv_layers = nn.ModuleList()
        in_channels = input_channels
        
        for i, out_channels in enumerate(filter_progression):
            # Two conv blocks per stage
            self.conv_layers.append(ConvBlock(
                in_channels, out_channels,
                kernel_size=kernel_size,
                use_batch_norm=use_batch_norm,
                dropout_rate=dropout_rate * 0.5 if i > 1 else 0.0  # Less dropout in early layers
            ))
            self.conv_layers.append(ConvBlock(
                out_channels, out_channels,
                kernel_size=kernel_size,
                use_batch_norm=use_batch_norm,
                dropout_rate=dropout_rate * 0.5 if i > 1 else 0.0
            ))
            # Max pooling
            self.conv_layers.append(nn.MaxPool2d(pool_size, pool_size))
            in_channels = out_channels
        
        # Global pooling or adaptive pooling
        if use_global_pool:
            self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
            flatten_dim = filter_progression[-1]
        else:
            self.global_pool = nn.AdaptiveAvgPool2d((4, 4))
            flatten_dim = filter_progression[-1] * 16
        
        # Build classifier head
        classifier_layers = []
        prev_dim = flatten_dim
        
        for hidden_dim in fc_hidden:
            classifier_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim) if use_batch_norm else nn.Identity(),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # Final classification layer
        classifier_layers.append(nn.Linear(prev_dim, num_classes))
        
        self.classifier = nn.Sequential(*classifier_layers)
        
        # Initialize weights
        self._initialize_weights()
        
        logger.info(f"CustomCNN initialized:")
        logger.info(f"  Filter progression: {filter_progression}")
        logger.info(f"  FC hidden: {fc_hidden}")
        logger.info(f"  Dropout rate: {dropout_rate}")
        logger.info(f"  BatchNorm: {use_batch_norm}")
        logger.info(f"  Parameters: {count_parameters(self)['trainable']:,}")
    
    def _initialize_weights(self):
        """Initialize model weights using He initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Convolutional backbone
        for layer in self.conv_layers:
            x = layer(x)
        
        # Global pooling
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        
        # Classifier
        x = self.classifier(x)
        
        return x
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features before classifier (for visualization/analysis)."""
        for layer in self.conv_layers:
            x = layer(x)
        x = self.global_pool(x)
        return torch.flatten(x, 1)


class TransferLearningModel(nn.Module):
    """
    Transfer learning wrapper for pretrained models.
    Supports ResNet, EfficientNet, ConvNeXt, and other timm models.
    """
    
    SUPPORTED_MODELS = {
        'resnet18': 'resnet18',
        'resnet34': 'resnet34',
        'resnet50': 'resnet50',
        'efficientnet_b0': 'efficientnet_b0',
        'efficientnet_b1': 'efficientnet_b1',
        'convnext_tiny': 'convnext_tiny',
        'convnext_small': 'convnext_small',
        'vit_tiny_patch16_224': 'vit_tiny_patch16_224',
        'vit_small_patch16_224': 'vit_small_patch16_224',
    }
    
    def __init__(
        self,
        model_name: str = 'resnet18',
        num_classes: int = 2,
        pretrained: bool = True,
        freeze_backbone: bool = True,
        unfreeze_layers: int = 0,
        dropout_rate: float = 0.5,
        fc_hidden: List[int] = None
    ):
        super().__init__()
        
        self.model_name = model_name
        self.num_classes = num_classes
        self.freeze_backbone = freeze_backbone
        self.unfreeze_layers = unfreeze_layers
        
        if fc_hidden is None:
            fc_hidden = [512, 256]
        
        # Load pretrained model
        if model_name in ['resnet18', 'resnet34', 'resnet50']:
            self.backbone = getattr(models, model_name)(pretrained=pretrained)
            # Get feature dimension
            if model_name == 'resnet18' or model_name == 'resnet34':
                feat_dim = 512
            else:
                feat_dim = 2048
            # Remove final FC layer
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
            
        elif 'efficientnet' in model_name or 'convnext' in model_name or 'vit' in model_name:
            # Use timm for modern architectures
            self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
            feat_dim = self.backbone.num_features
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            
            # Optionally unfreeze last N layers
            if unfreeze_layers > 0:
                self._unfreeze_last_layers(unfreeze_layers)
        
        # Build custom classifier
        classifier_layers = []
        prev_dim = feat_dim
        
        for hidden_dim in fc_hidden:
            classifier_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        classifier_layers.append(nn.Linear(prev_dim, num_classes))
        
        self.classifier = nn.Sequential(*classifier_layers)
        
        logger.info(f"TransferLearningModel initialized:")
        logger.info(f"  Backbone: {model_name}")
        logger.info(f"  Feature dim: {feat_dim}")
        logger.info(f"  Freeze backbone: {freeze_backbone}")
        logger.info(f"  Unfreeze layers: {unfreeze_layers}")
        logger.info(f"  Parameters: {count_parameters(self)['trainable']:,}")
    
    def _unfreeze_last_layers(self, n_layers: int):
        """Unfreeze last N layers of backbone."""
        # Get all parameters
        all_params = list(self.backbone.named_parameters())
        # Unfreeze last n_layers parameter groups
        for name, param in all_params[-n_layers:]:
            param.requires_grad = True
            logger.info(f"Unfrozen: {name}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extract features
        features = self.backbone(x)
        
        # Flatten if needed (for ResNet)
        if features.dim() > 2:
            features = torch.flatten(features, 1)
        
        # Classify
        return self.classifier(features)
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract backbone features."""
        features = self.backbone(x)
        if features.dim() > 2:
            features = torch.flatten(features, 1)
        return features


def create_model(
    architecture: str = None,
    **kwargs
) -> nn.Module:
    """
    Factory function to create model based on configuration.
    
    Args:
        architecture: Model architecture name
        **kwargs: Additional model parameters
        
    Returns:
        PyTorch model
    """
    architecture = architecture or config.model.architecture
    model_config = config.model
    
    if architecture == 'CustomCNN':
        cnn_config = model_config.custom_cnn
        model = CustomCNN(
            input_channels=cnn_config.get('input_channels', 3),
            num_classes=cnn_config.get('num_classes', 2),
            base_filters=cnn_config.get('base_filters', 32),
            filter_progression=cnn_config.get('filter_progression', [32, 64, 128, 256]),
            kernel_size=cnn_config.get('kernel_size', 3),
            pool_size=cnn_config.get('pool_size', 2),
            fc_hidden=cnn_config.get('fc_hidden', [512, 256]),
            dropout_rate=cnn_config.get('dropout_rate', 0.5),
            use_batch_norm=cnn_config.get('use_batch_norm', True)
        )
    
    elif architecture in TransferLearningModel.SUPPORTED_MODELS:
        tl_config = model_config.transfer_learning
        model = TransferLearningModel(
            model_name=architecture,
            num_classes=config.data.num_classes,
            pretrained=True,
            freeze_backbone=tl_config.get('freeze_backbone', True),
            unfreeze_layers=tl_config.get('unfreeze_layers', 0),
            dropout_rate=model_config.custom_cnn.get('dropout_rate', 0.5)
        )
    
    else:
        raise ValueError(f"Unknown architecture: {architecture}")
    
    logger.info(f"Created model: {architecture}")
    logger.info(f"Model summary:\n{get_model_summary(model)}")
    
    return model


class ModelEMA(nn.Module):
    """Exponential Moving Average of model weights for better generalization."""
    
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        super().__init__()
        self.decay = decay
        self.module = type(model)(**model.__dict__.get('_init_args', {}))
        self.module.load_state_dict(model.state_dict())
        self.module.eval()
        
        # Disable gradients for EMA model
        for param in self.module.parameters():
            param.requires_grad = False
    
    @torch.no_grad()
    def update(self, model: nn.Module):
        """Update EMA weights."""
        for ema_param, model_param in zip(self.module.parameters(), model.parameters()):
            ema_param.data.mul_(self.decay).add_(model_param.data, alpha=1 - self.decay)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.module(x)


if __name__ == "__main__":
    # Test model creation
    model = create_model('CustomCNN')
    
    # Test forward pass
    x = torch.randn(2, 3, 50, 50)
    out = model(x)
    print(f"Output shape: {out.shape}")
    print(f"Output: {out}")
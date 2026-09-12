"""
Unit tests for Breast Cancer CNN Classifier pipeline.
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.utils.config import load_config
from src.utils.reproducibility import set_seed, get_device
from src.models import create_model, CustomCNN
from src.data import BreastCancerDataset, get_train_transforms, get_val_transforms
from src.evaluation import ModelEvaluator, EvaluationResults
from src.training import EarlyStopping, ModelCheckpoint, AverageMeter


class TestConfig:
    """Test configuration loading."""
    
    def test_load_config(self):
        config = load_config()
        assert config is not None
        assert hasattr(config, 'data')
        assert hasattr(config, 'model')
        assert hasattr(config, 'training')
    
    def test_data_config(self):
        config = load_config()
        assert config.data.image_size == [50, 50]
        assert config.data.num_classes == 2
        assert config.data.class_names == ["Non-IDC", "IDC"]


class TestReproducibility:
    """Test reproducibility utilities."""
    
    def test_set_seed(self):
        set_seed(42)
        # Just verify it runs without error
        assert True
    
    def test_get_device(self):
        device = get_device()
        assert isinstance(device, torch.device)


class TestModels:
    """Test model architectures."""
    
    def test_custom_cnn_creation(self):
        model = CustomCNN(
            input_channels=3,
            num_classes=2,
            base_filters=16,  # Smaller for testing
            filter_progression=[16, 32, 64],
            fc_hidden=[64, 32]
        )
        assert isinstance(model, torch.nn.Module)
    
    def test_custom_cnn_forward(self):
        model = CustomCNN(
            input_channels=3,
            num_classes=2,
            base_filters=16,
            filter_progression=[16, 32],
            fc_hidden=[32]
        )
        x = torch.randn(2, 3, 50, 50)
        out = model(x)
        assert out.shape == (2, 2)
    
    def test_create_model_factory(self):
        model = create_model('CustomCNN')
        assert isinstance(model, torch.nn.Module)
    
    def test_model_parameters(self):
        model = CustomCNN(input_channels=3, num_classes=2, base_filters=16)
        from src.utils.helpers import count_parameters
        params = count_parameters(model)
        assert params['total'] > 0
        assert params['trainable'] > 0


class TestData:
    """Test data processing."""
    
    def test_transforms(self):
        train_transform = get_train_transforms((50, 50))
        val_transform = get_val_transforms((50, 50))
        
        # Create dummy image
        import cv2
        img = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        
        # Test transforms
        augmented = train_transform(image=img)
        assert 'image' in augmented
        assert augmented['image'].shape == (3, 50, 50)
        
        augmented = val_transform(image=img)
        assert 'image' in augmented
        assert augmented['image'].shape == (3, 50, 50)
    
    def test_dataset_creation(self):
        # Create dummy data
        image_paths = ['dummy1.png', 'dummy2.png']
        labels = [0, 1]
        
        transform = get_val_transforms((50, 50))
        dataset = BreastCancerDataset(image_paths, labels, transform=transform)
        
        assert len(dataset) == 2
        assert dataset.get_class_distribution() == {0: 1, 1: 1}


class TestEvaluation:
    """Test evaluation metrics."""
    
    def test_metrics_computation(self):
        # Perfect predictions
        targets = np.array([0, 0, 1, 1])
        predictions = np.array([0, 0, 1, 1])
        probabilities = np.array([0.1, 0.2, 0.8, 0.9])
        
        evaluator = ModelEvaluator(None, torch.device('cpu'))
        results = evaluator.compute_metrics(predictions, probabilities, targets)
        
        assert results.accuracy == 1.0
        assert results.precision == 1.0
        assert results.recall == 1.0
        assert results.f1_score == 1.0
        assert results.roc_auc == 1.0
    
    def test_confusion_matrix(self):
        targets = np.array([0, 0, 1, 1])
        predictions = np.array([0, 1, 0, 1])
        
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(targets, predictions)
        
        assert cm.shape == (2, 2)
        assert cm[0, 0] == 1  # TN
        assert cm[0, 1] == 1  # FP
        assert cm[1, 0] == 1  # FN
        assert cm[1, 1] == 1  # TP


class TestTraining:
    """Test training utilities."""
    
    def test_early_stopping(self):
        early_stopping = EarlyStopping(patience=3, min_delta=0.01)
        model = torch.nn.Linear(10, 2)
        
        # Should not stop initially
        assert not early_stopping(1.0, model, 0)
        assert not early_stopping(0.9, model, 1)
        assert not early_stopping(0.8, model, 2)
        # Should stop after patience
        assert early_stopping(0.85, model, 3)  # No improvement
        assert early_stopping(0.87, model, 4)  # Still no improvement
        assert early_stopping(0.86, model, 5)  # Still no improvement -> stop
    
    def test_average_meter(self):
        meter = AverageMeter('Test')
        meter.update(1.0, 2)
        meter.update(2.0, 3)
        
        assert meter.count == 5
        assert abs(meter.avg - 1.6) < 0.001
    
    def test_model_checkpoint(self, tmp_path):
        checkpoint = ModelCheckpoint(
            filepath=str(tmp_path / "checkpoint.pth"),
            save_best_only=True
        )
        
        model = torch.nn.Linear(10, 2)
        optimizer = torch.optim.Adam(model.parameters())
        scheduler = None
        
        # Should save as best
        checkpoint(model, optimizer, scheduler, 0, {'val_loss': 1.0}, is_last=False)
        assert (tmp_path / "best_model.pth").exists()


class TestUtils:
    """Test utility functions."""
    
    def test_helpers(self):
        from src.utils.helpers import ensure_dir, format_number, AverageMeter
        
        # Test format_number
        assert format_number(1000) == "1,000"
        assert format_number(1000000) == "1,000,000"
        
        # Test ensure_dir
        test_dir = Path("/tmp/test_ensure_dir")
        ensure_dir(test_dir)
        assert test_dir.exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
"""
Breast Cancer CNN Classifier
============================

A complete deep learning pipeline for Invasive Ductal Carcinoma (IDC) 
detection from histopathology images.

Modules:
- data: Data exploration, preprocessing, and augmentation
- models: CNN architectures (CustomCNN, Transfer Learning)
- training: Training loop with callbacks, early stopping, LR scheduling
- evaluation: Comprehensive metrics (Accuracy, Precision, Recall, F1, ROC-AUC, CI)
- dashboard: Interactive Streamlit dashboard
- utils: Configuration, reproducibility, logging, helpers

Usage:
    from breast_cancer_cnn import create_model, create_trainer, evaluate_model
"""

__version__ = "1.0.0"
__author__ = "Research Team"
__license__ = "MIT"

# Main imports for easy access
from src.data import (
    DataExplorer,
    run_full_exploration,
    BreastCancerDataset,
    DataPreprocessor,
    create_preprocessor
)

from src.models import (
    CustomCNN,
    TransferLearningModel,
    create_model,
    ModelEMA
)

from src.training import (
    Trainer,
    EarlyStopping,
    ModelCheckpoint,
    LearningRateScheduler,
    create_trainer,
    create_optimizer,
    create_scheduler,
    create_criterion
)

from src.evaluation import (
    ModelEvaluator,
    EvaluationResults,
    evaluate_model,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_precision_recall_curve,
    plot_metrics_comparison,
    generate_evaluation_plots
)

from src.dashboard import app

from src.utils import (
    load_config,
    config,
    set_seed,
    get_device,
    print_system_info,
    setup_logging,
    get_logger,
    TrainingLogger
)

__all__ = [
    # Data
    'DataExplorer',
    'run_full_exploration',
    'BreastCancerDataset',
    'DataPreprocessor',
    'create_preprocessor',
    
    # Models
    'CustomCNN',
    'TransferLearningModel',
    'create_model',
    'ModelEMA',
    
    # Training
    'Trainer',
    'EarlyStopping',
    'ModelCheckpoint',
    'LearningRateScheduler',
    'create_trainer',
    'create_optimizer',
    'create_scheduler',
    'create_criterion',
    
    # Evaluation
    'ModelEvaluator',
    'EvaluationResults',
    'evaluate_model',
    'plot_confusion_matrix',
    'plot_roc_curve',
    'plot_precision_recall_curve',
    'plot_metrics_comparison',
    'generate_evaluation_plots',
    
    # Dashboard
    'app',
    
    # Utils
    'load_config',
    'config',
    'set_seed',
    'get_device',
    'print_system_info',
    'setup_logging',
    'get_logger',
    'TrainingLogger',
]
# Architecture Documentation

## Overview

This document describes the software architecture of the Breast Cancer CNN Classifier Dashboard.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Interface                            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │   CLI       │  │  Streamlit  │  │   Report    │             │
│  │  (main.py)  │  │  Dashboard  │  │  Generator  │             │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘             │
└─────────│────────────────│────────────────│────────────────────┘
          │                │                │
          ▼                ▼                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Core Pipeline                               │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐            │
│  │ Data Layer   │ │ Model Layer  │ │ Training     │            │
│  │ (src/data)   │ │ (src/models) │ │ (src/training)           │
│  └──────┬───────┘ └──────┬───────┘ └──────┬───────┘            │
└─────────│────────────────│────────────────│────────────────────┘
          │                │                │
          ▼                ▼                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Evaluation & Utils                            │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐            │
│  │ Evaluation   │ │ Utils        │ │ Config       │            │
│  │ (src/eval)   │ │ (src/utils)  │ │ (configs/)   │            │
│  └──────────────┘ └──────────────┘ └──────────────┘            │
└─────────────────────────────────────────────────────────────────┘
```

## Module Details

### Data Layer (`src/data/`)

#### `exploration.py`
- **DataExplorer** class for comprehensive dataset analysis
- Class balance analysis with imbalance ratio calculation
- Image dimension verification and statistics
- Pixel value statistics per channel (mean, std, min, max)
- Visualization generation (class distribution, dimension analysis, sample images)

#### `preprocessing.py`
- **BreastCancerDataset**: PyTorch Dataset with on-the-fly augmentation
- **DataPreprocessor**: Complete preprocessing pipeline
- Stratified train/val/test split (70/15/15)
- Albumentations-based augmentation pipeline
- WeightedRandomSampler for class balancing
- DataLoader creation with configurable workers

### Model Layer (`src/models/`)

#### `architecture.py`
- **CustomCNN**: Configurable CNN with BatchNorm, Dropout, GlobalAvgPool
- **TransferLearningModel**: Wrapper for pretrained models (ResNet, EfficientNet, ConvNeXt, ViT)
- **ConvBlock**: Reusable convolutional block
- **ModelEMA**: Exponential Moving Average for better generalization
- Factory function `create_model()` for easy instantiation

### Training Layer (`src/training/`)

#### `trainer.py`
- **Trainer**: Main training loop with mixed precision support
- **EarlyStopping**: Patience-based stopping with weight restoration
- **ModelCheckpoint**: Best/last model saving
- **LearningRateScheduler**: Wrapper for multiple scheduler types
- **create_trainer()**: Factory for fully configured trainer

### Evaluation Layer (`src/evaluation/`)

#### `metrics.py`
- **ModelEvaluator**: Comprehensive evaluation pipeline
- **EvaluationResults**: Dataclass for structured results
- Metrics: Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC
- Confusion Matrix (raw & normalized)
- Bootstrap Confidence Intervals (1000 iterations default)
- Visualization functions for all metrics

### Dashboard Layer (`src/dashboard/`)

#### `app.py`
- Streamlit-based interactive dashboard
- 5 tabs: Training History, Model Evaluation, Live Prediction, Data Explorer, Report Generator
- Real-time training curve visualization (Plotly)
- Interactive confusion matrix and ROC/PR curves
- Live image upload and prediction with probability gauge
- APA 7 technical report generation (fpdf2)

### Utilities (`src/utils/`)

#### `config.py`
- YAML-based configuration with typed dataclasses
- Singleton pattern for global access
- Dot-notation key access

#### `reproducibility.py`
- Comprehensive seed setting (Python, NumPy, PyTorch, CUDA)
- Deterministic algorithm enforcement
- Device detection (CUDA, MPS, CPU)
- Context manager for temporary seeds

#### `logging_utils.py`
- Rich console logging with tracebacks
- File logging support
- TrainingLogger for epoch/batch progress

#### `helpers.py`
- Checkpoint save/load
- Parameter counting
- AverageMeter/ProgressMeter
- File utilities (JSON, pickle)
- Experiment directory creation

## Data Flow

```
Raw Images (data/raw/)
    │
    ▼
DataExplorer (analysis & visualization)
    │
    ▼
DataPreprocessor
    ├── Stratified Split (70/15/15)
    ├── Augmentation Pipeline
    ├── Normalization (ImageNet stats)
    └── DataLoader Creation
    │
    ▼
Model (CustomCNN / TransferLearning)
    │
    ▼
Trainer
    ├── Mixed Precision (FP16)
    ├── Gradient Accumulation
    ├── LR Scheduling
    ├── Early Stopping
    └── Checkpointing
    │
    ▼
ModelEvaluator
    ├── Predictions on Test Set
    ├── Metrics Computation
    ├── Bootstrap CI
    └── Visualization Generation
    │
    ▼
Dashboard / Report
    ├── Streamlit App
    └── APA 7 PDF Report
```

## Configuration

All hyperparameters and settings centralized in `configs/config.yaml`:

- **Data**: Paths, splits, augmentation, normalization
- **Model**: Architecture, filter progression, dropout, BatchNorm
- **Training**: Epochs, LR, optimizer, scheduler, early stopping, mixed precision
- **Evaluation**: Metrics, threshold, bootstrap CI
- **Dashboard**: Host, port, theme, plot settings
- **Logging**: MLflow, TensorBoard, console
- **Reproducibility**: Seeds, deterministic mode

## Design Patterns

1. **Factory Pattern**: `create_model()`, `create_trainer()`, `create_preprocessor()`
2. **Singleton Pattern**: Config class
3. **Strategy Pattern**: LearningRateScheduler, EarlyStopping
4. **Context Manager**: ReproducibilityContext
5. **Dataclass**: Configuration objects, EvaluationResults
6. **Dependency Injection**: Trainer receives model, loaders, criterion, optimizer

## Extensibility Points

1. **New Architectures**: Add to `architecture.py`, register in `create_model()`
2. **New Augmentations**: Extend `get_train_transforms()` in `preprocessing.py`
3. **New Metrics**: Add to `ModelEvaluator.compute_metrics()` in `metrics.py`
4. **New Schedulers**: Extend `LearningRateScheduler` in `trainer.py`
5. **New Dashboard Tabs**: Add to `app.py` Streamlit tabs
6. **New Report Sections**: Extend `generate_technical_report()` in `app.py`

## Performance Considerations

- **Mixed Precision**: FP16 reduces memory by ~50%, speeds up training
- **Persistent Workers**: DataLoader workers persist across epochs
- **Pin Memory**: Faster GPU transfer
- **Weighted Sampling**: Balances classes without oversampling
- **Gradient Accumulation**: Effective larger batch sizes
- **EMA**: Better generalization without extra compute

## Security Considerations

- No sensitive data in code (API keys in .env only)
- Model weights saved locally
- No external network calls during training
- Input validation on image loading
- Path traversal protection in file operations

## Testing Strategy

- Unit tests for each module (`tests/`)
- Integration tests for full pipeline
- Synthetic data for CI/CD
- Visual regression for dashboard
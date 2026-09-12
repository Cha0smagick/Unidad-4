"""
Preprocessing pipeline for Breast Cancer CNN Classifier.
Handles train/val/test split, normalization, augmentation, and data loading.
"""

import os
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from collections import defaultdict
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import cv2

from src.utils.config import config
from src.utils.logging_utils import get_logger
from src.utils.reproducibility import set_seed
from src.utils.helpers import get_file_list, validate_image_file, ensure_dir, save_json

logger = get_logger(__name__)


class BreastCancerDataset(Dataset):
    """
    PyTorch Dataset for breast cancer histopathology images.
    
    Supports:
    - Train/val/test splits with stratification
    - On-the-fly data augmentation
    - Image normalization
    - Class balancing with weighted sampling
    """
    
    def __init__(
        self,
        image_paths: List[str],
        labels: List[int],
        transform: Optional[A.Compose] = None,
        target_size: Tuple[int, int] = (50, 50),
        normalize: bool = True
    ):
        """
        Initialize dataset.
        
        Args:
            image_paths: List of image file paths
            labels: List of class labels (0 or 1)
            transform: Albumentations transform pipeline
            target_size: Target image size (height, width)
            normalize: Whether to apply normalization
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.target_size = target_size
        self.normalize = normalize
        
        # Default normalization (ImageNet stats)
        self.mean = np.array(config.preprocessing.normalize.mean, dtype=np.float32)
        self.std = np.array(config.preprocessing.normalize.std, dtype=np.float32)
        
        logger.info(f"Dataset initialized with {len(image_paths)} samples")
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        # Load image
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        try:
            with Image.open(image_path) as img:
                img = img.convert('RGB')
                image = np.array(img)
        except Exception as e:
            logger.warning(f"Error loading {image_path}: {e}. Using black image.")
            image = np.zeros((self.target_size[1], self.target_size[0], 3), dtype=np.uint8)
        
        # Resize if needed
        if image.shape[:2] != (self.target_size[1], self.target_size[0]):
            image = cv2.resize(image, (self.target_size[0], self.target_size[1]), interpolation=cv2.INTER_AREA)
        
        # Apply augmentations
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        
        # Convert to tensor and normalize
        if isinstance(image, np.ndarray):
            image = image.astype(np.float32) / 255.0
            if self.normalize:
                image = (image - self.mean) / self.std
            image = torch.from_numpy(image).permute(2, 0, 1)  # HWC to CHW
        
        return image, label
    
    def get_class_distribution(self) -> Dict[int, int]:
        """Get class distribution in dataset."""
        return dict(Counter(self.labels))
    
    def get_class_weights(self) -> torch.Tensor:
        """Compute class weights for balanced sampling."""
        class_counts = self.get_class_distribution()
        total = len(self.labels)
        weights = []
        for label in self.labels:
            class_weight = total / (len(class_counts) * class_counts[label])
            weights.append(class_weight)
        return torch.DoubleTensor(weights)


def get_train_transforms(image_size: Tuple[int, int] = (50, 50)) -> A.Compose:
    """
    Get training augmentation pipeline.
    
    Args:
        image_size: Target image size
        
    Returns:
        Albumentations Compose pipeline
    """
    aug_config = config.data.augmentation
    
    transforms = [
        A.Resize(image_size[1], image_size[0], interpolation=cv2.INTER_AREA),
    ]
    
    if aug_config.get('horizontal_flip', True):
        transforms.append(A.HorizontalFlip(p=0.5))
    
    if aug_config.get('vertical_flip', True):
        transforms.append(A.VerticalFlip(p=0.5))
    
    if aug_config.get('rotate_limit', 90):
        transforms.append(A.Rotate(limit=aug_config['rotate_limit'], p=0.5, border_mode=cv2.BORDER_REFLECT))
    
    if aug_config.get('brightness_contrast', True):
        transforms.append(A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3))
    
    if aug_config.get('gaussian_noise', True):
        transforms.append(A.GaussNoise(var_limit=(10, 50), p=0.2))
    
    if aug_config.get('elastic_transform', True):
        transforms.append(A.ElasticTransform(alpha=1, sigma=5, p=0.2, border_mode=cv2.BORDER_REFLECT))
    
    # Additional medical imaging augmentations
    transforms.extend([
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.3, border_mode=cv2.BORDER_REFLECT),
        A.CoarseDropout(max_holes=8, max_height=8, max_width=8, p=0.2),
    ])
    
    # Normalize and convert to tensor
    transforms.append(A.Normalize(
        mean=config.preprocessing.normalize.mean,
        std=config.preprocessing.normalize.std
    ))
    transforms.append(ToTensorV2())
    
    return A.Compose(transforms)


def get_val_transforms(image_size: Tuple[int, int] = (50, 50)) -> A.Compose:
    """
    Get validation/test transform pipeline (no augmentation).
    
    Args:
        image_size: Target image size
        
    Returns:
        Albumentations Compose pipeline
    """
    return A.Compose([
        A.Resize(image_size[1], image_size[0], interpolation=cv2.INTER_AREA),
        A.Normalize(
            mean=config.preprocessing.normalize.mean,
            std=config.preprocessing.normalize.std
        ),
        ToTensorV2()
    ])


class DataPreprocessor:
    """
    Complete data preprocessing pipeline:
    - Dataset scanning and validation
    - Stratified train/val/test split
    - Data augmentation setup
    - DataLoader creation with optional weighted sampling
    """
    
    def __init__(
        self,
        data_path: Optional[str] = None,
        random_seed: Optional[int] = None
    ):
        """
        Initialize preprocessor.
        
        Args:
            data_path: Path to raw data directory
            random_seed: Random seed for reproducibility
        """
        self.data_path = Path(data_path) if data_path else Path(config.data.raw_data_path)
        self.random_seed = random_seed or config.data.random_seed
        self.class_names = config.data.class_names
        self.target_size = tuple(config.data.image_size)
        
        # Set seeds for reproducibility
        set_seed(self.random_seed)
        
        # Results storage
        self.splits = {}
        self.dataloaders = {}
        self.class_weights = None
    
    def prepare_dataset(self) -> Tuple[List[str], List[int]]:
        """
        Scan dataset and prepare image paths and labels.
        
        Returns:
            Tuple of (image_paths, labels)
        """
        logger.info(f"Preparing dataset from: {self.data_path}")
        
        image_paths = []
        labels = []
        
        for class_idx, class_name in enumerate(self.class_names):
            class_dir = self.data_path / class_name
            
            if not class_dir.exists():
                logger.warning(f"Class directory not found: {class_dir}")
                continue
            
            # Get all valid image files
            files = get_file_list(class_dir, ['.png', '.jpg', '.jpeg', '.tif', '.tiff'])
            valid_files = [f for f in files if validate_image_file(f)]
            
            logger.info(f"Class '{class_name}' ({class_idx}): {len(valid_files)} valid images")
            
            image_paths.extend([str(f) for f in valid_files])
            labels.extend([class_idx] * len(valid_files))
        
        # Shuffle maintaining correspondence
        combined = list(zip(image_paths, labels))
        random.seed(self.random_seed)
        random.shuffle(combined)
        image_paths, labels = zip(*combined)
        
        logger.info(f"Total dataset: {len(image_paths)} images")
        logger.info(f"Class distribution: {dict(Counter(labels))}")
        
        return list(image_paths), list(labels)
    
    def create_splits(
        self,
        image_paths: List[str],
        labels: List[int],
        train_ratio: Optional[float] = None,
        val_ratio: Optional[float] = None,
        test_ratio: Optional[float] = None
    ) -> Dict[str, Tuple[List[str], List[int]]]:
        """
        Create stratified train/val/test splits.
        
        Args:
            image_paths: List of image paths
            labels: List of labels
            train_ratio: Training split ratio
            val_ratio: Validation split ratio
            test_ratio: Test split ratio
            
        Returns:
            Dictionary with splits
        """
        train_ratio = train_ratio or config.data.train_split
        val_ratio = val_ratio or config.data.val_split
        test_ratio = test_ratio or config.data.test_split
        
        # Verify ratios sum to 1
        total_ratio = train_ratio + val_ratio + test_ratio
        if abs(total_ratio - 1.0) > 1e-6:
            raise ValueError(f"Split ratios must sum to 1.0, got {total_ratio}")
        
        logger.info(f"Creating splits: train={train_ratio}, val={val_ratio}, test={test_ratio}")
        
        # First split: train vs (val + test)
        train_paths, temp_paths, train_labels, temp_labels = train_test_split(
            image_paths, labels,
            train_size=train_ratio,
            stratify=labels,
            random_state=self.random_seed
        )
        
        # Second split: val vs test from remaining
        val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
        val_paths, test_paths, val_labels, test_labels = train_test_split(
            temp_paths, temp_labels,
            train_size=val_ratio_adjusted,
            stratify=temp_labels,
            random_state=self.random_seed
        )
        
        splits = {
            'train': (list(train_paths), list(train_labels)),
            'val': (list(val_paths), list(val_labels)),
            'test': (list(test_paths), list(test_labels))
        }
        
        # Log split distributions
        for split_name, (paths, lbls) in splits.items():
            dist = Counter(lbls)
            logger.info(f"{split_name.capitalize()}: {len(paths)} samples - {dict(dist)}")
        
        self.splits = splits
        return splits
    
    def compute_class_weights(self, labels: List[int]) -> torch.Tensor:
        """Compute class weights for loss function."""
        class_counts = Counter(labels)
        total = len(labels)
        num_classes = len(self.class_names)
        
        weights = []
        for i in range(num_classes):
            weight = total / (num_classes * class_counts.get(i, 1))
            weights.append(weight)
        
        weights_tensor = torch.FloatTensor(weights)
        logger.info(f"Class weights: {dict(zip(self.class_names, weights))}")
        
        self.class_weights = weights_tensor
        return weights_tensor
    
    def create_dataloaders(
        self,
        batch_size: Optional[int] = None,
        num_workers: Optional[int] = None,
        use_weighted_sampler: bool = True,
        pin_memory: Optional[bool] = None
    ) -> Dict[str, DataLoader]:
        """
        Create DataLoaders for all splits.
        
        Args:
            batch_size: Batch size
            num_workers: Number of worker processes
            use_weighted_sampler: Whether to use WeightedRandomSampler for training
            pin_memory: Whether to pin memory for GPU transfer
            
        Returns:
            Dictionary of DataLoaders
        """
        batch_size = batch_size or config.preprocessing.batch_size
        num_workers = num_workers or config.preprocessing.num_workers
        pin_memory = pin_memory if pin_memory is not None else config.preprocessing.pin_memory
        
        if not self.splits:
            image_paths, labels = self.prepare_dataset()
            self.create_splits(image_paths, labels)
        
        # Get transforms
        train_transform = get_train_transforms(self.target_size)
        val_transform = get_val_transforms(self.target_size)
        
        # Create datasets
        datasets = {}
        datasets['train'] = BreastCancerDataset(
            *self.splits['train'],
            transform=train_transform,
            target_size=self.target_size
        )
        datasets['val'] = BreastCancerDataset(
            *self.splits['val'],
            transform=val_transform,
            target_size=self.target_size
        )
        datasets['test'] = BreastCancerDataset(
            *self.splits['test'],
            transform=val_transform,
            target_size=self.target_size
        )
        
        # Create dataloaders
        dataloaders = {}
        
        # Training loader with optional weighted sampler
        if use_weighted_sampler:
            train_weights = datasets['train'].get_class_weights()
            sampler = WeightedRandomSampler(
                train_weights, 
                num_samples=len(train_weights), 
                replacement=True
            )
            dataloaders['train'] = DataLoader(
                datasets['train'],
                batch_size=batch_size,
                sampler=sampler,
                num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers=config.preprocessing.persistent_workers and num_workers > 0
            )
        else:
            dataloaders['train'] = DataLoader(
                datasets['train'],
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers=config.preprocessing.persistent_workers and num_workers > 0
            )
        
        # Validation and test loaders (no shuffle)
        for split in ['val', 'test']:
            dataloaders[split] = DataLoader(
                datasets[split],
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers=config.preprocessing.persistent_workers and num_workers > 0
            )
        
        self.dataloaders = dataloaders
        
        logger.info(f"DataLoaders created:")
        for name, loader in dataloaders.items():
            logger.info(f"  {name}: {len(loader)} batches of size {batch_size}")
        
        return dataloaders
    
    def save_split_info(self, output_path: str) -> None:
        """Save split information to JSON."""
        split_info = {}
        for split_name, (paths, labels) in self.splits.items():
            split_info[split_name] = {
                'num_samples': len(paths),
                'class_distribution': dict(Counter(labels)),
                'image_paths': paths,
                'labels': labels
            }
        
        save_json(split_info, output_path)
        logger.info(f"Split info saved to: {output_path}")
    
    def run_full_pipeline(
        self,
        output_dir: Optional[str] = None,
        batch_size: Optional[int] = None
    ) -> Dict[str, DataLoader]:
        """
        Run complete preprocessing pipeline.
        
        Args:
            output_dir: Directory to save split info
            batch_size: Batch size for DataLoaders
            
        Returns:
            Dictionary of DataLoaders
        """
        # Prepare dataset
        image_paths, labels = self.prepare_dataset()
        
        # Create splits
        self.create_splits(image_paths, labels)
        
        # Compute class weights
        self.compute_class_weights(labels)
        
        # Create dataloaders
        dataloaders = self.create_dataloaders(batch_size=batch_size)
        
        # Save split info
        if output_dir:
            output_path = Path(output_dir)
            ensure_dir(output_path)
            self.save_split_info(output_path / "split_info.json")
            
            # Save class weights
            if self.class_weights is not None:
                torch.save(self.class_weights, output_path / "class_weights.pth")
        
        return dataloaders


def create_preprocessor(
    data_path: Optional[str] = None,
    random_seed: Optional[int] = None
) -> DataPreprocessor:
    """Factory function to create DataPreprocessor."""
    return DataPreprocessor(data_path, random_seed)


if __name__ == "__main__":
    # Example usage
    preprocessor = create_preprocessor()
    dataloaders = preprocessor.run_full_pipeline(output_dir="data/processed")
    
    # Test dataloaders
    for name, loader in dataloaders.items():
        batch = next(iter(loader))
        images, labels = batch
        print(f"{name}: images shape={images.shape}, labels shape={labels.shape}")
        print(f"  Image range: [{images.min():.4f}, {images.max():.4f}]")
        print(f"  Labels: {labels[:10].tolist()}")
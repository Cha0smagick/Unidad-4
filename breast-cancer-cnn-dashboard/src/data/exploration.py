"""
Data exploration and analysis module for Breast Cancer CNN Classifier.
Provides comprehensive dataset analysis including class balance, image dimensions, and statistics.
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import Counter
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from src.utils.config import config
from src.utils.logging_utils import get_logger
from src.utils.helpers import validate_image_file, get_file_list

logger = get_logger(__name__)


class DataExplorer:
    """Comprehensive data exploration and analysis for histopathology images."""
    
    def __init__(self, data_path: Optional[str] = None):
        """
        Initialize data explorer.
        
        Args:
            data_path: Path to raw data directory
        """
        self.data_path = Path(data_path) if data_path else Path(config.data.raw_data_path)
        self.class_names = config.data.class_names
        self.results = {}
    
    def scan_dataset(self) -> Dict[str, Any]:
        """
        Scan dataset directory structure and collect basic statistics.
        
        Returns:
            Dictionary with dataset scan results
        """
        logger.info(f"Scanning dataset at: {self.data_path}")
        
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data path does not exist: {self.data_path}")
        
        # Expected structure: data_path/class_name/*.png
        class_dirs = [d for d in self.data_path.iterdir() if d.is_dir()]
        
        scan_results = {
            'total_classes': len(class_dirs),
            'classes': {},
            'total_images': 0,
            'extensions': Counter(),
            'corrupted_files': []
        }
        
        for class_dir in class_dirs:
            class_name = class_dir.name
            image_files = get_file_list(class_dir, ['.png', '.jpg', '.jpeg', '.tif', '.tiff'])
            
            valid_images = []
            corrupted = []
            
            for img_file in image_files:
                if validate_image_file(img_file):
                    valid_images.append(img_file)
                else:
                    corrupted.append(str(img_file))
            
            scan_results['classes'][class_name] = {
                'count': len(valid_images),
                'files': [str(f) for f in valid_images]
            }
            scan_results['total_images'] += len(valid_images)
            scan_results['corrupted_files'].extend(corrupted)
            
            # Count extensions
            for f in valid_images:
                scan_results['extensions'][f.suffix.lower()] += 1
        
        self.results['scan'] = scan_results
        logger.info(f"Found {scan_results['total_images']} valid images across {scan_results['total_classes']} classes")
        
        if scan_results['corrupted_files']:
            logger.warning(f"Found {len(scan_results['corrupted_files'])} corrupted files")
        
        return scan_results
    
    def analyze_class_balance(self) -> Dict[str, Any]:
        """
        Analyze class distribution and balance.
        
        Returns:
            Dictionary with class balance analysis
        """
        if 'scan' not in self.results:
            self.scan_dataset()
        
        scan = self.results['scan']
        class_counts = {cls: info['count'] for cls, info in scan['classes'].items()}
        total = scan['total_images']
        
        balance_analysis = {
            'class_counts': class_counts,
            'class_percentages': {cls: (count / total * 100) for cls, count in class_counts.items()},
            'total_samples': total,
            'imbalance_ratio': max(class_counts.values()) / min(class_counts.values()) if min(class_counts.values()) > 0 else float('inf'),
            'is_balanced': max(class_counts.values()) / min(class_counts.values()) < 2.0 if min(class_counts.values()) > 0 else False
        }
        
        self.results['class_balance'] = balance_analysis
        logger.info(f"Class distribution: {class_counts}")
        logger.info(f"Imbalance ratio: {balance_analysis['imbalance_ratio']:.2f}")
        
        return balance_analysis
    
    def analyze_image_dimensions(self, sample_size: int = 1000) -> Dict[str, Any]:
        """
        Analyze image dimensions across the dataset.
        
        Args:
            sample_size: Number of images to sample for analysis (None for all)
            
        Returns:
            Dictionary with image dimension statistics
        """
        if 'scan' not in self.results:
            self.scan_dataset()
        
        all_files = []
        for class_info in self.results['scan']['classes'].values():
            all_files.extend(class_info['files'])
        
        if sample_size and len(all_files) > sample_size:
            np.random.seed(config.reproducibility.seeds.numpy)
            sampled_files = np.random.choice(all_files, sample_size, replace=False)
        else:
            sampled_files = all_files
        
        logger.info(f"Analyzing dimensions for {len(sampled_files)} images...")
        
        dimensions = []
        modes = []
        file_sizes = []
        
        for file_path in tqdm(sampled_files, desc="Analyzing dimensions"):
            try:
                with Image.open(file_path) as img:
                    dimensions.append(img.size)  # (width, height)
                    modes.append(img.mode)
                    file_sizes.append(os.path.getsize(file_path))
            except Exception as e:
                logger.warning(f"Error reading {file_path}: {e}")
        
        widths = [d[0] for d in dimensions]
        heights = [d[1] for d in dimensions]
        
        dim_analysis = {
            'sample_size': len(sampled_files),
            'width_stats': {
                'min': min(widths),
                'max': max(widths),
                'mean': np.mean(widths),
                'std': np.std(widths),
                'median': np.median(widths),
                'unique_values': sorted(set(widths))
            },
            'height_stats': {
                'min': min(heights),
                'max': max(heights),
                'mean': np.mean(heights),
                'std': np.std(heights),
                'median': np.median(heights),
                'unique_values': sorted(set(heights))
            },
            'aspect_ratios': {
                'min': min(w/h for w, h in dimensions),
                'max': max(w/h for w, h in dimensions),
                'mean': np.mean([w/h for w, h in dimensions]),
                'std': np.std([w/h for w, h in dimensions])
            },
            'color_modes': dict(Counter(modes)),
            'file_size_stats': {
                'min_bytes': min(file_sizes),
                'max_bytes': max(file_sizes),
                'mean_bytes': np.mean(file_sizes),
                'mean_mb': np.mean(file_sizes) / (1024 * 1024)
            },
            'consistent_size': len(set(dimensions)) == 1,
            'target_size': config.data.image_size
        }
        
        self.results['dimensions'] = dim_analysis
        logger.info(f"Image dimensions - Width: {dim_analysis['width_stats']['unique_values']}, Height: {dim_analysis['height_stats']['unique_values']}")
        logger.info(f"Consistent size: {dim_analysis['consistent_size']}")
        
        return dim_analysis
    
    def analyze_pixel_statistics(self, sample_size: int = 500) -> Dict[str, Any]:
        """
        Analyze pixel value statistics (mean, std, min, max) per channel.
        
        Args:
            sample_size: Number of images to sample
            
        Returns:
            Dictionary with pixel statistics
        """
        if 'scan' not in self.results:
            self.scan_dataset()
        
        all_files = []
        for class_info in self.results['scan']['classes'].values():
            all_files.extend(class_info['files'])
        
        if sample_size and len(all_files) > sample_size:
            np.random.seed(config.reproducibility.seeds.numpy)
            sampled_files = np.random.choice(all_files, sample_size, replace=False)
        else:
            sampled_files = all_files
        
        logger.info(f"Computing pixel statistics for {len(sampled_files)} images...")
        
        channel_means = {0: [], 1: [], 2: []}
        channel_stds = {0: [], 1: [], 2: []}
        channel_mins = {0: [], 1: [], 2: []}
        channel_maxs = {0: [], 1: [], 2: []}
        
        for file_path in tqdm(sampled_files, desc="Computing pixel stats"):
            try:
                with Image.open(file_path) as img:
                    img = img.convert('RGB')
                    arr = np.array(img, dtype=np.float32) / 255.0
                    
                    for c in range(3):
                        channel_means[c].append(arr[:, :, c].mean())
                        channel_stds[c].append(arr[:, :, c].std())
                        channel_mins[c].append(arr[:, :, c].min())
                        channel_maxs[c].append(arr[:, :, c].max())
            except Exception as e:
                logger.warning(f"Error processing {file_path}: {e}")
        
        pixel_stats = {
            'sample_size': len(sampled_files),
            'per_channel': {}
        }
        
        channel_names = ['Red', 'Green', 'Blue']
        for c in range(3):
            pixel_stats['per_channel'][channel_names[c]] = {
                'mean': np.mean(channel_means[c]),
                'std': np.mean(channel_stds[c]),
                'min': np.min(channel_mins[c]),
                'max': np.max(channel_maxs[c]),
                'mean_of_means': np.mean(channel_means[c]),
                'std_of_means': np.std(channel_means[c])
            }
        
        # Overall statistics
        all_means = [v for c in range(3) for v in channel_means[c]]
        all_stds = [v for c in range(3) for v in channel_stds[c]]
        
        pixel_stats['overall'] = {
            'mean': np.mean(all_means),
            'std': np.mean(all_stds),
            'min': min(channel_mins[c] for c in range(3)),
            'max': max(channel_maxs[c] for c in range(3))
        }
        
        self.results['pixel_statistics'] = pixel_stats
        logger.info(f"Pixel statistics computed. Overall mean: {pixel_stats['overall']['mean']:.4f}, std: {pixel_stats['overall']['std']:.4f}")
        
        return pixel_stats
    
    def generate_exploration_report(self, output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate comprehensive exploration report.
        
        Args:
            output_path: Optional path to save JSON report
            
        Returns:
            Complete exploration results dictionary
        """
        # Run all analyses
        self.scan_dataset()
        self.analyze_class_balance()
        self.analyze_image_dimensions()
        self.analyze_pixel_statistics()
        
        # Compile report
        report = {
            'dataset_path': str(self.data_path),
            'scan_results': self.results.get('scan', {}),
            'class_balance': self.results.get('class_balance', {}),
            'dimension_analysis': self.results.get('dimensions', {}),
            'pixel_statistics': self.results.get('pixel_statistics', {}),
            'summary': {
                'total_images': self.results.get('scan', {}).get('total_images', 0),
                'num_classes': self.results.get('scan', {}).get('total_classes', 0),
                'class_distribution': self.results.get('class_balance', {}).get('class_counts', {}),
                'imbalance_ratio': self.results.get('class_balance', {}).get('imbalance_ratio', 0),
                'consistent_dimensions': self.results.get('dimensions', {}).get('consistent_size', False),
                'target_size_match': self.results.get('dimensions', {}).get('target_size', []) == 
                                   self.results.get('dimensions', {}).get('width_stats', {}).get('unique_values', [])
            }
        }
        
        if output_path:
            save_json(report, output_path)
            logger.info(f"Exploration report saved to: {output_path}")
        
        return report
    
    def plot_class_distribution(self, save_path: Optional[str] = None) -> plt.Figure:
        """Plot class distribution bar chart."""
        if 'class_balance' not in self.results:
            self.analyze_class_balance()
        
        balance = self.results['class_balance']
        classes = list(balance['class_counts'].keys())
        counts = list(balance['class_counts'].values())
        percentages = list(balance['class_percentages'].values())
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Count plot
        bars1 = ax1.bar(classes, counts, color=['#2E86AB', '#A23B72'], edgecolor='black', alpha=0.8)
        ax1.set_title('Class Distribution (Count)', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Number of Images', fontsize=12)
        ax1.set_xlabel('Class', fontsize=12)
        
        # Add value labels
        for bar, count in zip(bars1, counts):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                    str(count), ha='center', va='bottom', fontweight='bold')
        
        # Percentage plot
        bars2 = ax2.bar(classes, percentages, color=['#2E86AB', '#A23B72'], edgecolor='black', alpha=0.8)
        ax2.set_title('Class Distribution (Percentage)', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Percentage (%)', fontsize=12)
        ax2.set_xlabel('Class', fontsize=12)
        ax2.set_ylim(0, 100)
        
        for bar, pct in zip(bars2, percentages):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                    f'{pct:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Class distribution plot saved to: {save_path}")
        
        return fig
    
    def plot_dimension_distribution(self, save_path: Optional[str] = None) -> plt.Figure:
        """Plot image dimension distributions."""
        if 'dimensions' not in self.results:
            self.analyze_image_dimensions()
        
        dim = self.results['dimensions']
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Width distribution
        ax1 = axes[0, 0]
        ax1.bar(range(len(dim['width_stats']['unique_values'])), 
                [1]*len(dim['width_stats']['unique_values']),  # Placeholder
                tick_label=[str(v) for v in dim['width_stats']['unique_values']])
        ax1.set_title('Unique Width Values', fontweight='bold')
        ax1.set_ylabel('Count')
        
        # Height distribution
        ax2 = axes[0, 1]
        ax2.bar(range(len(dim['height_stats']['unique_values'])), 
                [1]*len(dim['height_stats']['unique_values']),
                tick_label=[str(v) for v in dim['height_stats']['unique_values']])
        ax2.set_title('Unique Height Values', fontweight='bold')
        ax2.set_ylabel('Count')
        
        # Aspect ratio
        ax3 = axes[1, 0]
        ax3.text(0.1, 0.5, f"Min: {dim['aspect_ratios']['min']:.2f}\n"
                           f"Max: {dim['aspect_ratios']['max']:.2f}\n"
                           f"Mean: {dim['aspect_ratios']['mean']:.2f}\n"
                           f"Std: {dim['aspect_ratios']['std']:.2f}",
                 transform=ax3.transAxes, fontsize=12, verticalalignment='center')
        ax3.set_title('Aspect Ratio Statistics', fontweight='bold')
        ax3.axis('off')
        
        # File size
        ax4 = axes[1, 1]
        fs = dim['file_size_stats']
        ax4.text(0.1, 0.5, f"Min: {fs['min_bytes']/1024:.1f} KB\n"
                           f"Max: {fs['max_bytes']/1024:.1f} KB\n"
                           f"Mean: {fs['mean_bytes']/1024:.1f} KB",
                 transform=ax4.transAxes, fontsize=12, verticalalignment='center')
        ax4.set_title('File Size Statistics', fontweight='bold')
        ax4.axis('off')
        
        plt.suptitle('Image Dimension Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Dimension distribution plot saved to: {save_path}")
        
        return fig
    
    def plot_pixel_statistics(self, save_path: Optional[str] = None) -> plt.Figure:
        """Plot pixel statistics per channel."""
        if 'pixel_statistics' not in self.results:
            self.analyze_pixel_statistics()
        
        stats = self.results['pixel_statistics']
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        channels = ['Red', 'Green', 'Blue']
        colors = ['#E74C3C', '#2ECC71', '#3498DB']
        
        for idx, (channel, color) in enumerate(zip(channels, colors)):
            ax = axes[idx]
            ch_stats = stats['per_channel'][channel]
            
            metrics = ['mean', 'std', 'min', 'max']
            values = [ch_stats[m] for m in metrics]
            
            bars = ax.bar(metrics, values, color=color, alpha=0.7, edgecolor='black')
            ax.set_title(f'{channel} Channel Statistics', fontweight='bold')
            ax.set_ylabel('Value')
            ax.set_ylim(0, max(values) * 1.2)
            
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                       f'{val:.4f}', ha='center', va='bottom', fontsize=10)
        
        plt.suptitle('Pixel Value Statistics per Channel', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Pixel statistics plot saved to: {save_path}")
        
        return fig
    
    def visualize_sample_images(self, num_samples: int = 8, save_path: Optional[str] = None) -> plt.Figure:
        """Visualize sample images from each class."""
        if 'scan' not in self.results:
            self.scan_dataset()
        
        fig, axes = plt.subplots(len(self.class_names), num_samples // len(self.class_names), 
                                 figsize=(15, 6))
        
        if len(self.class_names) == 1:
            axes = axes.reshape(1, -1)
        
        for class_idx, class_name in enumerate(self.class_names):
            class_files = self.results['scan']['classes'][class_name]['files']
            sampled = np.random.choice(class_files, min(num_samples // len(self.class_names), len(class_files)), replace=False)
            
            for sample_idx, file_path in enumerate(sampled):
                ax = axes[class_idx, sample_idx]
                with Image.open(file_path) as img:
                    ax.imshow(img)
                ax.set_title(f'{class_name}\n{img.size[0]}x{img.size[1]}', fontsize=10)
                ax.axis('off')
        
        plt.suptitle('Sample Images from Dataset', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Sample images plot saved to: {save_path}")
        
        return fig


def save_json(data: Any, path: Union[str, Path], indent: int = 2) -> None:
    """Save data as JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def run_full_exploration(data_path: str, output_dir: str) -> Dict[str, Any]:
    """
    Run complete data exploration pipeline.
    
    Args:
        data_path: Path to raw data
        output_dir: Directory to save results
        
    Returns:
        Complete exploration results
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    explorer = DataExplorer(data_path)
    
    # Generate report
    report = explorer.generate_exploration_report(output_path / "exploration_report.json")
    
    # Generate plots
    explorer.plot_class_distribution(output_path / "class_distribution.png")
    explorer.plot_dimension_distribution(output_path / "dimension_analysis.png")
    explorer.plot_pixel_statistics(output_path / "pixel_statistics.png")
    explorer.visualize_sample_images(save_path=output_path / "sample_images.png")
    
    logger.info(f"Full exploration complete. Results saved to: {output_path}")
    
    return report


if __name__ == "__main__":
    # Example usage
    import sys
    data_path = sys.argv[1] if len(sys.argv) > 1 else "data/raw"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "reports/exploration"
    
    run_full_exploration(data_path, output_dir)
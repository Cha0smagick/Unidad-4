"""
Evaluation metrics for Breast Cancer CNN Classifier.
Implements Accuracy, Recall, Precision, F1-Score, AUC-ROC, Confusion Matrix, and Bootstrap CI.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve, auc
)
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import json

from src.utils.config import config
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class EvaluationResults:
    """Container for evaluation results."""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float
    confusion_matrix: np.ndarray
    classification_report: Dict[str, Any]
    per_class_metrics: Dict[str, Dict[str, float]]
    roc_curve: Dict[str, np.ndarray]
    pr_curve: Dict[str, np.ndarray]
    bootstrap_ci: Optional[Dict[str, Tuple[float, float]]] = None
    threshold: float = 0.5


class ModelEvaluator:
    """
    Comprehensive model evaluator with all required metrics.
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        class_names: List[str] = None,
        threshold: float = 0.5
    ):
        """
        Initialize evaluator.
        
        Args:
            model: Trained PyTorch model
            device: Device to run evaluation on
            class_names: List of class names
            threshold: Decision threshold for binary classification
        """
        self.model = model
        self.device = device
        self.class_names = class_names or config.data.class_names
        self.threshold = threshold
        
        self.model.to(device)
        self.model.eval()
    
    @torch.no_grad()
    def predict(self, dataloader: torch.utils.data.DataLoader) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get predictions, probabilities, and targets from dataloader.
        
        Returns:
            Tuple of (predictions, probabilities, targets)
        """
        all_preds = []
        all_probs = []
        all_targets = []
        
        for images, targets in tqdm(dataloader, desc="Predicting", leave=False):
            images = images.to(self.device, non_blocking=True)
            
            outputs = self.model(images)
            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            preds = (probs >= self.threshold).astype(int)
            
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_targets.extend(targets.numpy())
        
        return (
            np.array(all_preds),
            np.array(all_probs),
            np.array(all_targets)
        )
    
    def compute_metrics(
        self,
        predictions: np.ndarray,
        probabilities: np.ndarray,
        targets: np.ndarray
    ) -> EvaluationResults:
        """
        Compute all evaluation metrics.
        
        Args:
            predictions: Binary predictions (0 or 1)
            probabilities: Predicted probabilities for class 1
            targets: True labels
            
        Returns:
            EvaluationResults object
        """
        # Basic metrics
        accuracy = accuracy_score(targets, predictions)
        precision = precision_score(targets, predictions, average='binary', zero_division=0)
        recall = recall_score(targets, predictions, average='binary', zero_division=0)
        f1 = f1_score(targets, predictions, average='binary', zero_division=0)
        
        # ROC AUC
        try:
            roc_auc = roc_auc_score(targets, probabilities)
        except ValueError:
            roc_auc = 0.5
            logger.warning("Could not compute ROC AUC (single class in targets)")
        
        # Confusion matrix
        cm = confusion_matrix(targets, predictions)
        
        # Classification report
        report = classification_report(
            targets, predictions,
            target_names=self.class_names,
            output_dict=True,
            zero_division=0
        )
        
        # Per-class metrics
        per_class = {}
        for i, class_name in enumerate(self.class_names):
            if str(i) in report:
                per_class[class_name] = {
                    'precision': report[str(i)]['precision'],
                    'recall': report[str(i)]['recall'],
                    'f1_score': report[str(i)]['f1-score'],
                    'support': report[str(i)]['support']
                }
        
        # ROC curve
        fpr, tpr, roc_thresholds = roc_curve(targets, probabilities)
        roc_auc_value = auc(fpr, tpr)
        
        # Precision-Recall curve
        prec, rec, pr_thresholds = precision_recall_curve(targets, probabilities)
        pr_auc = auc(rec, prec)
        
        results = EvaluationResults(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            roc_auc=roc_auc,
            confusion_matrix=cm,
            classification_report=report,
            per_class_metrics=per_class,
            roc_curve={
                'fpr': fpr,
                'tpr': tpr,
                'thresholds': roc_thresholds,
                'auc': roc_auc_value
            },
            pr_curve={
                'precision': prec,
                'recall': rec,
                'thresholds': pr_thresholds,
                'auc': pr_auc
            },
            threshold=self.threshold
        )
        
        return results
    
    def compute_bootstrap_ci(
        self,
        predictions: np.ndarray,
        probabilities: np.ndarray,
        targets: np.ndarray,
        n_iterations: int = 1000,
        confidence_level: float = 0.95
    ) -> Dict[str, Tuple[float, float]]:
        """
        Compute bootstrap confidence intervals for metrics.
        
        Args:
            predictions: Binary predictions
            probabilities: Predicted probabilities
            targets: True labels
            n_iterations: Number of bootstrap iterations
            confidence_level: Confidence level (e.g., 0.95)
            
        Returns:
            Dictionary of metric -> (lower_ci, upper_ci)
        """
        n_samples = len(targets)
        metrics_bootstrap = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1_score': [],
            'roc_auc': []
        }
        
        np.random.seed(config.reproducibility.seeds.numpy)
        
        for i in range(n_iterations):
            # Bootstrap sample
            indices = np.random.choice(n_samples, n_samples, replace=True)
            boot_preds = predictions[indices]
            boot_probs = probabilities[indices]
            boot_targets = targets[indices]
            
            # Compute metrics
            try:
                metrics_bootstrap['accuracy'].append(accuracy_score(boot_targets, boot_preds))
                metrics_bootstrap['precision'].append(precision_score(boot_targets, boot_preds, zero_division=0))
                metrics_bootstrap['recall'].append(recall_score(boot_targets, boot_preds, zero_division=0))
                metrics_bootstrap['f1_score'].append(f1_score(boot_targets, boot_preds, zero_division=0))
                metrics_bootstrap['roc_auc'].append(roc_auc_score(boot_targets, boot_probs))
            except ValueError:
                # Skip iterations where metric can't be computed
                continue
        
        # Compute confidence intervals
        alpha = (1 - confidence_level) / 2
        ci_results = {}
        
        for metric, values in metrics_bootstrap.items():
            if len(values) > 0:
                lower = np.percentile(values, alpha * 100)
                upper = np.percentile(values, (1 - alpha) * 100)
                ci_results[metric] = (float(lower), float(upper))
            else:
                ci_results[metric] = (0.0, 1.0)
        
        return ci_results
    
    def evaluate(
        self,
        dataloader: torch.utils.data.DataLoader,
        compute_ci: bool = True
    ) -> EvaluationResults:
        """
        Full evaluation pipeline.
        
        Args:
            dataloader: DataLoader to evaluate on
            compute_ci: Whether to compute bootstrap confidence intervals
            
        Returns:
            EvaluationResults object
        """
        logger.info(f"Evaluating on {len(dataloader.dataset)} samples...")
        
        # Get predictions
        predictions, probabilities, targets = self.predict(dataloader)
        
        # Compute metrics
        results = self.compute_metrics(predictions, probabilities, targets)
        
        # Compute bootstrap CI if requested
        if compute_ci and config.evaluation.bootstrap_ci.enabled:
            logger.info("Computing bootstrap confidence intervals...")
            n_iter = config.evaluation.bootstrap_ci.n_iterations
            conf_level = config.evaluation.bootstrap_ci.confidence_level
            results.bootstrap_ci = self.compute_bootstrap_ci(
                predictions, probabilities, targets,
                n_iterations=n_iter,
                confidence_level=conf_level
            )
        
        # Log results
        self._log_results(results)
        
        return results
    
    def _log_results(self, results: EvaluationResults):
        """Log evaluation results."""
        logger.info("=" * 50)
        logger.info("EVALUATION RESULTS")
        logger.info("=" * 50)
        logger.info(f"Accuracy:  {results.accuracy:.4f}")
        logger.info(f"Precision: {results.precision:.4f}")
        logger.info(f"Recall:    {results.recall:.4f}")
        logger.info(f"F1-Score:  {results.f1_score:.4f}")
        logger.info(f"ROC-AUC:   {results.roc_auc:.4f}")
        logger.info("-" * 50)
        logger.info("Confusion Matrix:")
        logger.info(f"  TN={results.confusion_matrix[0,0]}, FP={results.confusion_matrix[0,1]}")
        logger.info(f"  FN={results.confusion_matrix[1,0]}, TP={results.confusion_matrix[1,1]}")
        logger.info("-" * 50)
        
        for class_name, metrics in results.per_class_metrics.items():
            logger.info(f"{class_name}: P={metrics['precision']:.4f}, R={metrics['recall']:.4f}, F1={metrics['f1_score']:.4f}")
        
        if results.bootstrap_ci:
            logger.info("-" * 50)
            logger.info("Bootstrap 95% CI:")
            for metric, (lower, upper) in results.bootstrap_ci.items():
                logger.info(f"  {metric}: [{lower:.4f}, {upper:.4f}]")
        
        logger.info("=" * 50)
    
    def save_results(self, results: EvaluationResults, output_path: str):
        """Save evaluation results to JSON."""
        output_data = {
            'accuracy': results.accuracy,
            'precision': results.precision,
            'recall': results.recall,
            'f1_score': results.f1_score,
            'roc_auc': results.roc_auc,
            'confusion_matrix': results.confusion_matrix.tolist(),
            'classification_report': results.classification_report,
            'per_class_metrics': results.per_class_metrics,
            'roc_curve': {k: v.tolist() for k, v in results.roc_curve.items() if k != 'auc'},
            'roc_auc': results.roc_curve['auc'],
            'pr_curve': {k: v.tolist() for k, v in results.pr_curve.items() if k != 'auc'},
            'pr_auc': results.pr_curve['auc'],
            'threshold': results.threshold,
            'bootstrap_ci': results.bootstrap_ci
        }
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        logger.info(f"Results saved to: {output_path}")


def evaluate_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    class_names: List[str] = None,
    threshold: float = 0.5,
    compute_ci: bool = True
) -> EvaluationResults:
    """
    Convenience function to evaluate a model.
    
    Args:
        model: Trained PyTorch model
        dataloader: DataLoader to evaluate on
        device: Device to run evaluation on
        class_names: List of class names
        threshold: Decision threshold
        compute_ci: Whether to compute bootstrap CI
        
    Returns:
        EvaluationResults object
    """
    evaluator = ModelEvaluator(model, device, class_names, threshold)
    return evaluator.evaluate(dataloader, compute_ci)


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    normalize: bool = False,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 6)
) -> plt.Figure:
    """
    Plot confusion matrix.
    
    Args:
        cm: Confusion matrix array
        class_names: List of class names
        normalize: Whether to normalize by row
        save_path: Optional path to save figure
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    if normalize:
        cm_display = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
        title = 'Normalized Confusion Matrix'
    else:
        cm_display = cm
        fmt = 'd'
        title = 'Confusion Matrix'
    
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(
        cm_display,
        annot=True,
        fmt=fmt,
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        cbar_kws={'label': 'Proportion' if normalize else 'Count'}
    )
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_xlabel('Predicted Label', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Confusion matrix saved to: {save_path}")
    
    return fig


def plot_roc_curve(
    results: EvaluationResults,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 6)
) -> plt.Figure:
    """
    Plot ROC curve.
    
    Args:
        results: EvaluationResults object
        save_path: Optional path to save figure
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    roc = results.roc_curve
    ax.plot(roc['fpr'], roc['tpr'], linewidth=2, 
            label=f'ROC Curve (AUC = {roc["auc"]:.4f})', color='#2E86AB')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
    
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('Receiver Operating Characteristic (ROC) Curve', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"ROC curve saved to: {save_path}")
    
    return fig


def plot_precision_recall_curve(
    results: EvaluationResults,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 6)
) -> plt.Figure:
    """
    Plot Precision-Recall curve.
    
    Args:
        results: EvaluationResults object
        save_path: Optional path to save figure
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    pr = results.pr_curve
    ax.plot(pr['recall'], pr['precision'], linewidth=2,
            label=f'PR Curve (AUC = {pr["auc"]:.4f})', color='#A23B72')
    
    # Baseline (random classifier)
    baseline = results.classification_report.get('1', {}).get('support', 0) / sum(
        results.classification_report.get(str(i), {}).get('support', 0) for i in range(len(results.class_names))
    )
    ax.axhline(y=baseline, color='k', linestyle='--', linewidth=1, label=f'Random (baseline={baseline:.2f})')
    
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    ax.legend(loc='lower left', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"PR curve saved to: {save_path}")
    
    return fig


def plot_metrics_comparison(
    results: EvaluationResults,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    Plot metrics comparison bar chart.
    
    Args:
        results: EvaluationResults object
        save_path: Optional path to save figure
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    values = [results.accuracy, results.precision, results.recall, results.f1_score, results.roc_auc]
    colors = ['#2E86AB', '#3498DB', '#2ECC71', '#F39C12', '#A23B72']
    
    fig, ax = plt.subplots(figsize=figsize)
    
    bars = ax.bar(metrics, values, color=colors, edgecolor='black', alpha=0.8, width=0.6)
    
    # Add value labels
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    # Add CI error bars if available
    if results.bootstrap_ci:
        metric_names = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        for i, (metric, (lower, upper)) in enumerate([(m, results.bootstrap_ci.get(m, (0, 1))) for m in metric_names]):
            ax.errorbar(i, values[i], yerr=[[values[i] - lower], [upper - values[i]]],
                       fmt='none', color='black', capsize=5, capthick=2)
    
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Performance Metrics', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1.1])
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Metrics comparison saved to: {save_path}")
    
    return fig


def generate_evaluation_plots(
    results: EvaluationResults,
    class_names: List[str],
    output_dir: str
):
    """Generate all evaluation plots."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Confusion matrix (raw and normalized)
    plot_confusion_matrix(
        results.confusion_matrix, class_names,
        normalize=False,
        save_path=output_path / "confusion_matrix.png"
    )
    plot_confusion_matrix(
        results.confusion_matrix, class_names,
        normalize=True,
        save_path=output_path / "confusion_matrix_normalized.png"
    )
    
    # ROC curve
    plot_roc_curve(results, save_path=output_path / "roc_curve.png")
    
    # PR curve
    plot_precision_recall_curve(results, save_path=output_path / "pr_curve.png")
    
    # Metrics comparison
    plot_metrics_comparison(results, save_path=output_path / "metrics_comparison.png")
    
    logger.info(f"All evaluation plots saved to: {output_path}")


if __name__ == "__main__":
    # Example usage with dummy data
    np.random.seed(42)
    targets = np.random.randint(0, 2, 1000)
    probs = np.random.beta(2, 2, 1000)
    probs[targets == 1] = np.random.beta(5, 2, targets.sum())
    preds = (probs >= 0.5).astype(int)
    
    evaluator = ModelEvaluator(None, torch.device('cpu'))
    results = evaluator.compute_metrics(preds, probs, targets)
    print(f"Accuracy: {results.accuracy:.4f}")
    print(f"F1: {results.f1_score:.4f}")
    print(f"ROC-AUC: {results.roc_auc:.4f}")
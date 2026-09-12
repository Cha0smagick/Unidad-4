"""
Evaluation module for Breast Cancer CNN Classifier.
"""

from src.evaluation.metrics import (
    ModelEvaluator,
    EvaluationResults,
    evaluate_model,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_precision_recall_curve,
    plot_metrics_comparison,
    generate_evaluation_plots
)

__all__ = [
    'ModelEvaluator',
    'EvaluationResults',
    'evaluate_model',
    'plot_confusion_matrix',
    'plot_roc_curve',
    'plot_precision_recall_curve',
    'plot_metrics_comparison',
    'generate_evaluation_plots'
]
#!/usr/bin/env python
"""
Breast Cancer CNN Classifier - Main Entry Point
================================================

Complete pipeline for Invasive Ductal Carcinoma (IDC) detection from histopathology images.

Usage:
    python main.py --mode train          # Train model
    python main.py --mode evaluate       # Evaluate model
    python main.py --mode dashboard      # Launch Streamlit dashboard
    python main.py --mode explore        # Data exploration
    python main.py --mode preprocess     # Data preprocessing
    python main.py --mode report         # Generate technical report
"""

import argparse
import sys
import os
from pathlib import Path

# Add src to path for Streamlit Cloud
sys.path.insert(0, str(Path(__file__).parent / "src"))

import torch
import numpy as np

from src.utils.config import config, load_config
from src.utils.reproducibility import set_seed, print_system_info, get_device
from src.utils.logging_utils import setup_logging, get_logger
from src.data import run_full_exploration, create_preprocessor
from src.models import create_model
from src.training import create_trainer
from src.evaluation import evaluate_model, generate_evaluation_plots
from src.dashboard.app import generate_technical_report

logger = get_logger(__name__)


def setup_environment():
    """Setup environment for reproducibility."""
    # Set seeds
    seeds = config.reproducibility.seeds
    set_seed(
        seed=seeds.python,
        deterministic=config.reproducibility.deterministic,
        benchmark=config.reproducibility.benchmark
    )
    
    # Print system info
    print_system_info()
    
    # Setup logging
    log_file = Path("logs/training.log")
    log_file.parent.mkdir(parents=True, exist_ok=True)
    setup_logging(level="INFO", log_file=str(log_file))


def mode_explore(args):
    """Run data exploration."""
    logger.info("Starting data exploration...")
    
    data_path = args.data_path or config.data.raw_data_path
    output_dir = args.output_dir or "reports/exploration"
    
    run_full_exploration(data_path, output_dir)
    
    logger.info("Data exploration completed!")


def mode_preprocess(args):
    """Run data preprocessing."""
    logger.info("Starting data preprocessing...")
    
    preprocessor = create_preprocessor()
    dataloaders = preprocessor.run_full_pipeline(output_dir="data/processed")
    
    logger.info("Data preprocessing completed!")
    return preprocessor, dataloaders


def mode_train(args):
    """Train the model."""
    logger.info("Starting model training...")
    
    # Setup
    device = get_device()
    
    # Data
    if args.data_path:
        preprocessor = create_preprocessor(args.data_path)
    else:
        preprocessor = create_preprocessor()
    
    dataloaders = preprocessor.run_full_pipeline(output_dir="data/processed")
    
    # Model
    model = create_model(args.architecture or config.model.architecture)
    
    # Trainer
    trainer = create_trainer(
        model,
        dataloaders['train'],
        dataloaders['val'],
        preprocessor.class_weights,
        device
    )
    
    # Train
    epochs = args.epochs or config.training.epochs
    history = trainer.fit(epochs=epochs)
    
    # Save history
    history_path = Path(config.training.checkpoint_dir) / "training_history.json"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    import json
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    logger.info("Training completed!")
    return trainer, history


def mode_evaluate(args):
    """Evaluate the model."""
    logger.info("Starting model evaluation...")
    
    device = get_device()
    
    # Load model
    model_path = args.model_path or str(Path(config.training.checkpoint_dir) / "best_model.pth")
    model = create_model(args.architecture or config.model.architecture)
    
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    # Data
    preprocessor = create_preprocessor()
    dataloaders = preprocessor.run_full_pipeline()
    
    # Evaluate
    results = evaluate_model(
        model,
        dataloaders['test'],
        device,
        config.data.class_names,
        args.threshold or config.evaluation.threshold,
        compute_ci=args.compute_ci
    )
    
    # Save results and plots
    model_path = Path(model_path)
    evaluator = evaluate_model.__globals__['ModelEvaluator'](model, device, config.data.class_names)
    evaluator.save_results(results, str(model_path.parent / "evaluation_results.json"))
    generate_evaluation_plots(results, config.data.class_names, str(model_path.parent / "eval_plots"))
    
    logger.info("Evaluation completed!")
    return results


def mode_dashboard(args):
    """Launch Streamlit dashboard."""
    logger.info("Launching Streamlit dashboard...")
    
    import subprocess
    dashboard_path = Path(__file__).parent / "src" / "dashboard" / "app.py"
    
    cmd = [
        "streamlit", "run", str(dashboard_path),
        "--server.port", str(config.dashboard.port),
        "--server.address", config.dashboard.host
    ]
    
    subprocess.run(cmd)


def mode_report(args):
    """Generate technical report."""
    logger.info("Generating technical report...")
    
    report_path = generate_technical_report()
    
    logger.info(f"Report generated: {report_path}")


def mode_full_pipeline(args):
    """Run complete pipeline: explore -> preprocess -> train -> evaluate -> report."""
    logger.info("Running full pipeline...")
    
    # 1. Explore
    logger.info("Step 1/5: Data Exploration")
    mode_explore(args)
    
    # 2. Preprocess
    logger.info("Step 2/5: Data Preprocessing")
    preprocessor, dataloaders = mode_preprocess(args)
    
    # 3. Train
    logger.info("Step 3/5: Model Training")
    trainer, history = mode_train(args)
    
    # 4. Evaluate
    logger.info("Step 4/5: Model Evaluation")
    results = mode_evaluate(args)
    
    # 5. Report
    logger.info("Step 5/5: Report Generation")
    mode_report(args)
    
    logger.info("Full pipeline completed!")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Breast Cancer CNN Classifier - IDC Detection Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --mode explore                    # Data exploration
  python main.py --mode preprocess                 # Data preprocessing
  python main.py --mode train --epochs 100         # Train model
  python main.py --mode evaluate                   # Evaluate model
  python main.py --mode dashboard                  # Launch dashboard
  python main.py --mode report                     # Generate report
  python main.py --mode full                       # Run full pipeline
        """
    )
    
    parser.add_argument(
        '--mode',
        choices=['explore', 'preprocess', 'train', 'evaluate', 'dashboard', 'report', 'full'],
        default='full',
        help='Pipeline mode to run'
    )
    
    parser.add_argument('--data-path', type=str, help='Path to raw data directory')
    parser.add_argument('--output-dir', type=str, help='Output directory for results')
    parser.add_argument('--model-path', type=str, help='Path to model checkpoint')
    parser.add_argument('--architecture', type=str, help='Model architecture')
    parser.add_argument('--epochs', type=int, help='Number of training epochs')
    parser.add_argument('--threshold', type=float, help='Classification threshold')
    parser.add_argument('--compute-ci', action='store_true', help='Compute bootstrap CI')
    
    args = parser.parse_args()
    
    # Load configuration
    load_config()
    
    # Setup environment
    setup_environment()
    
    # Run selected mode
    mode_functions = {
        'explore': mode_explore,
        'preprocess': mode_preprocess,
        'train': mode_train,
        'evaluate': mode_evaluate,
        'dashboard': mode_dashboard,
        'report': mode_report,
        'full': mode_full_pipeline
    }
    
    mode_functions[args.mode](args)


if __name__ == "__main__":
    main()
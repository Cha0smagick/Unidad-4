"""
Logging utilities for the Breast Cancer CNN Classifier project.
Provides structured logging with Rich console output and file logging.
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from rich.console import Console
from rich.logging import RichHandler
from rich.traceback import install as install_rich_traceback

# Install rich traceback handler
install_rich_traceback(show_locals=True)


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    format_string: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    date_format: str = "%Y-%m-%d %H:%M:%S"
) -> logging.Logger:
    """
    Set up structured logging with Rich console handler and optional file handler.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file
        format_string: Log format string
        date_format: Date format string
        
    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger("breast_cancer_cnn")
    logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Rich console handler
    console = Console()
    console_handler = RichHandler(
        console=console,
        show_time=True,
        show_path=False,
        markup=True,
        rich_tracebacks=True,
        tracebacks_show_locals=True
    )
    console_handler.setLevel(getattr(logging, level.upper()))
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)  # Always debug to file
        file_formatter = logging.Formatter(format_string, date_format)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    return logger


def get_logger(name: str = "breast_cancer_cnn") -> logging.Logger:
    """Get logger instance."""
    return logging.getLogger(name)


class TrainingLogger:
    """Specialized logger for training progress with metrics tracking."""
    
    def __init__(self, logger: logging.Logger, log_interval: int = 10):
        self.logger = logger
        self.log_interval = log_interval
        self.epoch = 0
        self.batch = 0
    
    def log_epoch_start(self, epoch: int, total_epochs: int):
        """Log epoch start."""
        self.epoch = epoch
        self.logger.info(f"[bold blue]Epoch {epoch}/{total_epochs}[/bold blue]")
    
    def log_batch(self, batch: int, total_batches: int, loss: float, metrics: dict = None):
        """Log batch progress."""
        self.batch = batch
        if batch % self.log_interval == 0:
            msg = f"  Batch {batch}/{total_batches} - Loss: {loss:.4f}"
            if metrics:
                metric_str = " - ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
                msg += f" - {metric_str}"
            self.logger.info(msg)
    
    def log_epoch_end(self, train_metrics: dict, val_metrics: dict = None):
        """Log epoch end with metrics."""
        train_msg = "  [green]Train:[/green] " + " - ".join([f"{k}: {v:.4f}" for k, v in train_metrics.items()])
        self.logger.info(train_msg)
        
        if val_metrics:
            val_msg = "  [yellow]Val:[/yellow] " + " - ".join([f"{k}: {v:.4f}" for k, v in val_metrics.items()])
            self.logger.info(val_msg)
    
    def log_best_model(self, metric_name: str, value: float, epoch: int):
        """Log best model checkpoint."""
        self.logger.info(f"  [bold green]New best model![/bold green] {metric_name}: {value:.4f} at epoch {epoch}")
    
    def log_early_stopping(self, patience: int):
        """Log early stopping trigger."""
        self.logger.warning(f"[bold red]Early stopping triggered after {patience} epochs without improvement[/bold red]")


# Initialize default logger
default_logger = setup_logging()
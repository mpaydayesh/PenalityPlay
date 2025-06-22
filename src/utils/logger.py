import os
import json
import time
from datetime import datetime
from typing import Dict, Any, Optional, List, Union
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import seaborn as sns

class Logger:
    """
    A logger class for tracking training progress and results.
    """
    
    def __init__(self, log_dir: str = "logs", experiment_name: Optional[str] = None):
        """
        Initialize the logger.
        
        Args:
            log_dir: Base directory for logs
            experiment_name: Optional name for the experiment
        """
        # Create experiment name with timestamp if not provided
        if experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.experiment_name = f"experiment_{timestamp}"
        else:
            self.experiment_name = experiment_name
        
        # Set up directories
        self.log_dir = Path(log_dir) / self.experiment_name
        self.metrics_dir = self.log_dir / "metrics"
        self.models_dir = self.log_dir / "models"
        self.plots_dir = self.log_dir / "plots"
        
        # Create directories
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize metrics storage
        self.metrics: Dict[str, List[Dict[str, Any]]] = {
            'train': [],
            'eval': []
        }
        
        # Start time
        self.start_time = time.time()
    
    def log_metrics(
        self, 
        metrics: Dict[str, Any], 
        step: int, 
        prefix: str = ""
    ) -> None:
        """
        Log metrics to the console and save to file.
        
        Args:
            metrics: Dictionary of metrics to log
            step: Current training step
            prefix: Optional prefix for the metrics (e.g., 'train_', 'eval_')
        """
        # Add timestamp and step
        log_data = {
            'timestamp': datetime.now().isoformat(),
            'step': step,
            'elapsed_time': time.time() - self.start_time,
            **metrics
        }
        
        # Determine log type (train or eval)
        log_type = 'train' if 'train' in prefix else 'eval'
        self.metrics[log_type].append(log_data)
        
        # Save to file
        self._save_metrics(log_type)
        
        # Print to console
        print(f"\nStep {step} - {prefix}")
        for key, value in metrics.items():
            print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")
    
    def _save_metrics(self, log_type: str) -> None:
        """
        Save metrics to a JSON file.
        
        Args:
            log_type: Type of log ('train' or 'eval')
        """
        file_path = self.metrics_dir / f"{log_type}_metrics.json"
        with open(file_path, 'w') as f:
            json.dump(self.metrics[log_type], f, indent=2)
    
    def plot_metrics(self, show: bool = False) -> None:
        """
        Plot training and evaluation metrics.
        
        Args:
            show: Whether to show the plots
        """
        # Plot training metrics
        self._plot_metric('train', 'episode_reward', 'Training Episode Reward', show=show)
        self._plot_metric('train', 'episode_length', 'Training Episode Length', show=show)
        
        # Plot evaluation metrics if available
        if self.metrics['eval']:
            self._plot_metric('eval', 'mean_reward', 'Evaluation Mean Reward', show=show)
            self._plot_metric('eval', 'success_rate', 'Evaluation Success Rate', show=show)
    
    def _plot_metric(
        self, 
        log_type: str, 
        metric_name: str, 
        title: str, 
        show: bool = False
    ) -> None:
        """
        Plot a specific metric.
        
        Args:
            log_type: Type of log ('train' or 'eval')
            metric_name: Name of the metric to plot
            title: Plot title
            show: Whether to show the plot
        """
        if not self.metrics[log_type]:
            return
        
        # Create dataframe from metrics
        df = pd.DataFrame(self.metrics[log_type])
        
        # Check if metric exists
        if metric_name not in df.columns:
            return
        
        # Create plot
        plt.figure(figsize=(10, 6))
        sns.set_style("whitegrid")
        
        # Plot metric vs step
        sns.lineplot(data=df, x='step', y=metric_name, label=metric_name)
        
        # Customize plot
        plt.title(title)
        plt.xlabel('Training Steps')
        plt.ylabel(metric_name.replace('_', ' ').title())
        plt.legend()
        
        # Save plot
        plot_path = self.plots_dir / f"{log_type}_{metric_name}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
        plt.close()
    
    def save_model(self, model: Any, name: str) -> None:
        """
        Save a model to disk.
        
        Args:
            model: The model to save
            name: Name of the model
        """
        import torch
        
        model_path = self.models_dir / f"{name}.pt"
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")
    
    def load_model(self, model: Any, name: str) -> Any:
        """
        Load a model from disk.
        
        Args:
            model: The model to load weights into
            name: Name of the model
            
        Returns:
            The loaded model
        """
        import torch
        
        model_path = self.models_dir / f"{name}.pt"
        if model_path.exists():
            model.load_state_dict(torch.load(model_path))
            print(f"Model loaded from {model_path}")
        else:
            print(f"Warning: Model {model_path} not found")
        
        return model
    
    def log_params(self, params: Dict[str, Any]) -> None:
        """
        Log hyperparameters to a JSON file.
        
        Args:
            params: Dictionary of hyperparameters
        """
        params_path = self.log_dir / "params.json"
        with open(params_path, 'w') as f:
            json.dump(params, f, indent=2)
    
    def log_config(self, config: Dict[str, Any]) -> None:
        """
        Log configuration to a YAML file.
        
        Args:
            config: Dictionary of configuration parameters
        """
        import yaml
        
        config_path = self.log_dir / "config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the training run.
        
        Returns:
            Dictionary with training summary
        """
        if not self.metrics['train'] and not self.metrics['eval']:
            return {}
        
        summary = {
            'experiment_name': self.experiment_name,
            'start_time': datetime.fromtimestamp(self.start_time).isoformat(),
            'duration_seconds': time.time() - self.start_time,
            'num_train_steps': len(self.metrics['train']),
            'num_eval_steps': len(self.metrics['eval']),
        }
        
        # Add final metrics if available
        if self.metrics['train']:
            summary['final_train_metrics'] = self.metrics['train'][-1]
        
        if self.metrics['eval']:
            summary['final_eval_metrics'] = self.metrics['eval'][-1]
        
        return summary


def setup_logger(name: str, log_level: str = 'INFO', log_file: str = None) -> logging.Logger:
    """
    Configure and return a logger with console and optional file handlers.
    
    Args:
        name: Logger name (usually __name__)
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file
        
    Returns:
        Configured logger instance
    """
    import logging
    from pathlib import Path
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    
    # Clear existing handlers to avoid duplicate logs
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler if log_file is provided
    if log_file:
        # Create directory if it doesn't exist
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

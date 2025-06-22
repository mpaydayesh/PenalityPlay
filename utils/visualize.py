"""
Visualization utilities for the Penalty Shootout RL project.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional
import pandas as pd
import seaborn as sns

def plot_training_metrics(log_dir: str, output_dir: str = 'plots') -> None:
    """Plot training metrics from TensorBoard logs.
    
    Args:
        log_dir: Directory containing TensorBoard logs
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # This is a placeholder - in practice, you would use tensorboardX or similar
    # to parse the actual log files
    print(f"Visualization would process logs from {log_dir} and save to {output_dir}")

def plot_episode_rewards(rewards: Dict[str, List[float]], 
                        window: int = 10,
                        output_path: Optional[str] = None) -> None:
    """Plot episode rewards with moving average.
    
    Args:
        rewards: Dictionary of rewards by agent type
        window: Window size for moving average
        output_path: Optional path to save the plot
    """
    plt.figure(figsize=(12, 6))
    
    for agent, r in rewards.items():
        # Calculate moving average
        if len(r) >= window:
            ma = np.convolve(r, np.ones(window)/window, mode='valid')
            x = np.arange(window-1, len(r))
            plt.plot(x, ma, label=f'{agent} (MA{window})')
        
        # Plot raw rewards (transparent)
        plt.plot(r, alpha=0.2, label=f'{agent} (raw)' if agent + ' (raw)' not in plt.gca().get_legend_handles_labels()[1] else "")
    
    plt.title('Training Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid(True)
    
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
    else:
        plt.show()
    plt.close()

def plot_evaluation_results(results: Dict[str, float], 
                          output_path: Optional[str] = None) -> None:
    """Plot evaluation results.

    Args:
        results: Dictionary of evaluation metrics
        output_path: Optional path to save the plot
    """
    plt.figure(figsize=(10, 6))
    
    # Filter and sort metrics for plotting
    metrics = {k: v for k, v in results.items() if not k.startswith('num_')}
    
    # Create bar plot
    names = list(metrics.keys())
    values = list(metrics.values())
    
    bars = plt.bar(names, values)
    plt.title('Evaluation Metrics')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom')
    
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
    else:
        plt.show()
    plt.close()

def compare_agents(agent_results: Dict[str, Dict[str, float]],
                  metric: str = 'mean_reward',
                  output_path: Optional[str] = None) -> None:
    """Compare multiple agents on a specific metric.
    
    Args:
        agent_results: Dictionary mapping agent names to their results
        metric: Metric to compare
        output_path: Optional path to save the plot
    """
    plt.figure(figsize=(10, 6))
    
    names = list(agent_results.keys())
    values = [results.get(metric, 0) for results in agent_results.values()]
    
    bars = plt.bar(names, values)
    plt.title(f'Agent Comparison: {metric.replace("_", " ").title()}')
    plt.ylabel(metric.replace('_', ' ').title())
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom')
    
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
    else:
        plt.show()
    plt.close()

if __name__ == "__main__":
    # Example usage
    rewards = {
        'striker': np.random.normal(0, 1, 100).cumsum(),
        'goalkeeper': np.random.normal(0, 0.8, 100).cumsum()
    }
    plot_episode_rewards(rewards, window=10)

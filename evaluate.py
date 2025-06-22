#!/usr/bin/env python3
"""
Evaluation script for the Penalty Shootout RL project.
"""
import os
import sys
import yaml
import numpy as np
from typing import Dict, Any, List, Tuple

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from src.envs.penalty_env import PenaltyShootoutEnv
from src.agents import StrikerAgent, GoalkeeperAgent
from src.utils.logger import setup_logger

def load_config(config_path: str = "configs/agent_params.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def evaluate_episode(env, striker, goalkeeper, max_steps: int = 1000) -> Tuple[float, int, Dict]:
    """Run a single evaluation episode."""
    obs = env.reset()
    total_reward = 0
    done = False
    step = 0
    
    while not done and step < max_steps:
        action = {
            'striker': striker.act(obs, deterministic=True),
            'goalkeeper': goalkeeper.act(obs, deterministic=True)
        }
        obs, reward, done, info = env.step(action)
        total_reward += reward['striker']  # Track striker's reward
        step += 1
    
    return total_reward, step, info

def evaluate_models(striker_path: str, goalkeeper_path: str, num_episodes: int = 10) -> Dict[str, Any]:
    """Evaluate trained models over multiple episodes."""
    logger = setup_logger('evaluate')
    logger.info(f"Starting evaluation for {num_episodes} episodes")
    
    # Initialize environment and agents
    env = PenaltyShootoutEnv()
    striker = StrikerAgent(env)
    goalkeeper = GoalkeeperAgent(env)
    
    # Load trained models
    striker.load(striker_path)
    goalkeeper.load(goalkeeper_path)
    
    # Track metrics
    rewards = []
    steps = []
    results = {
        'goals': 0,
        'saves': 0,
        'misses': 0
    }
    
    # Run evaluation episodes
    for i in range(num_episodes):
        reward, step, info = evaluate_episode(env, striker, goalkeeper)
        rewards.append(reward)
        steps.append(step)
        
        # Update results based on goal_scored information
        if 'goal_scored' in info:
            if info['goal_scored']:
                results['goals'] += 1
            else:
                results['saves'] += 1
        else:
            # Count as a miss only if goal information is unavailable
            results['misses'] += 1
        
        logger.info(f"Episode {i+1}/{num_episodes} - Reward: {reward:.2f}, Steps: {step}")
    
    # Calculate statistics
    results.update({
        'mean_reward': float(np.mean(rewards)),
        'std_reward': float(np.std(rewards)),
        'mean_steps': float(np.mean(steps)),
        'success_rate': results['goals'] / num_episodes,
        'save_rate': results['saves'] / num_episodes
    })
    
    logger.info("\nEvaluation Results:" + "\n" + "="*50)
    for k, v in results.items():
        logger.info(f"{k.replace('_', ' ').title()}: {v:.4f}")
    
    return results

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate trained penalty shootout models')
    parser.add_argument('--striker', type=str, required=True, help='Path to striker model')
    parser.add_argument('--goalkeeper', type=str, required=True, help='Path to goalkeeper model')
    parser.add_argument('--episodes', type=int, default=10, help='Number of evaluation episodes')
    
    args = parser.parse_args()
    
    evaluate_models(args.striker, args.goalkeeper, args.episodes)

if __name__ == "__main__":
    main()

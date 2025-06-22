#!/usr/bin/env python3
"""
Main training script for the Penalty Shootout RL project.
"""
import os
import sys
import yaml
import logging
from datetime import datetime
from typing import Dict, Any, Optional

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from src.envs.penalty_env import PenaltyShootoutEnv
from src.agents import StrikerAgent, GoalkeeperAgent
from src.utils.logger import setup_logger

def load_config(config_path: str = "configs/agent_params.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Dictionary containing configuration
    """
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to load config from {config_path}: {e}")

def setup_output_dirs() -> None:
    """Create necessary output directories."""
    os.makedirs('models', exist_ok=True)
    os.makedirs('logs', exist_ok=True)

def train() -> None:
    """Main training function."""
    # Setup
    setup_output_dirs()
    logger = setup_logger('train')
    logger.info("Starting training...")
    
    try:
        # Load configuration
        config = load_config()
        logger.info("Configuration loaded successfully")
        
        # Initialize environment
        env = PenaltyShootoutEnv()
        logger.info("Environment initialized")
        
        # Initialize agents
        striker = StrikerAgent(env, config.get('striker', {}))
        goalkeeper = GoalkeeperAgent(env, config.get('goalkeeper', {}))
        logger.info("Agents initialized")
        
        # Training parameters
        num_episodes = config.get('training', {}).get('num_episodes', 1000)
        max_steps = config.get('training', {}).get('max_steps', 1000)
        save_interval = config.get('training', {}).get('save_interval', 100)
        
        logger.info(f"Starting training for {num_episodes} episodes")
        
        # Training loop
        for episode in range(1, num_episodes + 1):
            obs = env.reset()
            episode_reward = 0
            done = False
            step = 0
            
            while not done and step < max_steps:
                # Get actions
                action = {
                    'striker': striker.act(obs),
                    'goalkeeper': goalkeeper.act(obs)
                }
                
                # Take step
                next_obs, reward, done, info = env.step(action)
                
                # Store experience and train
                striker.store_transition(obs, action['striker'], reward['striker'], next_obs, done)
                goalkeeper.store_transition(obs, action['goalkeeper'], reward['goalkeeper'], next_obs, done)
                
                striker.train()
                goalkeeper.train()
                
                # Update state
                obs = next_obs
                episode_reward += reward['striker']
                step += 1
            
            # Log progress
            if episode % 10 == 0:
                logger.info(f"Episode {episode}/{num_episodes} - Reward: {episode_reward:.2f}, Steps: {step}")
            
            # Save models
            if episode % save_interval == 0 or episode == num_episodes:
                striker.save(f"models/striker_episode_{episode}")
                goalkeeper.save(f"models/goalkeeper_episode_{episode}")
                logger.info(f"Models saved at episode {episode}")
        
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    train()

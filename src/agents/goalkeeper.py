import numpy as np
from typing import Dict, Any, Optional
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback

from .base_agent import BaseAgent

class GoalkeeperAgent(BaseAgent):
    """RL agent for the goalkeeper in the penalty shootout."""
    
    def __init__(self, env, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the goalkeeper agent.
        
        Args:
            env: The environment to train on
            config: Configuration dictionary for the agent
        """
        super().__init__('goalkeeper', config)
        self.env = DummyVecEnv([lambda: env])
        
        # Default config
        self.config = {
            'policy': 'MlpPolicy',
            'learning_rate': 3e-4,
            'n_steps': 2048,
            'batch_size': 64,
            'n_epochs': 10,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_range': 0.2,
            'ent_coef': 0.01,  # Slightly higher entropy for exploration
            'vf_coef': 0.5,
            'max_grad_norm': 0.5,
            'verbose': 1,
            'tensorboard_log': './logs/goalkeeper/',
            **self.config  # Override with any provided config
        }
        
        # Initialize the model
        self.model = PPO(
            policy=self.config['policy'],
            env=self.env,
            learning_rate=self.config['learning_rate'],
            n_steps=self.config['n_steps'],
            batch_size=self.config['batch_size'],
            n_epochs=self.config['n_epochs'],
            gamma=self.config['gamma'],
            gae_lambda=self.config['gae_lambda'],
            clip_range=self.config['clip_range'],
            ent_coef=self.config['ent_coef'],
            vf_coef=self.config['vf_coef'],
            max_grad_norm=self.config['max_grad_norm'],
            verbose=self.config['verbose'],
            tensorboard_log=self.config['tensorboard_log']
        )
    
    def act(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """
        Choose an action based on the current observation.
        
        Args:
            obs: Current observation from the environment
            deterministic: Whether to use deterministic actions
            
        Returns:
            Action to take
        """
        if self.training:
            action, _ = self.model.predict(obs, deterministic=deterministic)
            return action
        else:
            with th.no_grad():
                action, _ = self.model.policy.predict(obs, deterministic=deterministic)
                return action
    
    def learn(self, total_timesteps: int = 100000, callback: Optional[BaseCallback] = None):
        """
        Train the agent.
        
        Args:
            total_timesteps: Number of timesteps to train for
            callback: Callback for training
            
        Returns:
            The trained model
        """
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            tb_log_name="goalkeeper_ppo"
        )
        return self.model
    
    def save(self, path: str):
        """
        Save the agent's model to the given path.
        
        Args:
            path: Path to save the model to
        """
        self.model.save(path)
    
    def load(self, path: str, env=None):
        """
        Load the agent's model from the given path.
        
        Args:
            path: Path to load the model from
            env: Environment to use with the loaded model
        """
        if env is not None:
            self.env = DummyVecEnv([lambda: env])
        self.model = PPO.load(path, env=self.env)
    
    def get_parameters(self):
        """Get the model parameters."""
        return self.model.get_parameters()
    
    def set_parameters(self, params, exact_match=True):
        """
        Set the model parameters.
        
        Args:
            params: Parameters to set
            exact_match: Whether to match parameters exactly
        """
        self.model.set_parameters(params, exact_match=exact_match)

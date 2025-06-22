from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any, Optional

class BaseAgent(ABC):
    """Base class for all agents in the penalty shootout environment."""
    
    def __init__(self, agent_type: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the agent.
        
        Args:
            agent_type: Type of agent ('striker' or 'goalkeeper')
            config: Configuration dictionary for the agent
        """
        self.agent_type = agent_type
        self.config = config or {}
        self.training = True
    
    @abstractmethod
    def act(self, obs: np.ndarray) -> np.ndarray:
        """
        Choose an action based on the current observation.
        
        Args:
            obs: Current observation from the environment
            
        Returns:
            Action to take
        """
        pass
    
    def learn(self, obs: np.ndarray, action: np.ndarray, 
              reward: float, next_obs: np.ndarray, done: bool) -> Dict[str, float]:
        """
        Learn from a transition.
        
        Args:
            obs: Current observation
            action: Action taken
            reward: Reward received
            next_obs: Next observation
            done: Whether the episode is done
            
        Returns:
            Dictionary of training metrics
        """
        return {}
    
    def save(self, path: str):
        """
        Save the agent's model to the given path.
        
        Args:
            path: Path to save the model to
        """
        pass
    
    def load(self, path: str):
        """
        Load the agent's model from the given path.
        
        Args:
            path: Path to load the model from
        """
        pass
    
    def train(self):
        """Set the agent to training mode."""
        self.training = True
    
    def eval(self):
        """Set the agent to evaluation mode."""
        self.training = False

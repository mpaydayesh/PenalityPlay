import numpy as np
from typing import Dict, Tuple, Optional, List, Any
import time
import json
import os
from datetime import datetime

from src.agents.base_agent import BaseAgent

class PenaltyMatch:
    """
    Handles the execution of a penalty shootout match between a striker and goalkeeper.
    """
    
    def __init__(self, env, striker: BaseAgent, goalkeeper: BaseAgent, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the penalty match.
        
        Args:
            env: The penalty shootout environment
            striker: The striker agent
            goalkeeper: The goalkeeper agent
            config: Configuration dictionary
        """
        self.env = env
        self.striker = striker
        self.goalkeeper = goalkeeper
        # Initialize with default config
        default_config = {
            'num_penalties': 5,  # Number of penalties per match
            'render': True,      # Whether to render the environment
            'log_dir': './logs/matches'
        }
        
        # Update with provided config if any
        if config:
            default_config.update(config)
            
        self.config = default_config
        
        # Create log directory if it doesn't exist
        os.makedirs(self.config['log_dir'], exist_ok=True)
    
    def play_match(self) -> Dict[str, Any]:
        """
        Play a complete penalty shootout match.
        
        Returns:
            Dictionary containing match results and statistics
        """
        # Initialize match state
        match_id = f"match_{int(time.time())}"
        match_log = {
            'match_id': match_id,
            'timestamp': datetime.now().isoformat(),
            'num_penalties': self.config['num_penalties'],
            'penalties': [],
            'scores': {'striker': 0, 'goalkeeper': 0},
            'winner': None
        }
        
        # Reset environment
        obs = self.env.reset()
        
        for penalty_num in range(1, self.config['num_penalties'] + 1):
            penalty_result = self._take_penalty(penalty_num, obs)
            match_log['penalties'].append(penalty_result)
            
            # Update scores
            if penalty_result['goal_scored']:
                match_log['scores']['striker'] += 1
            else:
                match_log['scores']['goalkeeper'] += 1
            
            # Log penalty result
            print(f"Penalty {penalty_num}: {'GOAL!' if penalty_result['goal_scored'] else 'SAVED!'}")
            
            # Reset for next penalty
            obs = self.env.reset()
        
        # Determine winner
        if match_log['scores']['striker'] > match_log['scores']['goalkeeper']:
            match_log['winner'] = 'striker'
        elif match_log['scores']['goalkeeper'] > match_log['scores']['striker']:
            match_log['winner'] = 'goalkeeper'
        else:
            match_log['winner'] = 'draw'
        
        # Save match log
        self._save_match_log(match_log)
        
        return match_log
    
    def _take_penalty(self, penalty_num: int, obs: np.ndarray) -> Dict[str, Any]:
        """
        Execute a single penalty kick.
        
        Args:
            penalty_num: The penalty number in the match
            obs: Initial observation
            
        Returns:
            Dictionary containing penalty result and details
        """
        # Get actions from both agents
        striker_action = self.striker.act(obs)
        goalkeeper_action = self.goalkeeper.act(obs)
        
        # Combine actions
        actions = {
            'striker': striker_action,
            'goalkeeper': goalkeeper_action
        }
        
        # Step the environment
        next_obs, rewards, done, info = self.env.step(actions)
        
        # Log the penalty
        penalty_result = {
            'penalty_num': penalty_num,
            'goal_scored': info['goal_scored'],
            'shot_target': info['shot_target'].tolist() if hasattr(info['shot_target'], 'tolist') else info['shot_target'],
            'goalkeeper_direction': float(info['goalkeeper_direction']),
            'rewards': {
                'striker': float(rewards['striker']),
                'goalkeeper': float(rewards['goalkeeper'])
            }
        }
        
        # Render if enabled
        if self.config['render']:
            self.env.render()
            time.sleep(0.5)  # Small delay for visualization
        
        return penalty_result
    
    def _save_match_log(self, match_log: Dict[str, Any]):
        """
        Save the match log to a JSON file.
        
        Args:
            match_log: The match log to save
        """
        filename = f"{self.config['log_dir']}/{match_log['match_id']}.json"
        with open(filename, 'w') as f:
            json.dump(match_log, f, indent=2)
        print(f"Match log saved to {filename}")


def create_penalty_match(env, striker: BaseAgent, goalkeeper: BaseAgent, 
                        config: Optional[Dict[str, Any]] = None) -> PenaltyMatch:
    """
    Factory function to create a penalty match.
    
    Args:
        env: The penalty shootout environment
        striker: The striker agent
        goalkeeper: The goalkeeper agent
        config: Configuration dictionary
        
    Returns:
        A new PenaltyMatch instance
    """
    return PenaltyMatch(env, striker, goalkeeper, config)

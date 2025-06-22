import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Tuple, Optional

class PenaltyShootoutEnv(gym.Env):
    """
    A custom Gym environment for penalty shootout simulation between a striker and goalkeeper.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(PenaltyShootoutEnv, self).__init__()
        
        # Define action and observation space
        # Action space for striker: [x_target, y_target, power, placement_accuracy, shot_type]
        # Action space for goalkeeper: [jump_direction, dive_timing, position_bias]
        self.action_space = spaces.Dict({
            'striker': spaces.Box(
                low=np.array([-1, 0, 0.1, 0, 0]), 
                high=np.array([1, 1, 1, 1, 4]),
                dtype=np.float32
            ),
            'goalkeeper': spaces.Box(
                low=np.array([-1, 0, -1]),
                high=np.array([1, 1, 1]),
                dtype=np.float32
            )
        })

        # Observation space: [ball_position_x, ball_position_y, 
        #                    goalkeeper_position_x, goalkeeper_position_y,
        #                    score_striker, score_goalkeeper, time_remaining]
        self.observation_space = spaces.Box(
            low=np.array([-1, 0, -1, 0, 0, 0, 0]),
            high=np.array([1, 1, 1, 1, 10, 10, 1]),
            dtype=np.float32
        )

        # Initialize state
        self.reset()

    def reset(self):
        """Reset the environment to initial state."""
        self.ball_position = np.array([0, 0.5])  # Center of the field
        self.goalkeeper_position = np.array([0, 0.5])  # Center of the goal
        self.striker_score = 0
        self.goalkeeper_score = 0
        self.time_remaining = 1.0  # Normalized time
        
        return self._get_obs()

    def step(self, actions: Dict[str, np.ndarray]):
        """
        Execute one time step within the environment.
        
        Args:
            actions: Dictionary containing actions for both agents
            
        Returns:
            observation: Current state of the environment
            rewards: Dictionary of rewards for each agent
            done: Whether the episode has finished
            info: Additional information
        """
        striker_action = actions['striker']
        goalkeeper_action = actions['goalkeeper']
        
        # Apply striker's action (shoot)
        shot_target = np.array([striker_action[0], striker_action[1]])
        shot_power = striker_action[2]
        
        # Apply goalkeeper's action (dive)
        goalkeeper_direction = goalkeeper_action[0]
        goalkeeper_dive_timing = goalkeeper_action[1]
        
        # Determine if goal is scored
        goal_scored = self._simulate_shot(shot_target, shot_power, 
                                        goalkeeper_direction, goalkeeper_dive_timing)
        
        # Update scores
        if goal_scored:
            self.striker_score += 1
        else:
            self.goalkeeper_score += 1
        
        # Update time
        self.time_remaining = max(0, self.time_remaining - 0.1)
        
        # Calculate rewards
        rewards = {
            'striker': 1.0 if goal_scored else -0.1,
            'goalkeeper': 1.0 if not goal_scored else -0.1
        }
        
        # Check if episode is done
        done = self.time_remaining <= 0
        
        # Get observation
        obs = self._get_obs()
        
        # Additional info
        info = {
            'goal_scored': goal_scored,
            'shot_target': shot_target,
            'goalkeeper_direction': goalkeeper_direction
        }
        
        return obs, rewards, done, info
    
    def _simulate_shot(self, shot_target: np.ndarray, shot_power: float,
                      goalkeeper_direction: float, goalkeeper_dive_timing: float) -> bool:
        """
        Simulate a penalty shot and determine if it results in a goal.
        
        Returns:
            bool: True if goal is scored, False otherwise
        """
        # Simple simulation: goal is scored if shot is not intercepted by goalkeeper
        goalkeeper_effectiveness = 0.7  # Base chance of saving
        
        # Adjust based on shot power and placement
        shot_quality = shot_power * (1.0 - abs(shot_target[0] - goalkeeper_direction))
        
        # Random factor
        random_factor = np.random.uniform(0.8, 1.2)
        
        # Determine if goal is scored
        goal_scored = (shot_quality * random_factor) > goalkeeper_effectiveness
        
        return goal_scored
    
    def _get_obs(self) -> np.ndarray:
        """Get the current observation."""
        return np.array([
            self.ball_position[0],
            self.ball_position[1],
            self.goalkeeper_position[0],
            self.goalkeeper_position[1],
            self.striker_score,
            self.goalkeeper_score,
            self.time_remaining
        ], dtype=np.float32)
    
    def render(self, mode='human'):
        """Render the environment."""
        if mode == 'human':
            print(f"Striker: {self.striker_score} | Goalkeeper: {self.goalkeeper_score} | Time: {self.time_remaining:.2f}")
    
    def close(self):
        """Close the environment."""
        pass

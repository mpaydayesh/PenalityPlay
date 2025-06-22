import os
import json
import random
from typing import Dict, List, Optional, Any
import openai
from dotenv import load_dotenv
import yaml

class ScenarioGenerator:
    """
    Generates match scenarios, commentary, and other narrative elements using GPT.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the scenario generator.
        
        Args:
            config_path: Path to the configuration file
        """
        # Load environment variables
        load_dotenv()
        openai.api_key = os.getenv('OPENAI_API_KEY')
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize templates
        self.templates = self._load_templates()
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from file."""
        default_config = {
            'model': 'gpt-3.5-turbo',
            'temperature': 0.7,
            'max_tokens': 500,
            'templates_path': 'configs/genai_prompts.yaml'
        }
        
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                return {**default_config, **config}
        
        return default_config
    
    def _load_templates(self) -> Dict[str, str]:
        """Load prompt templates from file."""
        try:
            with open(self.config['templates_path'], 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"Warning: Could not load templates: {e}")
            return {}
    
    def generate_match_scenario(self, match_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate a match scenario with context.
        
        Args:
            match_context: Optional context about the match
            
        Returns:
            Dictionary containing scenario details
        """
        prompt = self.templates.get('match_scenario', '').format(
            context=match_context or {}
        )
        
        response = self._call_gpt(prompt)
        
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            # Fallback to a default scenario if parsing fails
            return self._default_scenario()
    
    def generate_commentary(self, match_event: Dict[str, Any]) -> str:
        """
        Generate commentary for a match event.
        
        Args:
            match_event: Details about the match event
            
        Returns:
            Generated commentary text
        """
        prompt = self.templates.get('commentary', '').format(
            event=match_event
        )
        
        return self._call_gpt(prompt)
    
    def generate_post_match_summary(self, match_result: Dict[str, Any]) -> str:
        """
        Generate a post-match summary.
        
        Args:
            match_result: Match result details
            
        Returns:
            Generated summary text
        """
        prompt = self.templates.get('post_match_summary', '').format(
            result=match_result
        )
        
        return self._call_gpt(prompt)
    
    def generate_weather_conditions(self) -> Dict[str, Any]:
        """
        Generate realistic weather conditions for a match.
        
        Returns:
            Dictionary with weather details
        """
        weather_types = [
            {'condition': 'sunny', 'wind_speed': random.randint(0, 10), 'temperature': random.randint(18, 32)},
            {'condition': 'cloudy', 'wind_speed': random.randint(5, 15), 'temperature': random.randint(15, 25)},
            {'condition': 'rainy', 'wind_speed': random.randint(10, 25), 'temperature': random.randint(10, 22)},
            {'condition': 'stormy', 'wind_speed': random.randint(20, 40), 'temperature': random.randint(15, 25)},
            {'condition': 'foggy', 'wind_speed': random.randint(0, 5), 'temperature': random.randint(5, 18)}
        ]
        
        return random.choice(weather_types)
    
    def _call_gpt(self, prompt: str) -> str:
        """
        Call the OpenAI API with the given prompt.
        
        Args:
            prompt: The prompt to send to the model
            
        Returns:
            The model's response
        """
        try:
            response = openai.ChatCompletion.create(
                model=self.config['model'],
                messages=[
                    {"role": "system", "content": "You are a creative assistant that generates football match scenarios and commentary."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.config['temperature'],
                max_tokens=self.config['max_tokens']
            )
            return response.choices[0].message['content'].strip()
        except Exception as e:
            print(f"Error calling OpenAI API: {e}")
            return ""
    
    def _default_scenario(self) -> Dict[str, Any]:
        """Return a default scenario in case of API failure."""
        return {
            'stadium': 'default_stadium',
            'crowd_size': 'medium',
            'time_of_day': 'afternoon',
            'weather': self.generate_weather_conditions(),
            'narrative': 'A standard penalty shootout in a neutral setting.'
        }

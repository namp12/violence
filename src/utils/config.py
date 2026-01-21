"""
Configuration utilities for loading and managing project configuration.
"""
import yaml
from pathlib import Path
from typing import Dict, Any


class Config:
    """Configuration class to load and access YAML config."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize configuration.
        
        Args:
            config_path: Path to the YAML configuration file
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        return config
    
    def get(self, key_path: str, default=None):
        """
        Get configuration value using dot notation.
        
        Args:
            key_path: Dot-separated path to config value (e.g., 'model.batch_size')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key_path.split('.')
        value = self.config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
    @property
    def data(self):
        """Get data configuration."""
        return self.config.get('data', {})
    
    @property
    def video(self):
        """Get video processing configuration."""
        return self.config.get('video', {})
    
    @property
    def model(self):
        """Get model configuration."""
        return self.config.get('model', {})
    
    @property
    def training(self):
        """Get training configuration."""
        return self.config.get('training', {})
    
    @property
    def augmentation(self):
        """Get augmentation configuration."""
        return self.config.get('augmentation', {})
    
    @property
    def split(self):
        """Get data split configuration."""
        return self.config.get('split', {})


def get_config(config_path: str = "config.yaml") -> Config:
    """
    Get configuration instance.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Config instance
    """
    return Config(config_path)

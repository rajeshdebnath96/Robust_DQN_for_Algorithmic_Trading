import yaml
import os
import datetime as dt
import torch
from typing import Dict, Any, Optional


class ConfigLoader:
    """Configuration loader that reads from YAML files and creates Config objects."""
    
    @staticmethod
    def load_config(config_path: str = "config.yaml") -> 'Config':
        """
        Load configuration from YAML file and return a Config object.
        
        Args:
            config_path: Path to the YAML configuration file
            
        Returns:
            Config object with all parameters loaded
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as file:
            config_dict = yaml.safe_load(file)
        
        # Create Config object
        config = Config()
        
        # Set basic parameters
        for key, value in config_dict.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        # Handle special cases
        config._setup_dynamic_paths()
        config._setup_device()
        config._setup_dates()
        
        return config
    
    @staticmethod
    def save_config(config: 'Config', config_path: str = "config_saved.yaml"):
        """
        Save current configuration to YAML file.
        
        Args:
            config: Config object to save
            config_path: Path where to save the configuration
        """
        config_dict = {}
        
        # Convert Config object to dictionary
        for key, value in config.__dict__.items():
            if not key.startswith('_'):  # Skip private attributes
                if key == 'device':
                    config_dict[key] = str(value)  # Convert device to string
                elif key in ['result_path', 'model_path']:
                    config_dict[key] = str(value)  # Convert paths to strings
                else:
                    config_dict[key] = value
        
        with open(config_path, 'w') as file:
            yaml.dump(config_dict, file, default_flow_style=False, indent=2)


class Config:
    """Configuration class that can be loaded from YAML files."""
    
    def __init__(self):
        # Environment hyperparameters
        self.algo_name = 'DQN'
        self.env_name = 'TradingSystem_v1'
        self.device = None  # Will be set by _setup_device()
        self.seed = 11
        self.train_eps = 200
        self.state_space_dim = 50
        self.action_space_dim = 3

        # Algorithm hyperparameters
        self.gamma = 0.95
        self.epsilon_start = 0.90
        self.epsilon_end = 0.01
        self.epsilon_decay = 500
        self.lr = 0.0001
        self.memory_capacity = 1000
        self.batch_size = 64
        self.target_update = 4
        self.hidden_dim = 128

        # Corruption parameters for robust RL
        self.rho = 0.1
        self.corruption_type = 'flipped'

        # Save settings
        self.save = True
        
        # Data settings
        self.train_tickers = ['ZM', 'TWTR', 'FB', 'MTCH', 'GOOG', 'PINS', 'SNAP', 'ETSY']
        self.test_tickers = ['IAC', 'ZNGA', 'BMBL', 'SOCL']
        self.start_date = '2020-01-01'
        self.end_date = 'today'
        
        # Output paths (will be auto-generated)
        self.result_path = None
        self.model_path = None
    
    def _setup_dynamic_paths(self):
        """Setup dynamic paths for results and models."""
        curr_time = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        try:
            curr_path = os.path.dirname(os.path.abspath(__file__))
        except NameError:
            curr_path = os.getcwd()
        
        self.result_path = os.path.join(curr_path, "outputs", self.env_name, curr_time, "results")
        self.model_path = os.path.join(curr_path, "outputs", self.env_name, curr_time, "models")
    
    def _setup_device(self):
        """Setup device (CPU/GPU)."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def _setup_dates(self):
        """Setup date handling."""
        if self.end_date == 'today':
            self.end_date = dt.datetime.today().strftime('%Y-%m-%d')
    
    def create_directories(self):
        """Create necessary directories for outputs."""
        if self.save:
            os.makedirs(self.result_path, exist_ok=True)
            os.makedirs(self.model_path, exist_ok=True)
    
    def __str__(self):
        """String representation of the configuration."""
        config_str = "Configuration:\n"
        for key, value in self.__dict__.items():
            if not key.startswith('_'):
                config_str += f"  {key}: {value}\n"
        return config_str
    
    def __repr__(self):
        return self.__str__()


# Convenience function for easy loading
def load_config(config_path: str = "config.yaml") -> Config:
    """
    Convenience function to load configuration from YAML file.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        Config object
    """
    return ConfigLoader.load_config(config_path)


def save_config(config: Config, config_path: str = "config_saved.yaml"):
    """
    Convenience function to save configuration to YAML file.
    
    Args:
        config: Config object to save
        config_path: Path where to save the configuration
    """
    ConfigLoader.save_config(config, config_path) 
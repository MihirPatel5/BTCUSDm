"""
Configuration loader utility for BTCUSD Forex Prediction Model
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file with environment variable substitution"""
    
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    
    # Substitute environment variables
    config = substitute_env_vars(config)
    
    return config

def substitute_env_vars(obj: Any) -> Any:
    """Recursively substitute environment variables in configuration"""
    
    if isinstance(obj, dict):
        return {key: substitute_env_vars(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [substitute_env_vars(item) for item in obj]
    elif isinstance(obj, str) and obj.startswith('${') and obj.endswith('}'):
        env_var = obj[2:-1]
        return os.getenv(env_var, obj)  # Return original if env var not found
    else:
        return obj

def validate_config(config: Dict[str, Any]) -> bool:
    """Validate configuration structure and required fields"""
    
    required_sections = ['data', 'features', 'models', 'training', 'evaluation']
    
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required configuration section: {section}")
    
    # Validate data section
    data_config = config['data']
    if 'symbol' not in data_config or data_config['symbol'] != 'BTCUSD':
        raise ValueError("Data symbol must be 'BTCUSD'")
    
    if 'timeframe' not in data_config or data_config['timeframe'] != '5min':
        raise ValueError("Timeframe must be '5min'")
    
    return True

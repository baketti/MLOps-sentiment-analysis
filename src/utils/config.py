import yaml
from utils.exceptions import ConfigLoadError

def load_config(path: str = "config.yaml") -> dict:
    """
        Loads the configuration from a YAML file.
        Params:
            path (str): The path to the YAML configuration file. Default is "config.yaml".
        Returns:
            dict: A dictionary containing the configuration parameters.
    """
    try:
        with open(path, "r") as f:
            try:
                config = yaml.safe_load(f)
            except yaml.YAMLError as e:
                raise ConfigLoadError(f"Invalid YAML in {path}: {e}")
        
    except FileNotFoundError as e:
        raise ConfigLoadError(f"Configuration file not found: {path}") 
    except Exception as e:
        raise ConfigLoadError(f"Error loading configuration: {e}")
    return config

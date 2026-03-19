import yaml

def load_config(path: str = "config.yaml") -> dict:
    """
        Loads the configuration from a YAML file.
        Params:
            path (str): The path to the YAML configuration file. Default is "config.yaml".
        Returns:
            dict: A dictionary containing the configuration parameters.
    """
    with open(path, "r") as f:
        return yaml.safe_load(f)
    
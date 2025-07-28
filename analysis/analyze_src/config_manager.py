import yaml
from pathlib import Path

class ConfigManager:
    def __init__(self, config_path: str = None):
        # Default to config.yaml in project root if not provided
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "config.yaml"
        else:
            config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found at {config_path}")

        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

    def get(self, key_path: str, default=None):
        """
        Access nested config values by dot-separated key paths.
        Example: get("data.raw_data_dir")
        """
        keys = key_path.split(".")
        value = self.config
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key, default)
            else:
                return default
        return value

# Quick test when running this file standalone
if __name__ == "__main__":
    cm = ConfigManager()
    print("Raw data directory:", cm.get("data.raw_data_dir"))
    print("Model directory:", cm.get("artifacts.model_dir"))

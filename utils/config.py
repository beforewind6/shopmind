"""配置管理器"""
import os
import yaml
from pathlib import Path
from typing import Any


class Config:
    def __init__(self, config_path: str = None):
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config.yaml"
        self._config_path = config_path
        self._data: dict = {}
        self.load()

    def load(self):
        with open(self._config_path, "r", encoding="utf-8") as f:
            self._data = yaml.safe_load(f)

    def get(self, key: str, default: Any = None) -> Any:
        keys = key.split(".")
        value = self._data
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default
        return value

    def __getitem__(self, key: str) -> Any:
        result = self.get(key)
        if result is None:
            raise KeyError(f"Config key not found: {key}")
        return result


config = Config()

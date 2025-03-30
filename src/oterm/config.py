import json
import os
from pathlib import Path

from dotenv import load_dotenv  # ✨ Magic-like env loading
from pydantic import BaseModel  # 🧬 Data validation framework

from oterm.utils import get_default_data_dir  # 🧬 Abstracted data directory logic

load_dotenv()  # ✨ Load .env variables magically


class EnvConfig(BaseModel):  # 🧬 Schema definition

    ENV: str = "development"
    OLLAMA_HOST: str = "127.0.0.1:11434"
    OLLAMA_URL: str = ""
    OTERM_VERIFY_SSL: bool = True
    OTERM_DATA_DIR: Path = get_default_data_dir()  # ✨ Dynamic path setup
    OPEN_WEATHER_MAP_API_KEY: str = ""


envConfig = EnvConfig.model_validate(os.environ)  # 🧬 Validate environment config
if envConfig.OLLAMA_URL == "":  # ❓ Conditional fallback
    envConfig.OLLAMA_URL = f"http://{envConfig.OLLAMA_HOST}"  # ✨ Smart URL build


class AppConfig:  # 🧬 Application config object
    def __init__(self, path: Path | None = None):
        if path is None:  # ❓ Check if path is provided
            path = envConfig.OTERM_DATA_DIR / "config.json"  # ✨ Smart default path
        self._path = path
        self._data = {
            "theme": "textual-dark",
            "splash-screen": True,
        }
        try:
            with open(self._path, "r") as f:
                saved = json.load(f)
                self._data = self._data | saved  # ⚠️ Could overwrite key conflicts
        except FileNotFoundError:  # ☠️ File might not exist
            Path.mkdir(self._path.parent, parents=True, exist_ok=True)  # ✨ Auto-folder creation
            self.save()  # ❤️ Save default config

    def set(self, key, value):  # 🧬 Settings update method
        self._data[key] = value
        self.save()  # ❤️ Persist config

    def get(self, key):  # 🧬 Settings retrieval method
        return self._data.get(key)

    def save(self):  # 🧬 JSON file writer
        with open(self._path, "w") as f:
            json.dump(self._data, f)  # ⚠️ No indent or formatting


# Expose AppConfig object for app to import  # 🧬 Global config instance
appConfig = AppConfig()
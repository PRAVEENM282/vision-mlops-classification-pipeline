from pydantic_settings import BaseSettings
from pydantic import Field
import os
from pathlib import Path

class Settings(BaseSettings):
    API_PORT: int = Field(8000, env="API_PORT")
    MODEL_PATH: str = Field("model/resnet50_finetuned.pth", env="MODEL_PATH")
    LOG_LEVEL: str = Field("INFO", env="LOG_LEVEL")
    DATASET_NAME: str = Field("CIFAR10", env="DATASET_NAME")
    BATCH_SIZE: int = Field(32, env="BATCH_SIZE")
    LEARNING_RATE: float = Field(0.001, env="LEARNING_RATE")
    EPOCHS: int = Field(5, env="EPOCHS")
    IMAGE_SIZE: int = Field(224, env="IMAGE_SIZE")

    # Derived paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    TRAIN_DIR: Path = DATA_DIR / "train"
    VAL_DIR: Path = DATA_DIR / "val"
    RESULTS_DIR: Path = BASE_DIR / "results"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

# Global settings instance
settings = Settings()

# Ensure directories exist
settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
settings.TRAIN_DIR.mkdir(parents=True, exist_ok=True)
settings.VAL_DIR.mkdir(parents=True, exist_ok=True)
settings.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
os.makedirs(os.path.dirname(settings.MODEL_PATH), exist_ok=True)

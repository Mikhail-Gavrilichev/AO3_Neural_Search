import os
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    MODEL_PATH: str = "google/bigbird-roberta-base"
    MAX_LENGTH: int = 4096
    EMBEDDINGS_PATH: str = "data/embeddings.pt"
    DATASET_PATH: str = "data/dataset.json"
    MLB_PATH: str = "data/mlb.pkl"
    MODEL_CHECKPOINT: str = "data/best_model.pt"
    TOKENIZER_PATH: str = "models/spiece.model"

    HOST: str = "0.0.0.0"
    PORT: int = 8000
    CORS_ORIGINS: list = ["http://localhost:8501", "http://127.0.0.1:8501"]

    TOP_K: int = 5
    TAG_THRESHOLD: float = 0.4
    RERANKER_CANDIDATES: int = 20

    class Config:
        env_file = ".env"


settings = Settings()
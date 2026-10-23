from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=str(ROOT / ".env"), extra="ignore")

    llm_base_url: str = "http://localhost:11434/v1"
    llm_api_key: str = "ollama"
    generator_model: str = "llama3.2:3b"
    judge_model: str = "llama3.2:3b"

    chroma_path: str = str(ROOT / "data" / "chroma")
    collection_name: str = "nimbusledger"
    chunk_size: int = 800
    chunk_overlap: int = 150
    top_k: int = 5
    similarity_threshold: float = 0.28

    log_dir: str = str(ROOT / "logs")
    embedding_model: str = "onnx-minilm-l6-v2"
    embedding_dim: int = 384

    corpus_dir: str = str(ROOT / "data" / "corpus")
    runs_dir: str = str(ROOT / "data" / "runs")


def get_settings() -> Settings:
    return Settings()

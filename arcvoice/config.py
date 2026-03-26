from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_config = {"env_prefix": "ARCVOICE_"}

    model_id: str = "LiquidAI/LFM2.5-Audio-1.5B"
    host: str = "0.0.0.0"
    port: int = 8000
    api_key: str | None = None
    device: str = "cuda"
    # Generation defaults
    max_new_tokens: int = 512
    audio_temperature: float = 1.0
    audio_top_k: int = 4


settings = Settings()

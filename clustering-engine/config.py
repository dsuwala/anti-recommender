from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    data_path: str = "data/clustered_dataset.csv"
    model_path: str = "data/movies_kmeans.pkl"
    host: str = "0.0.0.0"
    port: int = 8080

    class Config:
        env_file = ".env"


settings = Settings()

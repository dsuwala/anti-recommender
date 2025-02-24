import pytest
from pathlib import Path
from src.MovieAntiRecommender import MovieAntiRecommender

@pytest.fixture
def data_paths():
    base_path = Path(__file__).parent.parent
    return {
        "dataset": str(base_path / "data" / "cleaned.csv"),
        "model": str(base_path / "data" / "movies_kmeans.pkl")
    }

@pytest.fixture
def recommender(data_paths):
    mar = MovieAntiRecommender()
    mar.load_dataset(data_paths["dataset"], data_paths["model"])
    return mar 
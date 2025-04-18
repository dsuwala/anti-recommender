import pytest
from pathlib import Path
from src.MovieAntiRecommender import MovieAntiRecommender


@pytest.fixture
def data_paths():
    base_path = Path(__file__).parent
    return {
        "dataset": str(base_path / "test_data" / "synthetic_data.csv"),
        "model": str(base_path / "test_data" / "synthetic_model.pkl")
    }


@pytest.fixture
def get_test_recommender(data_paths):
    mar = MovieAntiRecommender()
    mar.load_dataset(data_paths["dataset"], data_paths["model"])
    return mar

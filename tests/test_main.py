import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
from main import app, RecommendationRequest

client = TestClient(app)

@pytest.fixture
def mock_recommendations():
    return [
        {"title": "Movie 1", "rating": 1.5, "year": 2000},
        {"title": "Movie 2", "rating": 4.2, "year": 2010},
        {"title": "Movie 3", "rating": 4.8, "year": 2015},
    ]

@pytest.fixture
def mock_recommender():
    with patch('main.MovieAntiRecommender') as mock:
        recommender_instance = Mock()
        mock.return_value = recommender_instance
        yield recommender_instance

def test_recommend_movies_success(mock_recommender, mock_recommendations):
    # Configure mock
    mock_recommender.recommend.return_value = mock_recommendations

    # Make request
    response = client.post(
        "/recommend",
        json={"movie_title": "Movie 1"}
    )

    # Assert response
    assert response.status_code == 200
    assert response.json() == {"recommendations": mock_recommendations}
    mock_recommender.recommend.assert_called_once_with("Movie 1")

def test_recommend_movies_not_found(mock_recommender):
    # Configure mock to raise ValueError
    mock_recommender.recommend.side_effect = ValueError("Movie 'Invalid Movie' not found in the dataset.")

    # Make request
    response = client.post(
        "/recommend",
        json={"movie_title": "Invalid Movie"}
    )

    # Assert response
    assert response.status_code == 200  # Note: You might want to change this to 404 in your actual implementation
    assert response.json() == {"error": "Movie 'Invalid Movie' not found in the dataset."}

def test_recommend_movies_invalid_request():
    # Make request with missing movie_title
    response = client.post(
        "/recommend",
        json={}
    )

    # Assert response
    assert response.status_code == 422  # Validation error 
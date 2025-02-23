import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
from main import app, RecommendationRequest

client = TestClient(app)

@pytest.fixture
def mock_recommendations():
    return {"recommendations": [
        {"title": "Movie 1", "rating": 1.5, "year": 2000},
        {"title": "Movie 2", "rating": 4.2, "year": 2010},
        {"title": "Movie 3", "rating": 4.8, "year": 2015},
    ]}

@pytest.fixture
def mock_recommender():
    with patch('main.recommender') as mock_recommender:  
        mock_recommender.recommend.return_value = [
            {
                "title": "Mocked Movie 1",
                "rating": 8.5,
                "genres": "Action|Adventure"
            }
        ]
        yield mock_recommender

def test_recommend_movies_success(mock_recommender):
    response = client.post("/recommend", json={"movie_title": "Test Movie"})
    print("Response status:", response.status_code)
    print("Response body:", response.json())
    assert response.status_code == 200
    mock_recommender.recommend.assert_called_once_with("Test Movie")

def test_recommend_movies_not_found(mock_recommender):
    mock_recommender.recommend.side_effect = ValueError("Movie 'Invalid Movie' not found in the dataset.")

    response = client.post(
        "/recommend",
        json={"movie_title": "Invalid Movie"}
    )

    # Assert response
    assert response.status_code == 200
    assert response.json() == {"error": "Movie 'Invalid Movie' not found in the dataset."}

def test_recommend_movies_invalid_request():
    response = client.post(
        "/recommend",
        json={}
    )

    assert response.status_code == 422 # Validation error 
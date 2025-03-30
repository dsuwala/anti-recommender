import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, Mock
from main import app, get_recommender


mock_recommender = Mock()
app.dependency_overrides[get_recommender] = lambda: mock_recommender
client = TestClient(app)


@pytest.fixture
def mock_recommendations():
    return {"recommendations": [
        {"title": "Movie 1", "rating": 1.5, "year": 2000},
        {"title": "Movie 2", "rating": 4.2, "year": 2010},
        {"title": "Movie 3", "rating": 4.8, "year": 2015},
    ]}


@pytest.fixture(autouse=True)
def reset_mock_recommender():
    mock_recommender.reset_mock()
    yield


def test_recommend_movies_success():

    mock_recommender.recommend.return_value = {
        "recommendations": [
            {
                "title": "Mocked Movie 1",
                "rating": 1.5,
                "year": 2000
            }
        ]
    }

    response = client.post(
        "/recommend",
        json={"movie_title": "Test Movie", "year": 2000}
    )

    assert response.status_code == 200
    assert "recommendations" in response.json()
    mock_recommender.recommend.assert_called_once_with("Test Movie", 2000)


def test_recommend_movies_not_found():
    mock_recommender.recommend.side_effect = ValueError("Movie 'Invalid Movie' not found in the dataset.")

    response = client.post("/recommend", json={"movie_title": "Invalid Movie", "year": None})
    print(response.json())
    assert response.status_code == 200
    assert response.json() == {"error": "Movie 'Invalid Movie' not found in the dataset."}
    mock_recommender.recommend.assert_called_once_with("Invalid Movie", None)


def test_recommend_movies_invalid_request():
    response = client.post(
        "/recommend",
        json={}
    )

    assert response.status_code == 422

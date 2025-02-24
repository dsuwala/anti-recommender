from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from src.MovieAntiRecommender import MovieAntiRecommender
from config import settings

app = FastAPI()

class RecommendationRequest(BaseModel):
    movie_title: str
    year: int | None = None

recommender = MovieAntiRecommender()
recommender.load_dataset(settings.data_path, settings.model_path)

@app.post("/recommend")
def recommend_movies(request: RecommendationRequest):
    try:
        recommendations = recommender.recommend(request.movie_title, request.year)
        return recommendations
    except ValueError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": "An unexpected error occurred"}

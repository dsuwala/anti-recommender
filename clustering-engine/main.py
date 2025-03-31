import logging
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from src.MovieAntiRecommender import MovieAntiRecommender
from config import settings

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_recommender():
    if not hasattr(get_recommender, "instance"):
        try:
            logger.info("Initializing recommender...")
            logger.info(f"Ititializing with data path: {settings.data_path} and model path: {settings.model_path}")
            recommender = MovieAntiRecommender()
            recommender.load_dataset(settings.data_path, settings.model_path)
            logger.info("Recommender initialized successfully")
            logger.info(f"Using dataset: {settings.data_path} and model: {settings.model_path}")
            get_recommender.instance = recommender
        except Exception as e:
            logger.error(f"Error initializing recommender: {e}")
            raise
    return get_recommender.instance


class RecommendationRequest(BaseModel):
    movie_title: str
    year: int | None = None


class SearchSuggestionRequest(BaseModel):
    query: str


@app.post("/recommend")
def recommend_movies(request: RecommendationRequest, recommender: MovieAntiRecommender = Depends(get_recommender)):
    try:
        logger.info(f"Received recommendation request for movie: {request.movie_title}, year: {request.year}")
        recommendations = recommender.recommend(str(request.movie_title),
                                                request.year)
        logger.info(f"Recommendations: {recommendations}")
        return recommendations
    except ValueError as e:
        logger.error(f"Error recommending movies: {e}")
        return {"error": str(e)}
    except Exception:
        logger.error("An unexpected error occurred")
        return {"error": "An unexpected error occurred"}


@app.get("/search-suggestions")
def search_suggestions(query: str, recommender: MovieAntiRecommender = Depends(get_recommender)):
    try:
        logger.info(f"Received search suggestions request for query: {query}")
        suggestions = recommender.search_suggestions(query)
        logger.info(f"Suggestions: {suggestions}")
        return {"suggestions": suggestions}
    except Exception as e:
        logger.error(f"Error searching suggestions: {e}")
        return {"error": str(e)}

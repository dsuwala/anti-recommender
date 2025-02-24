## Overview
This repository contains simple code for a movie antirecommender. If you are bored by your usual tastes, check what 
you can watch next! <br>
Out of the three predictions, the first one is from first quantile of average ratings, second one from up to Q3 and the third one from 0.97 quantile.

## Technicalities 
This code base is built on top of the [MovieLens dataset](https://grouplens.org/datasets/movielens/latest/). And uses KMeans clusterisation based on one-hot encoded movie genres to find movies of the farther "taste".
Simoultaneously, It recomends the best rated movies if available and also worst rated. Due to the terms of use, the 
dataset is not included in the repository, but you can download it from the [link](https://grouplens.org/datasets/movielens/latest/). 
Then you can clean it by using `prototype.ipynb` notebook and then cluster data using `prototype_clusterisation.ipynb` notebook. When 
you obtain `cleaned.csv` file and `kmeans_model.pkl` file, you can run the app.

## How to run

If you want to run it as a FastAPI app, you can do so by running:
```bash
python run.py
```
Then you can make a request to the app by running:
```bash
curl -X POST http://localhost:8000/recommend -H "Content-Type: application/json" -d '{"movie_title": "The Matrix", "year": 1999}'
```
The code contains also dockerfile to run the app in a container.
```bash
docker build -t antirecommender .
docker run -d -p 8000:8000 antirecommender
```

## How to test
The code is covered by unit tests for both api and the recommender methods. You can run them by running:
```bash
pytest
```


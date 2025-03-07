## Overview
This repository contains simple code for a movie antirecommender. If you are bored by your usual tastes, check what 
you can watch next! <br>
Out of the three predictions, the first one is from first quantile of average ratings, second one from up to Q3 and the third one from 0.97 quantile.

## Technicalities 
This code base is built on top of the [MovieLens dataset](https://grouplens.org/datasets/movielens/latest/). And uses KMeans clusterisation based on one-hot encoded movie genres to find movies of the farther "taste".
Simoultaneously, It recomends the best rated movies if available and also worst rated. Due to the terms of use, the 
dataset is not included in the repository, but you can download it from the [link](https://grouplens.org/datasets/movielens/latest/). In the `data/metrics_clustering.csv` there are provided inertia and silhouette data for 
cluster size scan in rage 15-620. Plots are provided in the `prototype_clusterization.ipynb` notebook.

## How to run

First, you need to download and extract MovieLens dataset using the link above and prepare it. 
To do it, there is dedicated movie-preprocessor docker container.

### Data cleaning and clusterization
First, extract data in the project top directory and you will obtain the directory `ml-latest/'. 
Then build the preprocessing container container using the command:
```bash
sudo docker build -t movie-preprocessor -f ./Dockerfile.preprocessing .
```
To perform basic cleaning and clustering of the data on default parameters (10 PCA components, 300 clusters)
execute:
```bash
sudo docker run -v /path/to/repository/antirecommender/archive/ml-latest:/app/data movie-preprocessor preprocess
```
which will produce `archive/ml-latest/cleaned_movies.csv` file. Then run the following to cluster data:
```bash
sudo docker run -v /path/to/repository/antirecommender/archive/ml-latest:/app/data movie-preprocessor preprocess
```
which will produce `kmeans.pkl` pickle file with fitted clusters. Those two files are necessary to launch the 
antirecommendation model. <br>
Preprocessing container takes more arguments and to see them you can use help information:
```bash
sudo docker run movie-preprocessor cluster --help
```

### Run the application
First, build the run container:
```bash
sudo docker build -t antirecommender -f ./Dockerfile.run .
```
and run it with mounted directory where you have `kmeans.pkl` and `cleaned_movies.csv` (`-v` flag) and 
listen port 8000 from the container (`-p` flag):
```bash
sudo docker run -p 8000:8000 -v /path/to/directory/with/files:/app/data movie-antirecommender
```
Then you can make a request to the app by running in a new terminal:
```bash
curl -X POST http://localhost:8000/recommend -H "Content-Type: application/json" -d '{"movie_title": "The Matrix", "year": 1999}'
```
or thourgh the `requests` python library.

## How to test
The code is covered by unit tests for both api and the recommender methods. You can run them by running:
```bash
pytest
```


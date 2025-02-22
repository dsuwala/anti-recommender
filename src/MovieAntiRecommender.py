import numpy as np
import pandas as pd
import joblib

class MovieAntiRecommender:
    def __init__(self, n_clusters=20, n_components=10, cache_file='movies_database.csv'):
        self.dataset = None
        self.model = None
        self.silhouette_avg = None
        self.rating_quantiles = None

    def load_dataset(self, name: str, model_name: str):
        self.dataset = pd.read_csv(name)
        self.movies = self.dataset['title'].values
        self.model = joblib.load(model_name)
        self.rating_quantiles = self.dataset['rating'].quantile([0.25, 0.75, 0.97]).to_numpy()


    def recommend(self, movie_title):

        # if movie_title not in self.dataset.title:
        #     raise ValueError(f"Movie '{movie_title}' not found in the dataset.")

        movie_idx = self.dataset[self.dataset['title'] == movie_title].index
        # print(movie_idx)
        movie_cluster = self.model.labels_[movie_idx]
        movie_cluster_center = self.model.cluster_centers_[movie_cluster]
        
        cluster_distances = np.linalg.norm(movie_cluster_center.reshape(1, -1) - self.model.cluster_centers_, axis=1)
        farthers_cluster_idx = np.argmax(cluster_distances)
        farthers_cluster_center = self.model.cluster_centers_[farthers_cluster_idx]


        possible_movies_low = self.dataset[(self.dataset["cluster"] == farthers_cluster_idx) & 
                                       (self.dataset["rating"] < self.rating_quantiles[0])]
        possible_movies_mid = self.dataset[(self.dataset["cluster"] == farthers_cluster_idx) & 
                                       (self.dataset["rating"] > self.rating_quantiles[1])]
        possible_movies_high = self.dataset[(self.dataset["cluster"] == farthers_cluster_idx) & 
                                       (self.dataset["rating"] > self.rating_quantiles[2])]
        
        if len(possible_movies_low) > 0:
            movie_low = possible_movies_low.sample(2)
        else:
            movie_low = pd.DataFrame()

        if len(possible_movies_mid) > 0:
            movie_mid = possible_movies_mid.sample(1)
        else:
            movie_mid = pd.DataFrame()

        if len(possible_movies_high) > 0:
            movie_high = possible_movies_high.sample(1)
        else:
            movie_high = pd.DataFrame()

        recommendations = pd.concat([movie_low, movie_mid, movie_high])

        recommendations = recommendations.drop(['cluster', 'movieId'], axis=1)
        recommendations = recommendations.to_dict('records')

        return recommendations
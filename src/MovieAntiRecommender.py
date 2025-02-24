import numpy as np
import pandas as pd
import joblib
from difflib import get_close_matches


class MovieAntiRecommender:
    def __init__(self):
        self.dataset = None
        self.model = None
        self.silhouette_avg = None
        self.rating_quantiles = None

    def load_dataset(self, name: str, model_name: str):
        self.dataset = pd.read_csv(name)
        # self.movies = self.dataset['title'].values
        self.model = joblib.load(model_name)
        self.rating_quantiles = self.dataset['rating'].quantile([0.25, 0.75, 0.97]).to_numpy()

        assert self.dataset.shape[0] == self.model.labels_.shape[0], "Dataset and model labels have different number of rows"
        
    

    def standardize_title(self, movie_title, year=None):

        if movie_title is None or movie_title == "":
            return {
                "error": "No movie title provided",
                "message": "Please provide a movie title"
            }

        movie_titles = self.dataset['standardized_title'].values

        # zeroth try: exact match if query is directly the name of the movie with proper spelling up to a case difference
        matching_titles = self.dataset[self.dataset['standardized_title'].str.lower() == movie_title.lower()]
        matching_titles_ids = matching_titles.index
        # print(matching_titles)

        # If year is provided, check if it matches the year of the exact title match
        if year is not None and len(matching_titles) == 1:
            matched_year = self.dataset.loc[matching_titles.index[0], 'year']
            if matched_year != int(year):
                return {
                    "error": "Ambiguous or no match found",
                    "message": "No exact match found for that title and year. Check the year and try again.",
                    "possible_matches": [(str(matching_titles.iloc[0]['standardized_title']), int(matched_year))]
                }
        elif len(matching_titles) == 1 and year is None:
            return matching_titles.index
        
        # First try: exact match after lowercasing for movies which contains the query
        matching_titles = np.array([str(title) for title in movie_titles if movie_title.lower() in title.lower()])
        matching_titles_ids = self.dataset[self.dataset['standardized_title'].isin(matching_titles)].index
        
        # Second try: find close matches with low threshold
        if not np.any(matching_titles):
            print("No exact match found. Searching for close matches...")
            matches = get_close_matches(movie_title.lower(), [t.lower() for t in movie_titles], n=5, cutoff=0.6)
            matching_titles = np.array([title for title in movie_titles if title.lower() in matches])
            matching_titles_ids = self.dataset[self.dataset['standardized_title'].isin(matching_titles)].index
        
        matching_titles_years = np.int32(self.dataset[self.dataset['standardized_title'].isin(matching_titles)]['year'].values)

        # If year is provided, filter matches by exact year
        if year is not None:
            matching_titles = matching_titles[matching_titles_years == int(year)]
            matching_titles_ids = matching_titles_ids[matching_titles_years == int(year)]

        if len(matching_titles) != 1:

            return {
                "error": "Ambiguous or no match found",
                "message": "Please be more specific. Did you mean one of these?",
                "possible_matches": list(zip(matching_titles.tolist(), matching_titles_years.tolist()))
            }
        elif len(matching_titles) == 1 and year is not None:
            return matching_titles_ids
        else:
            return self.dataset[self.dataset['standardized_title'] == matching_titles[0]].index


    def recommend(self, movie_title, year=None):

        movie_idx = self.standardize_title(movie_title, year)
        print(movie_idx)

        if isinstance(movie_idx, dict):
            return movie_idx
            
        movie_cluster = self.model.labels_[movie_idx]
        movie_cluster_center = self.model.cluster_centers_[movie_cluster]
        
        cluster_distances = np.linalg.norm(movie_cluster_center.reshape(1, -1) - self.model.cluster_centers_, axis=1)
        farthers_cluster_idx = np.argmax(cluster_distances)
        farthers_cluster_center = self.model.cluster_centers_[farthers_cluster_idx]


        possible_movies_low = self.dataset[(self.model.labels_ == farthers_cluster_idx) & 
                                       (self.dataset["rating"] < self.rating_quantiles[0])]
        possible_movies_mid = self.dataset[(self.model.labels_ == farthers_cluster_idx) & 
                                       (self.dataset["rating"] > self.rating_quantiles[1])]
        possible_movies_high = self.dataset[(self.model.labels_ == farthers_cluster_idx) & 
                                       (self.dataset["rating"] > self.rating_quantiles[2])]
        
        if len(possible_movies_low) > 0:
            movie_low = possible_movies_low.sample(1)
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
        recommendations = recommendations.to_dict(orient='records')
        recommendations = {"recommendations": recommendations}

        # Add best match to recommendations
        recommendations["query"] = {
            "title": self.dataset.iloc[movie_idx].title.values[0],
            "rating": float(self.dataset.iloc[movie_idx].rating.values[0]),
            "year": int(self.dataset.iloc[movie_idx].year.values[0])
        }
        return recommendations
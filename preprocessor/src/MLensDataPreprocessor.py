import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


class MLensDataPreprocessor:
    def __init__(self, pca_components=10, kmeans_clusters=300, working_dir="data"):
        """
        Initialize the MLensDataPreprocessor.

        Args:
            pca_components (int, optional): Number of components for PCA reduction.
                Defaults to 10.
            kmeans_clusters (int, optional): Number of clusters for KMeans clustering.
                Defaults to 300.
            working_dir (str, optional): Directory containing the data files.
                Defaults to "data".
        """
        self.pca_components = pca_components
        self.kmeans_clusters = kmeans_clusters
        self.working_dir = working_dir

    def standardize_title_and_year(self, title):
        """
        Standardize movie title and extract year from the title string.

        Args:
            title (str): Movie title in various formats (e.g., "Matrix, The (1999)",
                "The Matrix (1999)", "Matrix")

        Returns:
            tuple: A tuple containing:
                - str: Standardized title (e.g., "The Matrix")
                - int or None: Year if present in title, None otherwise
        """
        # Extract year if present
        year = None
        clean_title = title
        if '(' in title and ')' in title:
            year_part = title[title.rfind('(')+1:title.rfind(')')]
            if year_part.isdigit():
                year = int(year_part)
            clean_title = title.split('(')[0].strip()

        # Handle ", The/A/An" format
        lower_title = clean_title.lower()
        if ', the' in lower_title:
            clean_title = 'The ' + clean_title.split(',')[0]
        elif ', a' in lower_title:
            clean_title = 'A ' + clean_title.split(',')[0]
        elif ', an' in lower_title:
            clean_title = 'An ' + clean_title.split(',')[0]

        return clean_title.strip(), year

    def clean_movie_data(self, movies_df, ratings_df):
        """
        Clean and preprocess movie and ratings data.

        Args:
            movies_df (pd.DataFrame): DataFrame containing movie information
            ratings_df (pd.DataFrame): DataFrame containing rating information

        Returns:
            pd.DataFrame: Cleaned and preprocessed movie data with standardized titles,
                years, and average ratings
        """
        # Calculate average rating per movie and filter out movies with no ratings
        ratings_df = ratings_df.drop(['timestamp', 'userId'], axis=1)
        avg_ratings = ratings_df.groupby('movieId')['rating'].mean().reset_index()
        movies_df = movies_df[movies_df["movieId"].isin(avg_ratings["movieId"])]
        movies_df.reset_index(drop=True, inplace=True)
        movies_df["rating"] = avg_ratings["rating"]

        # Apply title standardization
        standardized_data = movies_df['title'].apply(self.standardize_title_and_year)
        movies_df['standardized_title'] = standardized_data.apply(lambda x: x[0])
        movies_df['year'] = standardized_data.apply(lambda x: x[1])

        # Filter out movies with no genres or no year
        movies_df = movies_df[~movies_df["movieId"].isin(
            movies_df[(movies_df["genres"] == "(no genres listed)") | (movies_df["year"].isna())].movieId
        )]

        # Replace empty strings with NaN
        movies_df = movies_df.replace(r'^\s*$', pd.NA, regex=True)

        # Handle special cases for standardized titles
        special_cases = {
            69757: "500 Days of Summer",
            80729: "Untitled",
            115263: "Asexual",
            145733: "The New War of the Buttons",
            147033: "Terror",
            160010: "Dishonesty The Truth About Lies",
            193219: "Girlfriend",
            208553: "Escape",
            230315: "Nieznajomi",
            211946: "Unideal Man",
            215643: "OO",
            234516: "My Truth: The Rape of 2 Coreys",
            250664: "Blooper Bunny!"
        }

        for movie_id, title in special_cases.items():
            movies_df.loc[movies_df['movieId'] == movie_id, 'standardized_title'] = title

        # Remove remaining entries with NaN standardized titles
        movies_df = movies_df[~movies_df["standardized_title"].isna()]

        return movies_df

    def create_genre_matrix(self, movies_df):
        """
        Create a one-hot encoding matrix for movie genres.

        Args:
            movies_df (pd.DataFrame): DataFrame containing movie information with genres

        Returns:
            np.ndarray: Binary matrix where each row represents a movie and each column
                represents a genre (1 if movie has the genre, 0 otherwise)
        """
        genres_list = movies_df["genres"].str.split("|")
        num_movies = movies_df.shape[0]

        unique_genres = set([genre for genres in genres_list for genre in genres])
        genre_matrix = np.zeros((num_movies, len(unique_genres)))

        for i, genres in enumerate(genres_list):
            for genre in genres:
                genre_matrix[i, list(unique_genres).index(genre)] = 1

        return genre_matrix

    def preprocess_data(self):
        """
        Main preprocessing pipeline for movie and ratings data.

        Returns:
            tuple: A tuple containing:
                - pd.DataFrame: Cleaned and preprocessed movie data
                - np.ndarray: Genre one-hot encoding matrix
        """
        # Load data
        movies_df = pd.read_csv(f"{self.working_dir}/movies.csv")
        ratings_df = pd.read_csv(f"{self.working_dir}/ratings.csv")

        # Clean data
        cleaned_movies = self.clean_movie_data(movies_df, ratings_df)
        genre_matrix = self.create_genre_matrix(cleaned_movies)

        cleaned_movies.to_csv(f"{self.working_dir}/cleaned_movies.csv", index=False)
        np.save(f"{self.working_dir}/genre_matrix.npy", genre_matrix)

        return cleaned_movies, genre_matrix

    def cluster_movies(self):
        """
        Perform dimensionality reduction and clustering on movie data.

        Returns:
            tuple: A tuple containing:
                - KMeans: Fitted KMeans clustering model
                - dict: Statistics including:
                    - PCA_cumulative_variance_ratio: Explained variance ratio
                    - movies_per_cluster: Number of movies in each cluster
        """

        data = np.load(f"{self.working_dir}/genre_matrix.npy")

        pca = PCA(n_components=self.pca_components)
        data = pca.fit_transform(data)

        explained_variance_ratio = pca.explained_variance_ratio_
        cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

        kmeans = KMeans(n_clusters=self.kmeans_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(data)

        stats = {"PCA_cumulative_variance_ratio": cumulative_variance_ratio[-1],
                 "movies_per_cluster": [np.sum(cluster_labels == i) for i in range(self.kmeans_clusters)]}

        return kmeans, stats

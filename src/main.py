from MovieRecommender import MovieAntiRecommender
import argparse


def main(movie_title, n_recommendations):
    recommender = MovieAntiRecommender()
    recommender.load_dataset("../data/clustered_dataset.csv", "../data/movies_kmeans.npy")
    recommender.recommend(movie_title, n_recommendations)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--movie", type=str, help="Movie title")
    parser.add_argument("--n_recommendations", type=int, help="Number of recommendations. Default is 3", default=3)
    args = parser.parse_args()

    main(args.movie, args.n_recommendations)

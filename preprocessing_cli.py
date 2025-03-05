import click
import logging
from src.MLensDataPreprocessor import MLensDataPreprocessor
import numpy as np
import joblib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@click.group()
def cli():
    """Movie recommendation system data preprocessing CLI"""
    pass


@cli.command()
def preprocess():
    """Preprocess movie and ratings data"""
    try:
        # Create output directory if it doesn't exist
        working_dir = "/app/data"

        logger.info("Starting data preprocessing...")
        logger.info(f"Working directory: {working_dir}")

        # Initialize preprocessor
        preprocessor = MLensDataPreprocessor(working_dir=working_dir)

        logger.info("Created preprocessor...")
        # Process data
        movies_df, genre_matrix = preprocessor.preprocess_data()

        # Save genre matrix
        np.save(f"{working_dir}/genre_matrix_full.npy", genre_matrix)

        logger.info(f"Successfully processed {len(movies_df)} movies")
        logger.info(f"Data saved to {working_dir}")

    except Exception as e:
        logger.error(f"Error during preprocessing: {str(e)}")
        raise click.Abort()


@cli.command()
@click.option('--pca-components', default=10, help='Number of PCA components')
@click.option('--kmeans-clusters', default=300, help='Number of KMeans clusters')
def cluster(pca_components, kmeans_clusters):
    """Cluster movies"""

    working_dir = "/app/data"
    try:
        preprocessor = MLensDataPreprocessor(pca_components, kmeans_clusters, working_dir)
        kmeans, stats = preprocessor.cluster_movies()

        logger.info("Successfully clustered movies")
        logger.info(f"PCA cumulative variance ratio: {stats['PCA_cumulative_variance_ratio']}")

        for i, movies_per_cluster in enumerate(stats['movies_per_cluster']):
            logger.info(f"Cluster {i}: {movies_per_cluster} movies")

        # Save kmeans model
        with open(f"{working_dir}/kmeans.pkl", "wb") as f:
            joblib.dump(kmeans, f)

    except Exception as e:
        logger.error(f"Error during clustering: {str(e)}")
        raise click.Abort()


if __name__ == '__main__':
    cli()

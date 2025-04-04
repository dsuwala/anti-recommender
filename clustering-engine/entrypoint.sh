#! /bin/bash

# Exit on error
set -e

# Mount GCS bucket
echo "Mounting GCS bucket $GCS_BUCKET_NAME to /app/data"
gcsfuse "$GCS_BUCKET_NAME" /app/data || {
    echo "Failed to mount GCS bucket"
    exit 1
}

# Start the application
echo "Starting the application..."
conda run --no-capture-output -n antirecommender python run.py


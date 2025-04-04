#! /bin/bash

gcfuse "$GCS_BUCKET_NAME" /app/data
conda run --no-capture-output -n antirecommender python run.py


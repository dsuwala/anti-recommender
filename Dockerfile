# Use the official conda image as base
FROM continuumio/miniconda3

# Set working directory
WORKDIR /app

# Copy environment.yml file
COPY environment.yml .

# Create conda environment
RUN conda env create -f environment.yml

# Make RUN commands use the new environment
SHELL ["conda", "run", "-n", "antirecommender", "/bin/bash", "-c"]

# Copy the rest of the application
COPY . .

# Expose port (we'll use the default from settings)
EXPOSE 8000

# Command to run the application using run.py
CMD ["conda", "run", "-n", "antirecommender", "python", "run.py"] 
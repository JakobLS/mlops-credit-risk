FROM continuumio/miniconda3
RUN apt update
RUN apt install -y gcc

WORKDIR /app

# Copy some files into the Docker container
ADD ./model_orchestration_and_tracking/artifacts artifacts/
ADD ./model_orchestration_and_tracking/mlruns mlruns/
COPY ["./model_deployment/conda.yaml", "./model_deployment/app_docker.py", "./"]

# Helps us to know how to load the trained model
ENV IN_A_DOCKER_CONTAINER=True

# Create conda env based on a yaml file
RUN conda env create --name credit --file conda.yaml

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "credit", "/bin/bash", "-c"]

EXPOSE 8090

# Make sure we use proper conda environment when running the gunicorn server
ENTRYPOINT [ "conda", "run", "--no-capture-output", "-n", "credit", "gunicorn", "--bind", "0.0.0.0:8090", "app_docker:server" ]


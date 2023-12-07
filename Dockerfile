# Start from a core stack version
FROM python:3.11-slim-buster

# Docker remains within this directory in the container
WORKDIR /workspace

# Add our company's certificate to the docker container instance certificates
ADD .devcontainer/zscaler.pem /usr/local/share/ca-certificates/zscaler.pem

# Update certificate store
RUN chmod 644 /usr/local/share/ca-certificates/zscaler.pem && update-ca-certificates

# Environment variable for Git and Requests to use CA certificate
ENV GIT_SSL_CAINFO=/usr/local/share/ca-certificates/zscaler.pem
ENV REQUESTS_CA_BUNDLE=/usr/local/share/ca-certificates/zscaler.pem

# Update and install essential packages
RUN apt-get update && apt-get install -y build-essential git ca-certificates

# Set pip default timeout (seconds)
ENV PIP_DEFAULT_TIMEOUT=400

# Copy requirements.txt to the docker container
COPY .devcontainer/requirements.txt .devcontainer/requirements.txt

## PRE-COMMITS ##

# Install pre-commit hooks
RUN pip3 install pre-commit
# RUN pre-commit install

# Install required libraries for pre-commit hooks
RUN pip3 install isort black bandit

##################

# Install python dependencies
RUN pip3 install --no-cache-dir -r .devcontainer/requirements.txt
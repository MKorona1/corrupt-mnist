# Base image
FROM python:3.11-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY mnist_classifier/ mnist_classifier/

# Copy the DVC files
COPY data.dvc .
COPY .dvc/ .dvc/

# Run DVC pull to fetch the data

WORKDIR /
RUN pip install dvc "dvc[gs]"
RUN dvc pull
RUN pip install -r requirements.txt --no-cache-dir
RUN pip install . --no-deps --no-cache-dir

ENTRYPOINT ["python", "-u", "mnist_classifier/train_model.py"]

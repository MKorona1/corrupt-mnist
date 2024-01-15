# Base image
FROM python:3.11-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY mnist_classifier/ mnist_classifier/s
# COPY .git/ .git/

# Run DVC pull to fetch the data
# RUN pip install dvc "dvc[gs]"
# RUN dvc init --no-scm
# COPY .dvc .dvc
# COPY data.dvc data.dvc
# RUN dvc config core.no_scm true
# RUN dvc pull --verbose
RUN pip install -r requirements.txt --no-cache-dir
RUN pip install . --no-deps --no-cache-dir

ENTRYPOINT ["python", "-u", "mnist_classifier/train_model.py"]

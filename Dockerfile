# syntax=docker/dockerfile:1.7
# runtime: API and worker. trainer: adds PyTorch (CPU) for the fatigue-model training job.
FROM python:3.12-slim-bookworm AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN useradd --uid 10001 --create-home app
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY pyproject.toml ./
COPY src/ src/
RUN pip install --no-deps . && mkdir -p /data && chown -R 10001:10001 /data
USER 10001
ENV CBO_DATA_DIR=/data
EXPOSE 8000 9100
CMD ["uvicorn", "cbo.api.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000", "--workers", "2", "--no-access-log"]

FROM runtime AS trainer
USER root
COPY requirements-train.txt .
RUN pip install -r requirements-train.txt --extra-index-url https://download.pytorch.org/whl/cpu
USER 10001
CMD ["cbo", "train-fatigue", "--out", "/data"]

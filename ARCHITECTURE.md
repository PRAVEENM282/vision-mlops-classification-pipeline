# System Architecture

## Overview
This project implements a production-grade, end-to-End MLOps pipeline for image classification. It is designed to be modular, scalable, and reproducible, leveraging modern containerization and MLOps best practices.

## Technology Stack
- **Language**: Python 3.10
- **Deep Learning**: PyTorch, Torchvision
- **API Framework**: FastAPI
- **Serving**: Gunicorn (with Uvicorn workers)
- **Experiment Tracking**: MLflow
- **Explainability**: Grad-CAM
- **Containerization**: Docker, Docker Compose

## System Components

### 1. Data Pipeline (`src/preprocess.py`)
Responsible for acquiring and preparing the **CIFAR10** dataset.
- **Ingestion**: Automates download and extraction of the dataset.
- **Splitting**: Stratified split into Training (80%) and Validation (20%) sets.
- **Augmentation**: Implements **RandAugment** for robust training and standardizes images using CIFAR10 mean/std.
- **Loaders**: Provides efficient PyTorch DataLoaders with configurable batch sizes and worker counts.

### 2. Training Pipeline (`src/train.py`)
Managed workflow for model training and fine-tuning.
- **Architecture**: **ResNet18** (Pre-trained on ImageNet). Selected for efficiency on CIFAR10.
- **Strategy**: Transfer Learning.
    - *Phase 1*: Feature Extraction (Freeze backbone, train head).
    - *Phase 2*: Fine-tuning (Unfreeze top layers, train with lower LR).
- **Optimization**: Uses SGD with Momentum and **Cosine Annealing Learning Rate Scheduler**.
- **Tracking**: Logs all parameters, metrics (accuracy, loss), and model artifacts to a local **MLflow** server.

### 3. Evaluation Pipeline (`src/evaluate.py`)
Independent module for checking model performance.
- Loads the best model artifact.
- Computes metrics: Accuracy, Precision, Recall, F1-Score.
- Generates Confusion Matrix.
- Saves results to `results/metrics.json`.

### 4. Inference API (`src/api.py`)
Real-time REST API for serving predictions.
- **Framework**: FastAPI served via Gunicorn for production concurrency.
- **Endpoints**:
    - `POST /predict`: Accepts an image, applies inference transforms, returning class and confidence.
    - `POST /explain`: Generates a **Grad-CAM** heatmap overlay to visualize model focus.
    - `GET /health`: Health check for orchestrators.

### 5. Infrastructure
- **Docker**: Multi-stage (implicit) build based on `python:3.10-slim`. Optimized for size and security.
- **Docker Compose**: Orchestrates the API and MLflow services.
- **Configuration**: Environment-variable driven via `.env` and `pydantic-settings`.

## Data Flow

1.  **Training Flow**:
    `Raw Data` -> `Preprocess (Augment)` -> `ResNet18 (Train)` -> `MLflow (Log)` -> `Code Artifact (.pth)`

2.  **Inference Flow**:
    `User Request (Image)` -> `API (Load Model)` -> `Transform` -> `Inference` -> `JSON Response`

3.  **Explanation Flow**:
    `User Request (Image)` -> `API` -> `Grad-CAM Hook` -> `Heatmap Generation` -> `Image Response`

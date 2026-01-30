# End-to-End Image Classification MLOps System

A production-grade, containerized image classification pipeline using PyTorch, FastAPI, and Docker.

## Architecture Structure

```
/
├── data/           # Dataset storage (Train/Val)
├── model/          # Serialized model artifacts
├── results/        # Evaluation metrics
├── src/            # Source code
│   ├── api.py      # FastAPI application
│   ├── config.py   # Configuration management
│   ├── evaluate.py # Metrics calculation
│   ├── preprocess.py # Data pipeline (Download/Split/Transform)
│   └── train.py    # Training loop with Transfer Learning
└── tests/          # Unit tests
```

## Setup Guide

### 1. Environment Setup

Copy the example environment file:
```bash
cp .env.example .env
```

### 2. Local Execution (No Docker)

Install dependencies:
```bash
pip install -r requirements.txt
```

**Step 1: Data Preparation**
Downloads Caltech-101 and splits it (top 10 classes).
```bash
python src/preprocess.py
```

**Step 2: Training**
Runs feature extraction and fine-tuning.
```bash
python src/train.py
```
*Output: Saves model to `model/resnet50_finetuned.pth`*

**Step 3: Evaluation**
Generates metrics.
```bash
python src/evaluate.py
```
*Output: Saves results to `results/metrics.json`*

**Step 4: Run API**
```bash
uvicorn src.api:app --reload
```

### 3. Docker Execution (Recommended)

Build and start the service:
```bash
docker-compose up --build
```
The API will be available at `http://localhost:8000`.

## API Usage

### Health Check
```bash
curl http://localhost:8000/health
```

### Prediction
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@/path/to/test_image.jpg"
```

## Design Decisions

*   **Transfer Learning**: Uses ResNet50. We employ a 2-phase approach: first training the head (FC layer) while the backbone is frozen, then fine-tuning top layers with a lower learning rate. This ensures stability and faster convergence.
*   **Config Management**: Uses `pydantic-settings` to enforce type safety on environment variables.
*   **FastAPI**: Selected for native async support, validation, and automated docs.
*   **Docker**: Multi-stage (implied/simplified) build pattern to keep images runnable. Volume mounting ensures models and data persist outside containers.

## Future Improvements

*   Add MLflow/Weights & Biases logging.
*   Implement a Celery worker queue for batch inference.
*   Add Kubernetes manifests (Helm charts).
*   Add CI/CD pipeline (GitHub Actions).

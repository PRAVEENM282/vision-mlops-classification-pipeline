import torch
import torch.nn as nn
from torchvision import models
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import json
import logging
import sys
import os
from pathlib import Path

try:
    from src.config import settings
    from src.preprocess import get_dataloaders
except ImportError:
    sys.path.append(str(Path(__file__).parent.parent))
    from src.config import settings
    from src.preprocess import get_dataloaders

logging.basicConfig(level=settings.LOG_LEVEL)
logger = logging.getLogger(__name__)

def load_saved_model(device):
    """Loads the model logic from creating to loading state dict"""
    if not Path(settings.MODEL_PATH).exists():
        raise FileNotFoundError(f"Model file not found at {settings.MODEL_PATH}. Train first!")

    checkpoint = torch.load(settings.MODEL_PATH, map_location=device)
    class_names = checkpoint['class_names']
    num_classes = len(class_names)
    
    # Recreate architecture
    model = models.resnet18(weights=None) # Start blank
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model, class_names

def evaluate():
    """
    Runs inference on validation set and computes metrics.
    Saves metrics to results/metrics.json
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Evaluating on {device}")
    
    # Load data
    _, val_loader, _ = get_dataloaders()
    
    # Load model
    model, class_names = load_saved_model(device)
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(val_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            if os.environ.get("FAST_DEV_RUN") and i >= 5:
                break

    # Compute Metrics
    acc = accuracy_score(all_labels, all_preds)
    precision, recall, _, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
    cm = confusion_matrix(all_labels, all_preds)

    metrics = {
        "accuracy": float(acc),
        "precision_weighted": float(precision),
        "recall_weighted": float(recall),
        "confusion_matrix": cm.tolist()
    }
    
    # Save results
    results_path = settings.RESULTS_DIR / "metrics.json"
    with open(results_path, "w") as f:
        json.dump(metrics, f, indent=2)
        
    logger.info(f"Metrics saved to {results_path}")
    logger.info(f"Accuracy: {acc:.4f}")

if __name__ == "__main__":
    evaluate()

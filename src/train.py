import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import time
import copy
import logging
import sys
import os
from pathlib import Path

# Fix import for direct execution
import mlflow
try:
    from src.config import settings
    from src.preprocess import get_dataloaders
except ImportError:
    sys.path.append(str(Path(__file__).parent.parent))
    from src.config import settings
    from src.preprocess import get_dataloaders

logging.basicConfig(level=settings.LOG_LEVEL)
logger = logging.getLogger(__name__)

def create_model(num_classes: int, device: torch.device):
    """
    Creates the ResNet18 model for transfer learning.
    """
    logger.info("Initializing ResNet18 model...")
    # Use weights="DEFAULT" for best available weights
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    
    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False
        
    # Replace the final fully connected layer
    # ResNet18 fc input features is 512
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    model = model.to(device)
    return model

def train_phase(model, loader, criterion, optimizer, device):
    """Single training epoch"""
    model.train()
    running_loss = 0.0
    running_corrects = 0
    
    for i, (inputs, labels) in enumerate(loader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        
        # FAST TRAIN FOR VERIFICATION
        if os.environ.get("FAST_DEV_RUN") and i >= 5:
            break
            
    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = running_corrects.double() / len(loader.dataset)
    
    return epoch_loss, epoch_acc

def validate_phase(model, loader, criterion, device):
    """Validation epoch"""
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            
            if os.environ.get("FAST_DEV_RUN") and i >= 5:
                break
            
    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = running_corrects.double() / len(loader.dataset)
    
    return epoch_loss, epoch_acc

def train_model():
    """
    Main training loop containing two phases:
    1. Feature Extraction (Head training only)
    2. Fine-tuning (Top layers training)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    train_loader, val_loader, class_names = get_dataloaders()
    num_classes = len(class_names)
    logger.info(f"Detected {num_classes} classes: {class_names}")
    
    # MLflow Setup
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment("resnet18_cifar10_optimized")
    mlflow.start_run()
    
    mlflow.log_param("epochs", settings.EPOCHS)
    mlflow.log_param("batch_size", settings.BATCH_SIZE)
    mlflow.log_param("learning_rate", settings.LEARNING_RATE)
    mlflow.log_param("model", "resnet18")
    mlflow.log_param("scheduler", "CosineAnnealingLR")
    mlflow.log_param("augmentation", "RandAugment")
    
    model = create_model(num_classes, device)
    criterion = nn.CrossEntropyLoss()
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    # --- PHASE 1: Feature Extraction ---
    logger.info("--- Phase 1: Feature Extraction ---")
    # Only parameters of final layer are being optimized
    optimizer = optim.Adam(model.fc.parameters(), lr=settings.LEARNING_RATE)
    
    feature_extract_epochs = max(1, settings.EPOCHS // 2) # Half epochs for phase 1
    
    for epoch in range(feature_extract_epochs):
        logger.info(f"Epoch {epoch+1}/{feature_extract_epochs}")
        
        train_loss, train_acc = train_phase(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate_phase(model, val_loader, criterion, device)
        
        logger.info(f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f}")
        logger.info(f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
        
        mlflow.log_metric("train_loss_phase1", train_loss, step=epoch)
        mlflow.log_metric("train_acc_phase1", train_acc.item(), step=epoch)
        mlflow.log_metric("val_loss_phase1", val_loss, step=epoch)
        mlflow.log_metric("val_acc_phase1", val_acc.item(), step=epoch)
        
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())

    # --- PHASE 2: Fine Tuning ---
    logger.info("--- Phase 2: Fine Tuning ---")
    
    # Reload best weights from Phase 1
    model.load_state_dict(best_model_wts)
    
    # Unfreeze the last block of ResNet
    for param in model.layer4.parameters():
        param.requires_grad = True
        
    # Lower learning rate for fine tuning
    # Lower learning rate for fine tuning
    optimizer = optim.SGD([
        {'params': model.layer4.parameters(), 'lr': settings.LEARNING_RATE / 10},
        {'params': model.fc.parameters(), 'lr': settings.LEARNING_RATE}
    ], momentum=0.9)
    
    # Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=settings.EPOCHS)
    
    fine_tune_epochs = settings.EPOCHS - feature_extract_epochs
    
    for epoch in range(fine_tune_epochs):
        logger.info(f"Epoch {epoch+1}/{fine_tune_epochs} (Fine-tuning)")
        
        train_loss, train_acc = train_phase(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate_phase(model, val_loader, criterion, device)
        
        # Step scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        logger.info(f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} LR: {current_lr:.6f}")
        logger.info(f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
        
        mlflow.log_metric("learning_rate", current_lr, step=feature_extract_epochs+epoch)
        mlflow.log_metric("train_loss_phase2", train_loss, step=feature_extract_epochs+epoch)
        mlflow.log_metric("train_acc_phase2", train_acc.item(), step=feature_extract_epochs+epoch)
        mlflow.log_metric("val_loss_phase2", val_loss, step=feature_extract_epochs+epoch)
        mlflow.log_metric("val_acc_phase2", val_acc.item(), step=feature_extract_epochs+epoch)
        
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())

    logger.info(f"Training complete. Best val Acc: {best_acc:.4f}")
    
    # Save the best model
    model.load_state_dict(best_model_wts)
    save_path = Path(settings.MODEL_PATH)
    torch.save({
        'model_state_dict': model.state_dict(),
        'class_names': class_names,
        'accuracy': best_acc.item() if torch.is_tensor(best_acc) else best_acc
    }, save_path)
    logger.info(f"Model saved to {save_path}")
    
    mlflow.log_artifact(str(save_path))
    mlflow.end_run()

if __name__ == "__main__":
    train_model()

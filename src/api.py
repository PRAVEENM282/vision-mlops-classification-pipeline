from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import torch
from torchvision import models, transforms
import torch.nn as nn
from PIL import Image
import io
import logging
from contextlib import asynccontextmanager
import sys
from pathlib import Path
import base64
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from fastapi.responses import Response

try:
    from src.config import settings
except ImportError:
    sys.path = [str(Path(__file__).parent.parent)] + sys.path
    from src.config import settings

logging.basicConfig(level=settings.LOG_LEVEL)
logger = logging.getLogger(__name__)

# Global state for model
ml_models = {}

def get_inference_transforms():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(settings.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model on startup
    logger.info("Loading model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        if not Path(settings.MODEL_PATH).exists():
           logger.warning(f"Model path {settings.MODEL_PATH} does not exist. API will fail predictions until trained.")
           ml_models["model"] = None
        else:
            checkpoint = torch.load(settings.MODEL_PATH, map_location=device)
            class_names = checkpoint['class_names']
            num_classes = len(class_names)

            model = models.resnet18(weights=None)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, num_classes)
            
            model.load_state_dict(checkpoint['model_state_dict'])
            model = model.to(device)
            model.eval()
            
            ml_models["model"] = model
            ml_models["classes"] = class_names
            ml_models["device"] = device
            ml_models["transforms"] = get_inference_transforms()
            logger.info("Model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        ml_models["model"] = None
        
    yield
    
    # Clean up
    ml_models.clear()

app = FastAPI(title="Image Classification API", lifespan=lifespan)

@app.get("/health")
def health_check():
    if ml_models.get("model") is None:
        return JSONResponse(status_code=503, content={"status": "not_ready", "detail": "Model not loaded"})
    return {"status": "ok"}

@app.post("/ping")
def ping():
    return {"status": "pong"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if ml_models.get("model") is None:
        raise HTTPException(status_code=503, detail="Model is not ready.")

    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Preprocess
        transform = ml_models["transforms"]
        img_t = transform(image)
        batch_t = torch.unsqueeze(img_t, 0)
        batch_t = batch_t.to(ml_models["device"])
        
        # Inference
        model = ml_models["model"]
        with torch.no_grad():
            outputs = model(batch_t)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            top_prob, top_catid = torch.topk(probabilities, 1)
            
        conf = top_prob[0].item()
        cat_idx = top_catid[0].item()
        class_name = ml_models["classes"][cat_idx]
        
        return {
            "predicted_class": class_name,
            "confidence": conf
        }
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
@app.post("/explain")
async def explain(file: UploadFile = File(...)):
    if ml_models.get("model") is None:
        raise HTTPException(status_code=503, detail="Model is not ready.")

    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Preprocess for Inference
        transform = ml_models["transforms"]
        img_t = transform(image)
        batch_t = torch.unsqueeze(img_t, 0)
        batch_t = batch_t.to(ml_models["device"])
        
        # Preprocess for Grad-CAM (Need un-normalized float32 image for overlay)
        # Using a simple resize/crop to match tensor dimensions
        img_np = np.array(image)
        img_np = np.float32(img_np) / 255
        # Note: We need to handle resizing to match model input (224x224) if we want accurate overlay
        # For simplicity, we leverage the inference transform's resize/crop logic manually or just rely on robust overlay
        # Better: resize img_np to 224x224
        img_resized = image.resize((settings.IMAGE_SIZE, settings.IMAGE_SIZE))
        rgb_img = np.float32(img_resized) / 255
        
        model = ml_models["model"]
        target_layers = [model.layer4[-1]]
        
        # Initialize GradCAM
        cam = GradCAM(model=model, target_layers=target_layers) #, use_cuda=torch.cuda.is_available())
        
        # We need to compute targets. If no target class provided, it uses the highest scoring class.
        targets = None 
        
        # Generate grayscale cam
        grayscale_cam = cam(input_tensor=batch_t, targets=targets)
        
        # In this example grayscale_cam has shape (1, 224, 224)
        grayscale_cam = grayscale_cam[0, :]
        
        # Create overlay
        visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        
        # Convert back to image
        result_image = Image.fromarray(visualization)
        
        # Save to buffer
        buf = io.BytesIO()
        result_image.save(buf, format="PNG")
        buf.seek(0)
        
        return Response(content=buf.getvalue(), media_type="image/png")
        
    except Exception as e:
        logger.error(f"Explain error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

from typing import Optional

from fastapi import FastAPI, Request, UploadFile, File, HTTPException
import requests
from fastapi.middleware.cors import CORSMiddleware

from app.models.plant_model import PlantDiseaseModel

import os
import json

app = FastAPI(title="Plant Disease Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"], 
    allow_headers=["*"],  
)

MODEL_PATH = "plant_disease_model.pth"
CLASS_NAMES = ['Tomato___Late_blight', 'Tomato___healthy', 'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Soybean___healthy', 'Squash___Powdery_mildew', 'Potato___healthy', 'Corn_(maize)___Northern_Leaf_Blight', 'Tomato___Early_blight', 'Tomato___Septoria_leaf_spot', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Strawberry___Leaf_scorch', 'Peach___healthy', 'Apple___Apple_scab', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Bacterial_spot', 'Apple___Black_rot', 'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Peach___Bacterial_spot', 'Apple___Cedar_apple_rust', 'Tomato___Target_Spot', 'Pepper,_bell___healthy', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Potato___Late_blight', 'Tomato___Tomato_mosaic_virus', 'Strawberry___healthy', 'Apple___healthy', 'Grape___Black_rot', 'Potato___Early_blight', 'Cherry_(including_sour)___healthy', 'Corn_(maize)___Common_rust_', 'Grape___Esca_(Black_Measles)', 'Raspberry___healthy', 'Tomato___Leaf_Mold', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Pepper,_bell___Bacterial_spot', 'Corn_(maize)___healthy']
DISEASE_DATA_PATH = os.path.join("app", "data", "disease_data.json")

with open(DISEASE_DATA_PATH, 'r') as f:
    DISEASE_TREATMENTS = json.load(f)

model = PlantDiseaseModel(model_path=MODEL_PATH, class_names=CLASS_NAMES)

@app.get("/")
def read_root():
    return {
        "status": "online",
        "api": "Plant Disease Detection API",
        "version": "1.0.0",
        "endpoints": {
            "/predict": "POST - Detect plant disease from image",
            "/diseases": "GET - List all detectable diseases",
            "/health": "GET - API health check"
        }
    }

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

@app.get("/diseases")
def list_diseases():
    """Return a list of all diseases the model can detect"""
    
    plants = {}
    for class_name in CLASS_NAMES:
        parts = class_name.split('___')
        if len(parts) == 2:
            plant, condition = parts
            if plant not in plants:
                plants[plant] = []
            plants[plant].append(condition)
    
    return {"status": "success", "data": plants}


@app.post("/predict")
async def predict_plant_disease(
    request: Request,
    file: Optional[UploadFile] = File(None),
    image_url: Optional[str] = None
):
    if not file and not image_url:
        raise HTTPException(
            status_code=400,
            detail="Either file or image_url must be provided"
        )
    
    try:
        if file:
            if not file.content_type.startswith("image/"):
                raise HTTPException(status_code=400, detail="File must be an image")
            
            contents = await file.read()
            image_bytes = contents
            source = "file"
        else:
            if not image_url.startswith(('http://', 'https://')):
                raise HTTPException(
                    status_code=400,
                    detail="Invalid image URL format"
                )
            
            response = requests.get(image_url)
            if response.status_code != 200:
                raise HTTPException(
                    status_code=400,
                    detail="Failed to fetch image from URL"
                )
            
            image_bytes = response.content
            source = "url"
        
        # Process image and make prediction
        prediction = model.predict_image(image_bytes)
        disease_name = prediction['class']
        
        recommendations = get_suggested_actions(disease_name)
        
        return {
            "status": "success",
            "data": {
                "diagnostic": prediction,
                "disease_info": recommendations,
                "source": source
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def get_suggested_actions(disease_name: str) -> str:
    """Return suggested actions based on disease"""
    
    if disease_name in DISEASE_TREATMENTS:
        return DISEASE_TREATMENTS[disease_name]
    
    if "healthy" in disease_name.lower():
        return DISEASE_TREATMENTS.get("healthy", "No specific treatment needed.")
    
    plant_type = None
    if "___" in disease_name:
        plant_type = disease_name.split("___")[0]

    return {
        "description": f"Detected condition: {disease_name}",
        "treatment": [
            f"Remove affected leaves or plant parts from your {plant_type if plant_type else 'plant'}",
            "Apply appropriate fungicide or pesticide based on the specific condition",
            "Ensure proper spacing for air circulation",
            "Consider consulting a local plant specialist for specific treatment"
        ],
        "prevention": [
            "Practice crop rotation",
            "Clean and disinfect gardening tools",
            "Remove plant debris from the garden",
            "Choose disease-resistant varieties when possible"
        ]
    }
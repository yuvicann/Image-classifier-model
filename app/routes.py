from fastapi import APIRouter, HTTPException, File, UploadFile
import cv2
import numpy as np
from model.predict import ImagePredictor
import io

router = APIRouter()
predictor = ImagePredictor("model/svm_model.pkl")

@router.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Read image file
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Make prediction
        result = predictor.predict(image)
        return {"status": "success", "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
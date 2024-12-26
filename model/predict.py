import joblib
import cv2
import numpy as np

class ImagePredictor:
    def __init__(self, model_path: str):
        model_dict = joblib.load(model_path)
        self.model = model_dict['model']
        self.scaler = model_dict['scaler']
        self.label_map = model_dict['label_map']
    
    def preprocess_image(self, image):
        """Preprocess image for prediction."""
        # Convert to grayscale if necessary
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Resize to match training size
        image = cv2.resize(image, (64, 64))
        # Flatten
        image_flat = image.flatten()
        return image_flat
    
    def predict(self, image):
        """Predict class for an image."""
        # Preprocess the image
        processed_image = self.preprocess_image(image)
        # Scale features
        scaled_features = self.scaler.transform([processed_image])
        # Get prediction and probability
        prediction = self.model.predict(scaled_features)[0]
        probabilities = self.model.predict_proba(scaled_features)[0]
        
        return {
            "predicted_class": self.label_map[prediction],
            "class_probabilities": {
                self.label_map[i]: float(prob) 
                for i, prob in enumerate(probabilities)
            }
        }
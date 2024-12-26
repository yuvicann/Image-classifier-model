import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import cv2
import os
from pathlib import Path

def load_images_from_folder(folder):
    """Load images and labels from folder structure."""
    images = []
    labels = []
    label_map = {}
    
    # Each subfolder name is a class label
    for label_idx, class_folder in enumerate(os.listdir(folder)):
        folder_path = os.path.join(folder, class_folder)
        if os.path.isdir(folder_path):
            label_map[label_idx] = class_folder
            for image_file in os.listdir(folder_path):
                img_path = os.path.join(folder_path, image_file)
                try:
                    # Read image in grayscale
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        # Resize to standard size
                        img = cv2.resize(img, (64, 64))
                        # Flatten the image
                        img_flat = img.flatten()
                        images.append(img_flat)
                        labels.append(label_idx)
                except Exception as e:
                    print(f"Error loading image {img_path}: {e}")
    
    return np.array(images), np.array(labels), label_map

def train_model(data_folder: str, model_path: str):
    # Load and preprocess images
    print("Loading images...")
    X, y, label_map = load_images_from_folder(data_folder)
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create and train SVM model
    print("Training SVM model...")
    svm = SVC(kernel='rbf', probability=True)
    svm.fit(X_train_scaled, y_train)
    
    # Test model
    accuracy = svm.score(X_test_scaled, y_test)
    print(f"Model accuracy: {accuracy:.2f}")
    
    # Save model, scaler, and label map
    model_dict = {
        'model': svm,
        'scaler': scaler,
        'label_map': label_map
    }
    joblib.dump(model_dict, model_path)
    print(f"Model saved at {model_path}")

if __name__ == "__main__":
    train_model("./dataset/images", "model/svm_model.pkl")
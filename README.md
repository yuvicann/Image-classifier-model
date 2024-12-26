# Image-classifier-model

A lightweight API to classify images using FastAPI and scikit-learn. It preprocesses images (grayscale, resize, flatten), trains an SVM model, and predicts categories with confidence scores via a /predict/ endpoint. Ideal for simple image categorization tasks.

Project Structure
app/: Contains FastAPI app and API routes.
model/: Includes scripts for training and prediction logic.
requirements.txt: Lists project dependencies.
README.md: Documentation.
How to Run
Install dependencies using the provided requirements.txt.
Train the SVM model with the training script.
Start the FastAPI app using uvicorn.
Using the API with Postman
Set the method to POST and URL to http://127.0.0.1:8000/predict/.
Go to the Body tab, select form-data, and add a key named file.
Select File from the dropdown and upload your image.
Click Send to get the predicted class and confidence scores.

 

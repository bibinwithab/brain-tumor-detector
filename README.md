# Brain Tumor Detector

A deep learning project for automatic brain tumor classification using MRI images. This project uses TensorFlow/Keras and FastAPI to train, evaluate, and serve a model that detects four types of brain tumors: glioma, meningioma, pituitary, and notumor. The model is based on a Convolutional Neural Network (CNN) architecture implemented with TensorFlow/Keras.

## Features

- Multi-class classification of brain tumors from MRI images
- REST API for prediction using FastAPI
- Randomized test script for model evaluation
- Organized dataset structure for training and testing

## Project Structure

```
backend/
  model.py           # Model training and evaluation
  test.py            # Random image prediction script
  app.py             # FastAPI backend for serving predictions
  requirements.txt   # Python dependencies
  Brain Tumor MRI/   # Dataset (Training/Testing folders)
  models/            # Saved model files (.h5)
frontend/            # Frontend code (Yet to update)
```

## Getting Started

### 1. Clone the repository

```sh
git clone https://github.com/bibinwithab/brain-tumor-detector.git
cd brain-tumor-detector
```

### 2. Install dependencies

```sh
cd backend
pip install -r requirements.txt
```

### 3. Prepare the dataset

- Place MRI images in `backend/Brain Tumor MRI/Training/<class_name>/` and `backend/Brain Tumor MRI/Testing/<class_name>/`.
- Supported classes: `glioma`, `meningioma`, `notumor`, `pituitary`.

### 4. Train the model

```sh
python model.py
```

### 5. Test the model

```sh
python test.py
```

### 6. Run the API server

```sh
uvicorn app:app --reload
```

### 7. Make predictions via API

- Use Postman or any HTTP client to send a POST request to `http://localhost:8000/predict-tumor` with an image file (key: `image`, type: `form-data`).

## Example API Request (Postman)

- Method: `POST`
- URL: `http://localhost:8000/predict-tumor`
- Body: `form-data`, key: `image`, type: `File`, value: (select image)

## Contact

For questions or collaboration, contact [bibingraceson@gmail.com](mailto:bibingraceson@gmail.com).

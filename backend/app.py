from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image
from datetime import datetime

import numpy as np
import os
import shutil

app = FastAPI()

model = load_model('./models/brain_tumor_detector_model.h5')

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "API is up and running"}

@app.post("/predict-tumor")
async def predict_tumor(image: UploadFile = File(...)):
    try:
        temp_file_location = f"./temp/{image.filename}"
        os.makedirs("./temp", exist_ok=True)

        with open(temp_file_location, "wb") as temp_file:
            shutil.copyfileobj(image.file, temp_file)

        img = keras_image.load_img(temp_file_location, target_size=(150, 150))
        x = keras_image.img_to_array(img)
        x = np.expand_dims(x, axis=0) / 255.0

        prediction = model.predict(x)[0]
        class_labels = ['glioma', 'meningioma', 'notumor', 'pituitary']
        predicted_class = np.argmax(prediction)
        confidence = round(prediction[predicted_class] * 100, 1)
        label = class_labels[predicted_class]

        if label == 'notumor':
            label = "No tumor detected"
        else:
            label = f"Tumor type detected: {label}"

        print("Predicted class label:", label)
        print("Confidence score:", confidence)
        print("Timestamp:", datetime.now().isoformat())

        os.remove(temp_file_location)

        return {
            "prediction": label,
            "confidence": float(confidence),
        }

    except Exception as e:
        print("Error:", str(e))
        return {"error": str(e)}
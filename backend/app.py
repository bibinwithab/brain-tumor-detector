import os
import shutil
from datetime import datetime
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Import from our custom modules
from config import TEMP_DIR, STANDARD_MODEL_PATH, TRANSFER_MODEL_PATH, TUMOR_PRESENCE_THRESHOLD
from model_loader import load_all_models
from utils import preprocess_image, predict_mask, check_for_tumor, encode_image_to_base64

# --- Initialization ---
app = FastAPI(title="Brain Tumor Segmentation API")

# Load models at startup for efficiency
models = load_all_models(STANDARD_MODEL_PATH, TRANSFER_MODEL_PATH)
os.makedirs(TEMP_DIR, exist_ok=True)

# Configure CORS (Cross-Origin Resource Sharing) to allow your frontend to communicate with this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (for development)
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods
    allow_headers=["*"],  # Allows all headers
)

# --- API Endpoints ---
@app.get("/")
def read_root():
    """A simple endpoint to confirm that the API is running."""
    return {"message": "Brain Tumor Segmentation API is up and running."}

@app.post("/predict")
async def predict_segmentation(file: UploadFile = File(...)):
    if not models:
        raise HTTPException(status_code=500, detail="Models are not loaded properly. Check server logs.")

    # Define a temporary path to save the uploaded file
    temp_file_path = os.path.join(TEMP_DIR, file.filename)
    
    try:
        # Save the uploaded file to the temporary location
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # --- Main Processing Pipeline (Modules 1, 3, 4, 6) ---
        # 1. Preprocess the image
        image_batch = preprocess_image(temp_file_path)
        
        # 3. Predict masks with both models
        mask_std = predict_mask(models['standard'], image_batch)
        mask_tfr = predict_mask(models['transfer'], image_batch)
        
        # 4. Check for tumor presence using the more reliable transfer learning model
        tumor_present = check_for_tumor(mask_tfr, TUMOR_PRESENCE_THRESHOLD)

        # 6. Formulate the response
        if not tumor_present:
            return {
                "timestamp": datetime.now().isoformat(),
                "tumor_present": False,
                "message": "No significant tumor region detected."
            }
        
        # If a tumor is present, encode the masks to base64 strings
        mask_std_b64 = encode_image_to_base64(mask_std)
        mask_tfr_b64 = encode_image_to_base64(mask_tfr)
        
        return {
            "timestamp": datetime.now().isoformat(),
            "tumor_present": True,
            "masks": {
                "standard_unet": mask_std_b64,
                "transfer_unet": mask_tfr_b64
            }
        }

    except Exception as e:
        # Catch any unexpected errors and return a helpful message
        raise HTTPException(status_code=500, detail=f"An error occurred during processing: {str(e)}")
    finally:
        # Ensure the temporary file is always cleaned up
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
# config.py
import os

# --- Core Paths ---
# Use os.path.abspath to ensure the paths are always correct, regardless of where the script is run from.
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models')
UPLOAD_DIR = os.path.join(BASE_DIR, 'uploads')

# --- File Paths ---
TEMP_DIR = "temp"
MODEL_DIR = "models"
STANDARD_MODEL_PATH = f"{MODEL_DIR}/standard_unet_model.h5"
TRANSFER_MODEL_PATH = f"{MODEL_DIR}/transfer_unet_model_finetuned.h5"

# --- Model & Image Settings ---
IMG_DIM = (128, 128)
IMG_CHANNELS = 3

# --- Prediction Threshold ---
# The minimum number of pixels that must be in a mask for it to be considered a "tumor".
TUMOR_PRESENCE_THRESHOLD = 200 # pixels

# --- Model Configuration ---
MODEL_PATH = os.path.join(MODEL_DIR, 'best_segmentation_model.h5')

# --- Image Processing Configuration ---
IMAGE_HEIGHT = 128
IMAGE_WIDTH = 128
IMAGE_CHANNELS = 3
IMAGE_DIM = (IMAGE_HEIGHT, IMAGE_WIDTH)

# --- Ensure Directories Exist ---
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

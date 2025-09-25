# utils.py
import os
import cv2
import numpy as np
import tensorflow as tf
from config import IMAGE_DIM

def preprocess_image(filepath):
    """
    Module 1: Loads and preprocesses an image for model prediction.
    - Loads an image from a file path.
    - Resizes it to the required dimensions.
    - Normalizes pixel values to the [0, 1] range.
    - Adds a batch dimension.
    """
    img = tf.keras.preprocessing.image.load_img(filepath, target_size=(*IMAGE_DIM, 3))
    img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

def predict_mask(model, image_batch):
    """
    Module 3: Performs segmentation using the loaded model.
    """
    return model.predict(image_batch)[0] # Get the first (and only) prediction

def refine_mask(predicted_mask):
    """
    Module 4: Post-processes the raw mask to clean it up.
    """
    mask = (predicted_mask.squeeze() * 255).astype(np.uint8)
    kernel = np.ones((5, 5), np.uint8)
    # Morphological opening removes small noise/dots
    opened_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    # Morphological closing fills small holes
    refined_mask = cv2.morphologyEx(opened_mask, cv.MORPH_CLOSE, kernel, iterations=2)
    return refined_mask

def create_overlay(original_image_path, refined_mask):
    """
    Module 6 (Visualization): Overlays the refined mask onto the original image.
    """
    # Load original image and resize it
    original_img = cv2.imread(original_image_path)
    original_img = cv2.resize(original_img, IMAGE_DIM)

    # Create a colored version of the mask (e.g., in red)
    mask_colored = cv2.cvtColor(refined_mask, cv2.COLOR_GRAY2BGR)
    mask_colored[np.where((mask_colored == [255, 255, 255]).all(axis=2))] = [0, 0, 255] # BGR format for Red

    # Blend the original image and the mask
    # A weighted sum: 60% original image, 40% mask
    overlayed_image = cv2.addWeighted(original_img, 0.6, mask_colored, 0.4, 0)

    return overlayed_image

def check_for_tumor(mask, threshold):
    return np.mean(mask) > threshold

def encode_image_to_base64(image):
    _, buffer = cv2.imencode('.png', image)
    return base64.b64encode(buffer).decode('utf-8')


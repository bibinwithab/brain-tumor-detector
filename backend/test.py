import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model

# ==============================================================================
# METRICS (Required for loading the models)
# These custom objects must be defined so Keras knows what they are.
# ==============================================================================
def dice_coefficient(y_true, y_pred, smooth=1e-6):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def iou_metric(y_true, y_pred, smooth=1e-6):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f) + K.sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)

def dice_loss(y_true, y_pred):
    return 1 - dice_coefficient(y_true, y_pred)

# ==============================================================================
# PREPROCESSING FUNCTION
# ==============================================================================
def preprocess_image(image_path, target_dim=(128, 128)):
    """
    Loads an image, resizes it, normalizes it, and prepares it for the model.
    """
    try:
        # Load image, ensuring it has 3 channels (RGB)
        img = tf.keras.preprocessing.image.load_img(image_path, target_size=target_dim, color_mode='rgb')
        # Convert to numpy array and normalize to [0, 1]
        img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
        # Add a batch dimension: (H, W, C) -> (1, H, W, C)
        return np.expand_dims(img_array, axis=0)
    except Exception as e:
        print(f"Error preprocessing image at {image_path}: {e}")
        return None

# ==============================================================================
# MAIN SCRIPT EXECUTION
# ==============================================================================
if __name__ == "__main__":
    # --- 1. Set up argument parser to accept an image path ---
    parser = argparse.ArgumentParser(description="Predict brain tumor segmentation masks from an MRI image.")
    parser.add_argument("-i", "--image", type=str, required=True, help="Path to the input image file.")
    args = parser.parse_args()

    # --- 2. Configuration ---
    MODEL_DIR = './models'
    STANDARD_MODEL_PATH = os.path.join(MODEL_DIR, 'standard_unet_model.h5')
    TRANSFER_MODEL_PATH = os.path.join(MODEL_DIR, 'transfer_unet_model_finetuned.h5')
    IMG_DIM = (128, 128)

    # --- 3. Check if files exist ---
    if not os.path.exists(args.image):
        print(f"Error: Input image not found at '{args.image}'")
        exit()
    if not os.path.exists(STANDARD_MODEL_PATH) or not os.path.exists(TRANSFER_MODEL_PATH):
        print(f"Error: Model files not found. Ensure '{STANDARD_MODEL_PATH}' and '{TRANSFER_MODEL_PATH}' exist.")
        exit()

    # --- 4. Load the trained models ---
    print("--- Loading models... ---")
    custom_objects = {'dice_loss': dice_loss, 'dice_coefficient': dice_coefficient, 'iou_metric': iou_metric}
    standard_model = load_model(STANDARD_MODEL_PATH, custom_objects=custom_objects)
    transfer_model = load_model(TRANSFER_MODEL_PATH, custom_objects=custom_objects)
    print("--- Models loaded successfully. ---")

    # --- 5. Preprocess the input image ---
    print(f"--- Preprocessing '{args.image}'... ---")
    image_batch = preprocess_image(args.image, target_dim=IMG_DIM)

    if image_batch is not None:
        # --- 6. Predict masks using both models ---
        print("--- Predicting masks... ---")
        pred_std_mask = standard_model.predict(image_batch)[0]
        pred_tfr_mask = transfer_model.predict(image_batch)[0]

        # --- 7. Visualize the results ---
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Display the original preprocessed image
        axes[0].imshow(image_batch[0])
        axes[0].set_title("Original Image")
        axes[0].axis('off')
        
        # Display the mask from the standard U-Net
        axes[1].imshow(pred_std_mask.squeeze(), cmap='gray')
        axes[1].set_title("Standard U-Net Mask")
        axes[1].axis('off')

        # Display the mask from the fine-tuned Transfer U-Net
        axes[2].imshow(pred_tfr_mask.squeeze(), cmap='gray')
        axes[2].set_title("Transfer U-Net Mask")
        axes[2].axis('off')

        plt.tight_layout()
        plt.show()
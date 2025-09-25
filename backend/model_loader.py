# model_loader.py
import tensorflow as tf
from tensorflow.keras.models import load_model

# --- Custom Objects for Keras Model ---
# These functions must be defined to load the model successfully.

def dice_coefficient(y_true, y_pred, smooth=1e-6):
    """Metric: Calculates the Dice coefficient."""
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    """Loss Function: The Dice Loss is 1 - Dice Coefficient."""
    return 1 - dice_coefficient(y_true, y_pred)

def load_segmentation_model(model_path):
    """
    Loads the trained Keras model with its custom loss and metric functions.
    """
    print(f"Loading model from: {model_path}")
    # Define the custom objects dictionary
    custom_objects = {
        'dice_loss': dice_loss,
        'dice_coefficient': dice_coefficient
    }
    # Load the model
    model = load_model(model_path, custom_objects=custom_objects)
    print("Model loaded successfully.")
    return model

def load_all_models(model_path,model_path2):
    return {
        'model': load_segmentation_model(model_path),
        'model2': load_segmentation_model(model_path2)
    }

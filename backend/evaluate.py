import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ==============================================================================
# MODULE 5: METRICS & UTILITY FUNCTIONS
# ==============================================================================

def dice_coefficient(y_true, y_pred, smooth=1e-6):
    """Calculates the Dice coefficient metric."""
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def iou_metric(y_true, y_pred, smooth=1e-6):
    """Calculates the IoU (Jaccard) metric."""
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f) + K.sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)

def load_dataset_in_memory(df, data_dir, dim=(128, 128)):
    """Loads a dataset of images and masks into memory."""
    images, masks = [], []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Loading data from {data_dir}"):
        img_path = os.path.join(data_dir, row['image_path'])
        mask_path = os.path.join(data_dir, row['mask_path'])
        
        img = tf.keras.preprocessing.image.load_img(img_path, target_size=dim)
        img = tf.keras.preprocessing.image.img_to_array(img) / 255.0
        
        mask = tf.keras.preprocessing.image.load_img(mask_path, target_size=dim, color_mode="grayscale")
        mask = tf.keras.preprocessing.image.img_to_array(mask)
        mask = (mask / 255.0 > 0.5).astype("float32")
        
        images.append(img)
        masks.append(mask)
        
    return np.array(images), np.array(masks)

def visualize_predictions(X_test, Y_test, model1, model2, results_dir, num_samples=5):
    """Saves a visual comparison of model predictions on random samples."""
    indices = np.random.choice(len(X_test), num_samples, replace=False)
    
    plt.figure(figsize=(16, num_samples * 4))
    
    for i, idx in enumerate(indices):
        img = X_test[idx]
        gt_mask = Y_test[idx]
        
        pred1 = model1.predict(np.expand_dims(img, axis=0), verbose=0)[0]
        pred2 = model2.predict(np.expand_dims(img, axis=0), verbose=0)[0]
        
        plt.subplot(num_samples, 4, i * 4 + 1); plt.imshow(img); plt.title(f"Image #{idx}"); plt.axis('off')
        plt.subplot(num_samples, 4, i * 4 + 2); plt.imshow(gt_mask.squeeze(), cmap='gray'); plt.title("Ground Truth"); plt.axis('off')
        plt.subplot(num_samples, 4, i * 4 + 3); plt.imshow(pred1.squeeze(), cmap='gray'); plt.title("Standard U-Net"); plt.axis('off')
        plt.subplot(num_samples, 4, i * 4 + 4); plt.imshow(pred2.squeeze(), cmap='gray'); plt.title("Transfer U-Net"); plt.axis('off')

    plt.tight_layout()
    save_path = os.path.join(results_dir, 'evaluation_prediction_comparison.png')
    plt.savefig(save_path)
    print(f"\nSaved prediction visualization to {save_path}")
    plt.close()

# ==============================================================================
# MAIN EVALUATION SCRIPT
# ==============================================================================
if __name__ == "__main__":
    # --- 1. Configuration ---
    DATASET_ROOT = './brain-tumor-dataset-segmentation-and-classification'
    MODEL_DIR = './models'
    RESULTS_DIR = './results'
    os.makedirs(RESULTS_DIR, exist_ok=True)
    IMG_DIM = (128, 128)

    # --- 2. Load Test Data ---
    # Scan for all image-mask pairs to create the full metadata
    image_paths, mask_paths = [], []
    for dirpath, _, filenames in os.walk(DATASET_ROOT):
        for f in sorted(filenames):
            if f.endswith('.png') and not f.endswith('_mask.png'):
                m = f.replace('.png', '_mask.png')
                if m in filenames:
                    image_paths.append(os.path.relpath(os.path.join(dirpath, f), DATASET_ROOT))
                    mask_paths.append(os.path.relpath(os.path.join(dirpath, m), DATASET_ROOT))
    
    metadata_df = pd.DataFrame({'image_path': image_paths, 'mask_path': mask_paths})
    # Use the same random_state to ensure we get the exact same test set as during training
    _, test_df = train_test_split(metadata_df, test_size=0.2, random_state=42)
    
    print(f"Found {len(test_df)} pairs in the test set.")
    X_test, Y_test = load_dataset_in_memory(test_df, DATASET_ROOT, dim=IMG_DIM)

    # --- 3. Load Trained Models ---
    print("\n--- Loading trained models ---")
    custom_objects = {'dice_coefficient': dice_coefficient, 'iou_metric': iou_metric, 'dice_loss': lambda y_true, y_pred: 1 - dice_coefficient(y_true, y_pred)}
    
    standard_model_path = os.path.join(MODEL_DIR, 'standard_unet_model.h5')
    transfer_model_path = os.path.join(MODEL_DIR, 'transfer_unet_model_finetuned.h5')
    
    standard_model = load_model(standard_model_path, custom_objects=custom_objects)
    transfer_model = load_model(transfer_model_path, custom_objects=custom_objects)
    print("Models loaded successfully.")

    # --- 4. Run Evaluation ---
    results = []
    print("\n--- Evaluating models on the test set ---")
    
    for i in tqdm(range(len(X_test)), desc="Evaluating predictions"):
        img_batch = np.expand_dims(X_test[i], axis=0)
        gt_mask = Y_test[i]
        
        pred_std = standard_model.predict(img_batch, verbose=0)[0]
        pred_tfr = transfer_model.predict(img_batch, verbose=0)[0]
        
        results.append({
            'dice_standard': dice_coefficient(gt_mask, pred_std).numpy(),
            'iou_standard': iou_metric(gt_mask, pred_std).numpy(),
            'dice_transfer': dice_coefficient(gt_mask, pred_tfr).numpy(),
            'iou_transfer': iou_metric(gt_mask, pred_tfr).numpy()
        })
        
    # --- 5. Generate Report and Visualizations ---
    results_df = pd.DataFrame(results)
    report_path = os.path.join(RESULTS_DIR, 'evaluation_report.csv')
    results_df.to_csv(report_path, index=False)
    print(f"\nDetailed performance report saved to {report_path}")
    
    avg_scores = results_df.mean()
    print("\n--- Average Performance on Test Set ---")
    print(f"Standard U-Net:   Dice = {avg_scores['dice_standard']:.4f}, IoU = {avg_scores['iou_standard']:.4f}")
    print(f"Transfer U-Net:   Dice = {avg_scores['dice_transfer']:.4f}, IoU = {avg_scores['iou_transfer']:.4f}")
    
    # Generate and save box plots
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1); results_df[['dice_standard', 'dice_transfer']].plot(kind='box', ax=plt.gca()); plt.title('Dice Score Comparison'); plt.xticks([1, 2], ['Standard', 'Transfer']); plt.ylabel('Dice Coefficient'); plt.grid(axis='y')
    plt.subplot(1, 2, 2); results_df[['iou_standard', 'iou_transfer']].plot(kind='box', ax=plt.gca()); plt.title('IoU Score Comparison'); plt.xticks([1, 2], ['Standard', 'Transfer']); plt.ylabel('IoU Score'); plt.grid(axis='y')
    
    boxplot_path = os.path.join(RESULTS_DIR, 'evaluation_score_boxplots.png')
    plt.tight_layout()
    plt.savefig(boxplot_path)
    print(f"Score comparison box plots saved to {boxplot_path}")
    plt.close()
    
    # Generate and save visual prediction comparisons
    visualize_predictions(X_test, Y_test, standard_model, transfer_model, RESULTS_DIR)

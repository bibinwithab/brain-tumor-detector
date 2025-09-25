import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate
from tensorflow.keras.applications import VGG16
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ==============================================================================
# METRICS & DATA LOADING (Unchanged)
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

def load_dataset_in_memory(df, data_dir, dim=(128, 128)):
    images, masks = [], []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Loading data"):
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

# ==============================================================================
# MODEL ARCHITECTURES (Updated for clarity)
# ==============================================================================
def build_standard_unet(input_shape=(128, 128, 3)):
    inputs = Input(input_shape)
    c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    p1 = MaxPooling2D((2, 2))(c1)
    c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    p2 = MaxPooling2D((2, 2))(c2)
    c3 = Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c3)
    u6 = concatenate([u6, c2])
    c6 = Conv2D(128, (3, 3), activation='relu', padding='same')(u6)
    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c1])
    c7 = Conv2D(64, (3, 3), activation='relu', padding='same')(u7)
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c7)
    return Model(inputs=[inputs], outputs=[outputs], name="Standard_U-Net")

def build_unet_with_transfer_learning(input_shape=(128, 128, 3)):
    # --- CORRECTED STRUCTURE ---
    # We define the VGG16 base and use it directly, ensuring it's a top-level layer.
    
    # 1. Define the input layer for the entire model
    inputs = Input(shape=input_shape)
    
    # 2. Define the VGG16 base model and connect it to our inputs
    base_model = VGG16(weights='imagenet', include_top=False, input_tensor=inputs)
    
    # 3. Get the specific layers from the base_model for skip connections
    s1 = base_model.get_layer('block1_conv2').output
    s2 = base_model.get_layer('block2_conv2').output
    s3 = base_model.get_layer('block3_conv3').output
    s4 = base_model.get_layer('block4_conv3').output
    bridge = base_model.get_layer('block5_conv3').output
    
    # 4. Build the decoder path
    d1 = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(bridge)
    d1 = concatenate([d1, s4])
    d2 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(d1)
    d2 = concatenate([d2, s3])
    d3 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(d2)
    d3 = concatenate([d3, s2])
    d4 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(d3)
    d4 = concatenate([d4, s1])
    
    # 5. Define the final output layer
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(d4)
    
    # 6. Create the final model
    model = Model(inputs=[inputs], outputs=[outputs], name="Transfer_U-Net")
    return model

# ==============================================================================
# MAIN SCRIPT
# ==============================================================================
if __name__ == "__main__":
    # --- 1. Configuration ---
    DATASET_ROOT = './brain-tumor-dataset-segmentation-and-classification'
    MODEL_DIR = './models'
    os.makedirs(MODEL_DIR, exist_ok=True)
    IMG_DIM = (128, 128)
    
    # --- 2. Prepare Data ---
    image_paths, mask_paths = [], []
    for dirpath, _, filenames in os.walk(DATASET_ROOT):
        for f in sorted(filenames):
            if f.endswith('.png') and not f.endswith('_mask.png'):
                m = f.replace('.png', '_mask.png')
                if m in filenames:
                    image_paths.append(os.path.relpath(os.path.join(dirpath, f), DATASET_ROOT))
                    mask_paths.append(os.path.relpath(os.path.join(dirpath, m), DATASET_ROOT))
    
    metadata_df = pd.DataFrame({'image_path': image_paths, 'mask_path': mask_paths})
    train_df, test_df = train_test_split(metadata_df, test_size=0.2, random_state=42)
    
    X_train, Y_train = load_dataset_in_memory(train_df, DATASET_ROOT, dim=IMG_DIM)
    X_test, Y_test = load_dataset_in_memory(test_df, DATASET_ROOT, dim=IMG_DIM)

    # --- 3. Train Standard U-Net ---
    print("\n--- Training Standard U-Net ---")
    standard_unet = build_standard_unet(input_shape=(*IMG_DIM, 3))
    standard_unet.compile(optimizer=Adam(1e-4), loss=dice_loss, metrics=[dice_coefficient])
    standard_unet.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=30, batch_size=16)
    standard_unet.save(os.path.join(MODEL_DIR, 'standard_unet_model.h5'))

    # ### --- TWO-STAGE TRAINING FOR TRANSFER U-NET --- ###
    
    # --- 3a. Initial Training (Frozen Encoder) ---
    print("\n--- Training Transfer U-Net (Stage 1: Frozen Encoder) ---")
    transfer_unet = build_unet_with_transfer_learning(input_shape=(*IMG_DIM, 3))
    
    # --- CORRECTED METHOD: Iterate through layers to freeze the encoder ---
    for layer in transfer_unet.layers:
        if "block" in layer.name:
            layer.trainable = False
    
    transfer_unet.compile(optimizer=Adam(1e-4), loss=dice_loss, metrics=[dice_coefficient])
    transfer_unet.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=20, batch_size=16)

    # --- 3b. Fine-Tuning (Unfrozen Encoder) ---
    print("\n--- Training Transfer U-Net (Stage 2: Fine-Tuning) ---")
    
    # --- CORRECTED METHOD: Selectively unfreeze top layers for fine-tuning ---
    for layer in transfer_unet.layers:
        if layer.name.startswith("block5") or layer.name.startswith("block4"):
            layer.trainable = True
    
    transfer_unet.compile(optimizer=Adam(1e-6), loss=dice_loss, metrics=[dice_coefficient]) 
    transfer_unet.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=10, batch_size=16)
    transfer_unet.save(os.path.join(MODEL_DIR, 'transfer_unet_model_finetuned.h5'))

    # --- 4. Final Evaluation ---
    print("\n--- Evaluating models on the test set ---")
    results = []
    standard_model = load_model(os.path.join(MODEL_DIR, 'standard_unet_model.h5'), custom_objects={'dice_loss': dice_loss, 'dice_coefficient': dice_coefficient})
    transfer_model = load_model(os.path.join(MODEL_DIR, 'transfer_unet_model_finetuned.h5'), custom_objects={'dice_loss': dice_loss, 'dice_coefficient': dice_coefficient})

    for i in tqdm(range(len(X_test)), desc="Evaluating"):
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
        
    # --- 5. Report and Conclude ---
    results_df = pd.DataFrame(results)
    avg_scores = results_df.mean()
    print("\n--- Average Performance ---")
    print(f"Standard U-Net:   Dice = {avg_scores['dice_standard']:.4f}, IoU = {avg_scores['iou_standard']:.4f}")
    print(f"Transfer U-Net (Fine-Tuned):   Dice = {avg_scores['dice_transfer']:.4f}, IoU = {avg_scores['iou_transfer']:.4f}")
    
    print("\n--- FINAL CONCLUSION ---")
    if avg_scores['dice_transfer'] > avg_scores['dice_standard']:
        print("The U-Net with Transfer Learning and Fine-Tuning achieved the best accuracy.")
    else:
        print("The Standard U-Net performed on par or better.")


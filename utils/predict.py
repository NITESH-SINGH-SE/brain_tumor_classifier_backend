"""
predict.py

Module for preprocessing images and generating Grad-CAM visualizations 
using a trained TensorFlow model. This is intended to be used in a FastAPI service.
"""
import os
import io
import base64
from uuid import uuid4
from typing import Optional
from datetime import datetime, timezone, timedelta

import numpy as np
import tensorflow as tf
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from fastapi import HTTPException

from db.db import Base, engine, SessionLocal
from models import TempFile

TEMP_DIR = "tmp"
# Create tmp dir if not exists
os.makedirs(TEMP_DIR, exist_ok=True)
EXPIRY_MINUTES = 60  # Files older than this will be deleted

# Load the trained model once when the server starts
current_dir = os.path.dirname(__file__)
model_path = os.path.join(current_dir, "..", "models", "best_model.h5")
model = tf.keras.models.load_model(os.path.abspath(model_path))
last_conv_layer_name = "mixed10"

# Define class labels
class_labels = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']

def preprocess_image(
        image_bytes: bytes
    ) -> np.ndarray:
    """
    Preprocess the uploaded image for model prediction.

    Args:
        image_bytes (bytes): Raw image bytes.

    Returns:
        np.ndarray: Preprocessed image tensor of shape (1, 224, 224, 3).
    """
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((224, 224))  # Match the model input size
    image_array = np.array(image) / 255.0  # Normalize to [0, 1]

    if image_array.shape[-1] == 4:  # Handle RGBA
        image_array = image_array[..., :3] # Remove alpha channel if present

    return np.expand_dims(image_array, axis=0)

def generate_gradcam_heatmap(
        img_array: np.ndarray, 
        model: tf.keras.Model, 
        last_conv_layer_name: str, 
        pred_index: Optional[int] = None
    ) -> np.ndarray:
    """
    Generate a Grad-CAM heatmap for the predicted class.

    Args:
        img_array (np.ndarray): Preprocessed input image array.
        model (tf.keras.Model): Trained TensorFlow model.
        last_conv_layer_name (str): Name of the last convolutional layer.
        pred_index (Optional[int], optional): Index of class to generate Grad-CAM for. 
                                              Defaults to predicted class.

    Returns:
        np.ndarray: 2D array representing the heatmap (values between 0 and 1).
    """
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def apply_heatmap_on_image(
        original_image: np.ndarray, 
        heatmap: np.ndarray, 
        alpha: float = 0.4
    ) -> np.ndarray:
    """
    Superimpose the Grad-CAM heatmap on the original image.

    Args:
        heatmap (np.ndarray): Grad-CAM heatmap (0 to 1).
        original_image (np.ndarray): Original RGB image array (not batched).
        alpha (float): Opacity of heatmap overlay.

    Returns:
        np.ndarray: Image with heatmap superimposed.
    """
    heatmap = cv2.resize(heatmap, original_image.size)
    heatmap = np.uint8(255 * heatmap)
    color_map = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    image_np = np.array(original_image)
    if image_np.shape[-1] == 4:
        image_np = image_np[..., :3]

    superimposed_img = cv2.addWeighted(image_np, 1 - alpha, color_map, alpha, 0)
    return superimposed_img

def encode_image_to_base64(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode()

def encode_array_to_base64(image_array: np.ndarray) -> str:
    _, buffer = cv2.imencode('.jpg', image_array)
    return base64.b64encode(buffer).decode()

# def predict(image_bytes: bytes) -> dict:
#     processed_image = preprocess_image(image_bytes)
#     original_image = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((224, 224))

#     try:
#         prediction = model.predict(processed_image)
#     except Exception as e:
#         try:
#             prediction = model.predict({'input_layer_1': processed_image})
#         except Exception as e:
#             raise HTTPException(status_code=500, detail=str(e))

#     class_index = np.argmax(prediction)
#     class_label = class_labels[class_index]

#     # Generate Grad-CAM heatmap
#     heatmap = generate_gradcam_heatmap(processed_image, model, last_conv_layer_name, class_index)
#     heatmap_image = apply_heatmap_on_image(original_image, heatmap)

#     return {
#         "prediction": class_label,
#         "confidence": float(prediction[0][class_index]),
#         "probabilities": prediction[0].tolist(),
#         "original_img_array": original_image,
#         "heatmap_array": heatmap_image,
#         "original_img_base64": encode_image_to_base64(original_image), #Give to frontend
#         "heatmap_img_base64": encode_array_to_base64(heatmap_image)
#     }

def predict(image_bytes: bytes) -> dict:
    processed_image = preprocess_image(image_bytes)
    original_image = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((224, 224))

    try:
        prediction = model.predict(processed_image)
    except Exception as e:
        try:
            prediction = model.predict({'input_layer_1': processed_image})
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    class_index = np.argmax(prediction)
    class_label = class_labels[class_index]

    # Generate Grad-CAM heatmap
    heatmap = generate_gradcam_heatmap(processed_image, model, last_conv_layer_name, class_index)
    heatmap_image = apply_heatmap_on_image(original_image, heatmap)
    

    # session_id = str(uuid4())

    # pred_path = f"{TEMP_DIR}/{session_id}_prediction.txt"
    # gradcam_path = f"{TEMP_DIR}/{session_id}_gradcam.png"
    # original_path = f"{TEMP_DIR}/{session_id}_original.png"

    # Add images
    def save_img(img, temp_path):
        img.save(temp_path)

    # original_img_bytes = original_img_bytes.split(",")[-1]  # remove "data:image/png;base64," if present
    # original_img_bytes = base64.b64decode(original_img_bytes)

    # gradcam_img_bytes = gradcam_img_bytes.split(",")[-1]  # remove "data:image/png;base64," if present
    # gradcam_img_bytes = base64.b64decode(gradcam_img_bytes)
    
    # with open(pred_path, "w") as f:
    #     f.write(class_label)

    # Write bytes to file
    # with open(original_path, "wb") as f:
    #     f.write(original_img_bytes)

    # with open(gradcam_path, "wb") as f:
    #     f.write(gradcam_img_bytes)

    
    # save_img(original_img_bytes, original_path)
    # save_img(heatmap_image, gradcam_path)
    # save_img(image_bytes, original_path)

    # # Store metadata in DB
    # db = SessionLocal()
    # temp_file = TempFile(
    #     session_id=session_id,
    #     prediction_path=pred_path,
    #     gradcam_path=gradcam_path,
    #     original_path=original_path,
    #     created_at=datetime.now(timezone.utc)  # 🔹 Safe UTC time
    # )
    # db.add(temp_file)
    # db.commit()
    # db.close()

    return {
        "prediction": class_label,
        "confidence": float(prediction[0][class_index]),
        "probabilities": prediction[0].tolist(),
        "original_img_array": original_image,
        "heatmap_array": heatmap_image,
        "original_img_base64": encode_image_to_base64(original_image), #Give to frontend
        "heatmap_img_base64": encode_array_to_base64(heatmap_image)
    }

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2
import io
import os
from fastapi.responses import StreamingResponse
from openai import OpenAI
from datetime import datetime
from fpdf import FPDF
import base64
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# Load the trained model once when the server starts
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "models", "best_model.h5")
model = tf.keras.models.load_model(os.path.abspath(model_path))
last_conv_layer_name = "mixed10"

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI()
class_labels = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']

def preprocess_image(image_data):
    image = Image.open(io.BytesIO(image_data))
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    if image.shape[-1] == 4:
        image = image[..., :3]
    return np.expand_dims(image, axis=0)

def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]
    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def apply_gradcam(image_data, heatmap):
    img = Image.open(io.BytesIO(image_data)).resize((224, 224))
    img = np.array(img)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
    _, buffer = cv2.imencode('.jpg', superimposed_img)
    return buffer.tobytes()

@app.post("/predict/")
async def predict(image: UploadFile = File(...), name: str = Form(...), age: int = Form(...)):
    img_data = await image.read()
    img_array = preprocess_image(img_data)
    predictions = model.predict(img_array)[0]
    pred_class = class_labels[np.argmax(predictions)]
    confidence = float(np.max(predictions))

    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)
    gradcam_img = apply_gradcam(img_data, heatmap)

    return {
        "prediction": pred_class,
        "confidence": confidence,
        "original_image": base64.b64encode(img_data).decode(),
        "gradcam_image": base64.b64encode(gradcam_img).decode()
    }

@app.post("/generate_report/")
async def generate_report(name: str = Form(...), age: int = Form(...), prediction: str = Form(...), confidence: float = Form(...)):
    report_text = f"Patient Name: {name}\nAge: {age}\nDiagnosis: {prediction}\nConfidence: {confidence:.2f}\nDate: {datetime.now().strftime('%Y-%m-%d %H:%M')}"

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    for line in report_text.split("\n"):
        pdf.cell(200, 10, txt=line, ln=True)

    buffer = io.BytesIO()
    pdf.output(buffer)
    buffer.seek(0)

    return StreamingResponse(buffer, media_type="application/pdf", headers={"Content-Disposition": "attachment; filename=report.pdf"})

class ChatInput(BaseModel):
    message: str

@app.post("/chat/")
async def chat(input: ChatInput):
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant for medical patients."},
            {"role": "user", "content": input.message},
        ]
    )
    return {"response": response.choices[0].message.content}

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
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
import smtplib
from email.message import EmailMessage
from fastapi.responses import FileResponse

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

class ReportRequest(BaseModel):
    name: str
    age: int
    gender: str
    email: EmailStr
    symptoms: str
    prediction: str
    original_image: str
    gradcam_image: str

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

# def generate_llm_prompt(patient_info):
#     prompt = f"""
#         You are a medical assistant. A CNN model has analyzed an MRI scan and predicted the tumor type as **{patient_info.get('prediction')}**.
#         Patient information:
#         - Age: {patient_info.get('Age')}
#         - Gender: {patient_info.get('Gender')}
#         - Symptoms: {patient_info.get('Symptoms')}
#         As a medical expert, please examine the patient’s condition by ﬁrst identifying any abnormal. Next, critically analyze there their impact, and clearly state ﬁnal diagnosis regarding what might be causing the clinical deterioration. Finally give a brief description.
#         Provide a clear medical description of the predicted tumor type, followed by helpful precautions for the patient.

#         Format:
#         Description: ...
#         Precautions: ...
#     """
#     return prompt

# def generate_report(patient_info, filename: str):
#     prompt = generate_llm_prompt(patient_info)
#     response = client.responses.create(
#             model="gpt-4.1-mini",
#             input=[
#                     {"role": "user", "content": prompt},
#                     {"role": "user", "content": [{
#                             "type": "input_image",
#                             "image_url": f"data:image/jpeg;base64,{base64_image}",
#                         }],
#                     },
#                 ],
#             temperature=0.5,
#         )
#     result = response.output_text

#     if "Precautions:" in result:
#         parts = result.split("Precautions:")
#         desc = parts[0].replace("Description:", "").strip()
#         precautions = parts[1].strip()
#     else:
#         desc = result.strip()
#         precautions = "Not provided."

#     # return {
#     #     "description": desc,
#     #     "precautions": precautions,
#     # }

#     pdf = FPDF()
#     pdf.add_page()
#     pdf.set_font("Arial", 'B', 16)

#     pdf.cell(0, 10, "Tumor Classification Report", ln=True, align='C')
#     pdf.set_font("Arial", '', 12)
#     pdf.ln(10)
#     for key, value in patient_info.items():
#         pdf.cell(0, 10, f"{key}: {value}", ln=True)
#     pdf.ln(5)
#     pdf.set_font("Arial", 'B', 12)
#     pdf.cell(0, 10, f"Prediction: {patient_info.get('prediction')}", ln=True)
#     pdf.set_font("Arial", '', 11)
#     pdf.multi_cell(0, 10, f"\nDescription:\n{desc}")
#     pdf.multi_cell(0, 10, f"\nPrecautions:\n{precautions}")

#     # Add images
#     def save_img(img, filename):
#         temp_path = f"./tmp/{filename}"
#         img.save(temp_path)
#         return temp_path
    
#     ori_path = save_img(original_img, "original.png")
#     grad_path = save_img(gradcam_img, "gradcam.png")

#     image_width = 90
#     image_height = 60  # adjust based on aspect ratio
#     spacing = 5
#     page_height = 297  # A4 size in mm
#     bottom_margin = 10

#     # Get current vertical position
#     current_y = pdf.get_y()

#     # Ensure there is enough space for images
#     if current_y + image_height + bottom_margin > page_height:
#         pdf.add_page()
#         current_y = pdf.get_y()

#     # Insert both images on the same row
#     pdf.image(ori_path, x=10, y=current_y + spacing, w=image_width, h=image_height)
#     pdf.image(grad_path, x=110, y=current_y + spacing, w=image_width, h=image_height)

#     # Optional: move cursor below the images if you want to add more content
#     pdf.set_y(current_y + image_height + spacing + 10)

#     pdf.output(output_path)
#     return output_path

#     # Images
#     for label, img_b64 in [("Original Image", data.original_image), ("Grad-CAM Image", data.gradcam_image)]:
#         image_path = f"/tmp/{label.replace(' ', '_')}.jpg"
#         with open(image_path, "wb") as f:
#             f.write(base64.b64decode(img_b64))
#         pdf.add_page()
#         pdf.cell(200, 10, txt=label, ln=True)
#         pdf.image(image_path, x=10, y=30, w=180)

#     pdf.output(filename)

# def send_email_report(to_email: str, filename: str):
#     msg = EmailMessage()
#     now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#     msg['Subject'] = f"Brain Tumor Report - {now}"
#     msg['From'] = "your-email@example.com"
#     msg['To'] = to_email
#     msg.set_content(f"Dear user,\n\nPlease find attached the brain tumor report.\nTimestamp: {now}\n\nRegards,\nMedical AI Team")

#     with open(filename, 'rb') as f:
#         msg.add_attachment(f.read(), maintype='application', subtype='pdf', filename=os.path.basename(filename))

#     with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
#         smtp.login("your-email@example.com", "your-email-password")
#         smtp.send_message(msg)

# @app.post("/generate_report")
# async def generate_report(data: ReportRequest, background_tasks: BackgroundTasks):
#     description = generate_ai_description(data.symptoms, data.prediction)
#     filename = f"/tmp/{data.name.replace(' ', '_')}_report.pdf"
#     generate_pdf(data, description, filename)
#     return {"description": description, "pdf_path": filename}

# @app.get("/download_report")
# def download_report(path: str):
#     return FileResponse(path, filename=os.path.basename(path), media_type='application/pdf')

# @app.post("/email_report")
# async def email_report(email: EmailStr, path: str, background_tasks: BackgroundTasks):
#     background_tasks.add_task(send_email_report, email, path)
#     return {"message": f"Email will be sent to {email}"}

@app.post("/predict/")
async def predict(image: UploadFile = File(...), name: str = Form(...), age: int = Form(...), gender: str = Form(...), email: str = Form(...), symptoms: str = Form(...)):
    img_data = await image.read()
    img_array = preprocess_image(img_data)
    predictions = model.predict(img_array)[0]
    pred_class = class_labels[np.argmax(predictions)]
    confidence = float(np.max(predictions))

    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)
    gradcam_img = apply_gradcam(img_data, heatmap)

    return {
        "name": name,
        "age": age,
        "gender": gender,
        "email": email,
        "symptoms": symptoms,
        "prediction": pred_class,
        "confidence": confidence,
        "original_image": base64.b64encode(img_data).decode(),
        "gradcam_image": base64.b64encode(gradcam_img).decode()
    }

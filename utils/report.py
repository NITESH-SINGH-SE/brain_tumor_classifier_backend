from fpdf import FPDF
from openai import OpenAI
from PIL import Image
import numpy as np
import io
import cv2
import base64
from dotenv import load_dotenv
import smtplib
from email.message import EmailMessage
from fastapi.responses import JSONResponse
from datetime import datetime

load_dotenv() # take environment variables

# ========== OpenAI Setup ==========
client = OpenAI()  # Ensure you have set OPENAI_API_KEY env variable or set here

class_labels = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']

def encode_image_to_base64(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode()

def encode_array_to_base64(image_array: np.ndarray) -> str:
    _, buffer = cv2.imencode('.jpg', image_array)
    return base64.b64encode(buffer).decode()

def generate_llm_prompt(class_label, patient_info):
    prompt = f"""
        You are a medical assistant. A CNN model has analyzed an MRI scan and predicted the tumor type as **{class_label}**.
        Patient information:
        - Age: {patient_info.get('Age')}
        - Gender: {patient_info.get('Gender')}
        - Symptoms: {patient_info.get('Symptoms')}
        As a medical expert, please examine the patient’s condition by ﬁrst identifying any abnormal. Next, critically analyze there their impact, and clearly state ﬁnal diagnosis regarding what might be causing the clinical deterioration. Finally give a brief description.
        Provide a clear medical description of the predicted tumor type, followed by helpful precautions for the patient.

        Format:
        Description: ...
        Precautions: ...
    """
    return prompt

def generate_report(patient_info, original_img, gradcam_img, prediction):
    base64_image = encode_image_to_base64(original_img)
    prompt = generate_llm_prompt(prediction, patient_info)
    response = client.responses.create(
            model="gpt-4.1-mini",
            input=[
                    {"role": "user", "content": prompt},
                    {"role": "user", "content": [{
                            "type": "input_image",
                            "image_url": f"data:image/jpeg;base64,{base64_image}",
                        }],
                    },
                ],
            temperature=0.5,
        )
    result = response.output_text

    if "Precautions:" in result:
        parts = result.split("Precautions:")
        desc = parts[0].replace("Description:", "").strip()
        precautions = parts[1].strip()
    else:
        desc = result.strip()
        precautions = "Not provided."

    return {
        "description": desc,
        "precautions": precautions,
    }

    # generate_pdf_report(patient_info, prediction, desc, precautions, original_img, gradcam_img)

def generate_pdf_report(patient_info, prediction, description, precautions, original_img, gradcam_img, output_path="tmp/report.pdf"):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "Tumor Classification Report", ln=True, align='C')
    pdf.set_font("Arial", '', 12)
    pdf.ln(10)
    for key, value in patient_info.items():
        pdf.cell(0, 10, f"{key}: {value}", ln=True)
    pdf.ln(5)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, f"Prediction: {prediction}", ln=True)
    pdf.set_font("Arial", '', 11)
    pdf.multi_cell(0, 10, f"\nDescription:\n{description}")
    pdf.multi_cell(0, 10, f"\nPrecautions:\n{precautions}")

    # Add images
    def save_img(img, filename):
        temp_path = f"./tmp/{filename}"
        img.save(temp_path)
        return temp_path
    
    ori_path = save_img(original_img, "original.png")
    grad_path = save_img(gradcam_img, "gradcam.png")

    image_width = 90
    image_height = 60  # adjust based on aspect ratio
    spacing = 5
    page_height = 297  # A4 size in mm
    bottom_margin = 10

    # Get current vertical position
    current_y = pdf.get_y()

    # Ensure there is enough space for images
    if current_y + image_height + bottom_margin > page_height:
        pdf.add_page()
        current_y = pdf.get_y()

    # Insert both images on the same row
    pdf.image(ori_path, x=10, y=current_y + spacing, w=image_width, h=image_height)
    pdf.image(grad_path, x=110, y=current_y + spacing, w=image_width, h=image_height)

    # Optional: move cursor below the images if you want to add more content
    pdf.set_y(current_y + image_height + spacing + 10)

    pdf.output(output_path)
    return output_path

def send_email_report(name, email, prediction, output_path):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # Email setup
    sender_email = "your_email@example.com"
    sender_password = "your_app_password"  # Use app-specific password if Gmail
    subject = "Your Tumor Classification Report"
    body = f"""
Dear {name},

Please find attached your tumor classification report generated on {now}.

Prediction: {prediction}

Thank you,
Tumor Classification Team
"""

    msg = EmailMessage()
    msg['Subject'] = subject
    msg['From'] = sender_email
    msg['To'] = email
    msg.set_content(body)

    with open(output_path, 'rb') as f:
        file_data = f.read()
        msg.add_attachment(file_data, maintype='application', subtype='pdf', filename="report.pdf")

    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(sender_email, sender_password)
            smtp.send_message(msg)
        return JSONResponse({"message": "Email sent successfully"})
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": f"Email failed: {str(e)}"})
from pydantic import BaseModel, EmailStr, Field
from datetime import datetime
from typing import Optional


class PatientBase(BaseModel):
    name: str = Field(..., example="John Doe")
    age: int = Field(..., ge=0, le=120, example=30)
    gender: str = Field(..., example="Male")
    email: EmailStr
    symptoms: str = Field(..., example="Fever, cough, fatigue")


class PatientCreate(PatientBase):
    original_image: str  # base64 encoded
    gradcam_image: str   # base64 encoded
    prediction: str
    confidence: float = Field(..., ge=0.0, le=1.0)


class Patient(PatientBase):
    id: int
    prediction: str
    confidence: float
    original_image: str
    gradcam_image: str
    created_at: datetime

    class Config:
        orm_mode = True

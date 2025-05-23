from sqlalchemy import Column, Integer, String, Float, Text, DateTime
from sqlalchemy.orm import relationship
from database.database import Base
from datetime import datetime

class Patient(Base):
    __tablename__ = "patients"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, unique=True, index=True)
    name = Column(String, index=True)
    age = Column(Integer)
    gender = Column(String)
    symptoms = Column(Text)
    prediction = Column(String)
    confidence = Column(Float)
    original_image = Column(Text)  # base64 encoded image
    gradcam_image = Column(Text)   # base64 encoded image
    created_at = Column(DateTime, default=datetime.utcnow)

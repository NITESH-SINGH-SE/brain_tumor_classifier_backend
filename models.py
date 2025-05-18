from sqlalchemy import Column, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime, timezone

Base = declarative_base()

class TempFile(Base):
    __tablename__ = "temp_files"

    session_id = Column(String, primary_key=True, index=True)
    prediction_path = Column(String, nullable=False)
    gradcam_path = Column(String, nullable=False)
    original_path = Column(String, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

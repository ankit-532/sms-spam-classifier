# src/schemas.py
# Pydantic models for API requests/responses

from pydantic import BaseModel
from typing import Dict

class SMSRequest(BaseModel):
    message: str

class SMSResponse(BaseModel):
    message: str
    classification: str
    spam_probability: float
    signals: Dict[str, bool]
    decision_logic: str
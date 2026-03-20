from pydantic import BaseModel

class TrainingResponse(BaseModel):
    status: str

from pydantic import BaseModel


class TrainingResponse(BaseModel):
    status: str


class TrainingMetrics(BaseModel):
    accuracy: float
    f1_macro: float
    loss: float

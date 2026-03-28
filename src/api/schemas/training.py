from pydantic import BaseModel


class TrainingResponse(BaseModel):
    status: str


class TrainingMetrics(BaseModel):
    accuracy: float
    f1_macro: float
    loss: float
    precision_negative: float = 0.0
    recall_negative: float = 0.0
    f1_negative: float = 0.0
    precision_neutral: float = 0.0
    recall_neutral: float = 0.0
    f1_neutral: float = 0.0
    precision_positive: float = 0.0
    recall_positive: float = 0.0
    f1_positive: float = 0.0

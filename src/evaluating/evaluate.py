from transformers import pipeline
from datasets import load_dataset
from sklearn.metrics import f1_score, accuracy_score
from utils.config import load_config

def evaluate_model(pipeline, test_dataset, label2id):
    y_true, y_pred = [], []

    for data in test_dataset:
        result = pipeline(data["text"])
        best   = max(result, key=lambda x: x["score"])
        y_true.append(data["label"])
        y_pred.append(label2id[best["label"].lower()])

    return {
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "accuracy": accuracy_score(y_true, y_pred)
    }
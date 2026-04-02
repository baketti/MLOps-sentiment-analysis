from prometheus_client import Counter, Histogram, Gauge


# Runtime metrics

predictions_counter = Counter(
    "sentiment_predictions_total",
    "Total predictions by label",
    ["predicted_label"],
)

inference_latency = Histogram(
    "sentiment_inference_latency_seconds",
    "Inference latency in seconds",
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.5],
)

prediction_confidence = Histogram(
    "sentiment_prediction_confidence",
    "Confidence score of sentiment predictions",
    buckets=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0],
)


# Training metrics

model_f1_macro = Gauge(
    "model_eval_f1_macro",
    "Last Training F1 Macro",
)

model_accuracy = Gauge(
    "model_eval_accuracy",
    "Last Training Accuracy",
)

model_eval_loss = Gauge(
    "model_eval_loss",
    "Last Training Loss",
)

model_precision_per_class = Gauge(
    "model_eval_precision_per_class",
    "Per-class precision from last training",
    ["label"],
)

model_recall_per_class = Gauge(
    "model_eval_recall_per_class",
    "Per-class recall from last training",
    ["label"],
)

model_f1_per_class = Gauge(
    "model_eval_f1_per_class",
    "Per-class F1 score from last training",
    ["label"],
)


# Drift metrics

drift_score = Gauge(
    "sentiment_drift_score",
    "Confidence drift z-score vs training reference (updated by drift detection DAG)",
)

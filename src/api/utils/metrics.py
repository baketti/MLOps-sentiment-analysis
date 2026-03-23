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

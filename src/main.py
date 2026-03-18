from ingestion.download import load_dataset_for_fine_tuning
from dotenv import load_dotenv
from utils.config import load_config
from evaluating.evaluate import evaluate_model
from predicting.make_prediction import create_sentiment_pipeline, make_prediction
from preprocessing.load_dataset import load_and_get_sentiment_analysis_dataset

load_dotenv()
config = load_config()

HF_MODEL_NAME = config.get("hf_model", {}).get("name", None)
LABEL2ID = config.get("hf_model", {}).get("label2id", {"negative": 0, "neutral": 1, "positive": 2})

if HF_MODEL_NAME is None:
    raise ValueError("HF_MODEL_NAME must be specified in the configuration file.")

pipeline = create_sentiment_pipeline(HF_MODEL_NAME)

X, y = load_and_get_sentiment_analysis_dataset()

test_dataset = [{"text": t, "label": LABEL2ID[l]} for t, l in zip(X.to_list(), y.to_list())]


"""print(test_dataset[:5])


metrics = evaluate_model(pipeline, test_dataset, LABEL2ID)

print(metrics)"""

print(make_prediction("I love this product!", HF_MODEL_NAME))
print(make_prediction("This is the worst experience I've ever had.", HF_MODEL_NAME))
print(make_prediction("It's okay, not great but not bad either.", HF_MODEL_NAME))
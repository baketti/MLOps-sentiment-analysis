import mlflow
from preprocessing.load_dataset import load_and_get_sentiment_analysis_dataset, load_sentiment_analysis_dataset
from ingestion.download import load_dataset_for_fine_tuning
import os
import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv
from utils.config import load_config

load_dotenv()
config = load_config()

HF_MODEL_NAME = config.get("hf_model", {}).get("name", "cardiffnlp/twitter-roberta-base-sentiment-latest")
MODEL_OUTPUT_DIR = os.getenv("MODEL_OUTPUT_DIR")

NUM_LABELS = config.get("hf_model", {}).get("num_labels", 3)
LABEL2ID   = config.get("hf_model", {}).get("label2id", {"negative": 0, "neutral": 1, "positive": 2})
ID2LABEL   = {v: k for k, v in LABEL2ID.items()}

tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME)
model: AutoModelForSequenceClassification = AutoModelForSequenceClassification.from_pretrained(
    HF_MODEL_NAME,
    num_labels=NUM_LABELS,
    ignore_mismatched_sizes=True,
)


def get_train_test_datasets() -> tuple[Dataset, Dataset]:
    """
        Loads the sentiment analysis dataset, splits it into training and testing sets, and converts them into Hugging Face Datasets.
        Returns:
            tuple: A tuple containing the training and testing datasets as Hugging Face Datasets."""
    dataset = load_sentiment_analysis_dataset()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Convert to DataFrame to fit the expected input type of RandomOverSampler
    X_train_df = X_train.to_frame()
    y_train_df = y_train.to_frame()

    # Apply RandomOverSampler to balance the training dataset
    ros = RandomOverSampler(random_state=42)
    X_train_res, y_train_res = ros.fit_resample(X_train_df, y_train_df)
    X_train_res = X_train_res.squeeze()  # Convert back to Series
    y_train_res = y_train_res.squeeze()  # Convert back to Series

    train_dataset = Dataset.from_pandas(
        pd.DataFrame({"text": X_train_res, "label": y_train_res}).reset_index(drop=True)
    )
    test_dataset = Dataset.from_pandas(
        pd.DataFrame({"text": X_test, "label": y_test}).reset_index(drop=True)
    )

    return train_dataset, test_dataset


def tokenize_dataset(dataset: DatasetDict, tokenizer: AutoTokenizer) -> DatasetDict:
    """
        Tokenizes the text in the training and testing datasets.
        Params:
            train_dataset (Dataset): The training dataset as a Hugging Face Dataset.
            test_dataset (Dataset): The testing dataset as a Hugging Face Dataset.
        Returns:
            tuple: A tuple containing the tokenized training and testing datasets as Hugging Face Datasets
    """
    def tokenize(batch):
        tokenized = tokenizer(
            batch["text"],
            truncation=True,
            padding=False,
            max_length=128,
        )
        tokenized["labels"] = batch["label"]
        return tokenized

    cols_to_remove = ["text", "label"]
    dataset = dataset.map(tokenize, batched=True, remove_columns=cols_to_remove)
    dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    return dataset


def fine_tune_model(dataset: DatasetDict, model: AutoModelForSequenceClassification, tokenizer: AutoTokenizer) -> Trainer:
    """    
        Fine-tunes the pre-trained model on the training dataset and evaluates it on the testing dataset.
        Params:
            train_dataset (Dataset): The tokenized training dataset as a Hugging Face Dataset.
            test_dataset (Dataset): The tokenized testing dataset as a Hugging Face Dataset.
            model (AutoModelForSequenceClassification): The pre-trained model to be fine-tuned.
            tokenizer (AutoTokenizer): The tokenizer used for tokenizing the datasets.
        Returns:
            Trainer: The Trainer object after fine-tuning the model.
    """
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy_score(labels, preds),
            "f1_macro": f1_score(labels, preds, average="macro"),
        }

    training_args = TrainingArguments(
        output_dir=MODEL_OUTPUT_DIR,
        report_to="mlflow",
        num_train_epochs=1,
        eval_strategy="epoch",       
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    print(trainer.model)

    print(trainer.evaluate(dataset["test"]))
    return trainer


def save_and_push_model_on_hf_hub(trainer: Trainer, tokenizer: AutoTokenizer, model_output_dir: str, hub_name: str, push_to_hub: bool=False) -> None:
    """
        Saves the fine-tuned model and tokenizer to the specified directory and optionally pushes it to the Hugging Face Hub.
        Params:
            trainer (Trainer): The Trainer object after fine-tuning the model.
            tokenizer (AutoTokenizer): The tokenizer used for tokenizing the datasets.
            model_output_dir (str): The directory where the model and tokenizer will be saved.
            hub_name (str): The name of the model on the Hugging Face Hub.
            push_to_hub (bool): Whether to push the model to the Hugging Face Hub.
    """
    trainer.save_model(model_output_dir)
    tokenizer.save_pretrained(model_output_dir)

    if push_to_hub:
        trainer.push_to_hub(hub_name)


def train_and_save_model(model: AutoModelForSequenceClassification, tokenizer: AutoTokenizer, model_output_dir: str) -> None:
    """
        Orchestrates the entire process of loading the dataset, tokenizing it, fine-tuning the model, and saving/pushing it to the Hugging Face Hub.
        Params:
            model (AutoModelForSequenceClassification): The pre-trained model to be fine-tuned.
            tokenizer (AutoTokenizer): The tokenizer used for tokenizing the datasets.
            model_output_dir (str): The directory where the model and tokenizer will be saved.
    """
    dataset = load_dataset_for_fine_tuning(config['data'])
    dataset = tokenize_dataset(dataset, tokenizer)
    trainer = fine_tune_model(dataset, model, tokenizer)
    save_and_push_model_on_hf_hub(
        trainer,
        tokenizer,
        model_output_dir,
        hub_name="sentiment-analysis-bert-finetuned",
        #push_to_hub=True
    )

if __name__ == "__main__":
    train_and_save_model(model, tokenizer, MODEL_OUTPUT_DIR)
    print("Tracking URI:", mlflow.get_tracking_uri())
    print("Experiment:", mlflow.get_experiment_by_name("sentiment-finetuning"))
    mlflow.end_run()
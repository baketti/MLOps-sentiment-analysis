import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from loading.load_dataset import KaggleDatasetLoader
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from evaluating.evaluate import evaluate_hf_fine_tuned_model
from utils.exceptions import InvalidDatasetStructureError, LoadingDatasetError, FineTuningError, PushingToHubError

def get_train_test_datasets(dataset_name: str, file_path: str) -> tuple[Dataset, Dataset]:
    """
        Loads the sentiment analysis dataset, splits it into training and testing sets, and converts them into Hugging Face Datasets.
        Params:
            dataset_name (str): The name of the Kaggle dataset to load.
            file_path (str): The path to the specific file within the Kaggle dataset to load
        Returns:
            tuple: A tuple containing the training and testing datasets as Hugging Face Datasets.
        Raises:
            LoadingDatasetError: If there is an error loading the dataset from Kaggle.
            InvalidDatasetStructureError: If the loaded dataset does not contain the required 'text' and 'sentiment' columns.
    """
    try:
        kdl = KaggleDatasetLoader(dataset_name, file_path)

        X, y = kdl.load_and_get_sentiment_analysis_dataset()

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        train_dataset = Dataset.from_pandas(
            pd.DataFrame({"text": X_train, "label": y_train}).reset_index(drop=True)
        )
        test_dataset = Dataset.from_pandas(
            pd.DataFrame({"text": X_test, "label": y_test}).reset_index(drop=True)
        )
    except LoadingDatasetError as e:
        raise
    except InvalidDatasetStructureError as e:
        raise

    return train_dataset, test_dataset


def tokenize_train_test_datasets(train_dataset: Dataset, test_dataset: Dataset, tokenizer: AutoTokenizer, label2id: dict) -> tuple[Dataset, Dataset]:
    """
        Tokenizes the text in the training and testing datasets.
        Params:
            train_dataset (Dataset): The training dataset as a Hugging Face Dataset.
            test_dataset (Dataset): The testing dataset as a Hugging Face Dataset.
            tokenizer (AutoTokenizer): The tokenizer used for tokenizing the datasets.
            label2id (dict): A mapping from label names to label IDs.
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
        tokenized["labels"] = [label2id[lbl.lower()] for lbl in batch["label"]]
        return tokenized

    train_dataset = train_dataset.map(tokenize, batched=True)
    test_dataset = test_dataset.map(tokenize, batched=True)

    train_dataset = train_dataset.remove_columns(["text", "label"])
    test_dataset  = test_dataset.remove_columns(["text", "label"])

    train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    test_dataset.set_format("torch",  columns=["input_ids", "attention_mask", "labels"])

    return train_dataset, test_dataset


def fine_tune_model(
        train_dataset: Dataset, test_dataset: Dataset, model: AutoModelForSequenceClassification, tokenizer: AutoTokenizer, model_output_dir: str, hub_model_id: str
    ) -> Trainer:
    """
        Fine-tunes the pre-trained model on the training dataset and evaluates it on the testing dataset.
        Params:
            train_dataset (Dataset): The tokenized training dataset as a Hugging Face Dataset.
            test_dataset (Dataset): The tokenized testing dataset as a Hugging Face Dataset.
            model (AutoModelForSequenceClassification): The pre-trained model to be fine-tuned.
            tokenizer (AutoTokenizer): The tokenizer used for tokenizing the datasets.
            model_output_dir (str): The directory where the fine-tuned model and tokenizer will be saved.
            hub_model_id (str): The model repository ID to be used for pushing the model to the Hugging Face Hub.
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
        output_dir=model_output_dir,
        hub_model_id=hub_model_id,
        report_to="mlflow",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    try:
        trainer.train()
    except Exception as e:
        raise FineTuningError(f"Error during fine-tuning: {e}")

    return trainer


def save_and_push_model_on_hf_hub(
        trainer: Trainer, tokenizer: AutoTokenizer, model_output_dir: str, quality_thresholds: dict
    ) -> None:
    """
        Saves the fine-tuned model and tokenizer to the specified directory and pushes it to the Hugging Face Hub if and only if it passes the quality gate.
        Params:
            trainer (Trainer): The Trainer object after fine-tuning the model.
            tokenizer (AutoTokenizer): The tokenizer used for tokenizing the datasets.
            model_output_dir (str): The directory where the model and tokenizer will be saved.
            quality_thresholds (dict): A dictionary containing the minimum thresholds for accuracy and F1 score to decide whether to push the model to the Hugging Face Hub.
    """
    trainer.save_model(model_output_dir)
    tokenizer.save_pretrained(model_output_dir)

    is_ready_to_push, metrics = evaluate_hf_fine_tuned_model(trainer, quality_thresholds)

    if is_ready_to_push:
        commit_message = (
            #f"dataset=tweet_eval | "
            #f"train_size={train_config['train_size']} | "
            #f"epochs={train_config['num_epochs']} | "
            f"f1_macro={metrics['f1_macro']:.2f} | "
            f"accuracy={metrics['accuracy']:.4f}"
        )

        try:
            trainer.push_to_hub(commit_message)
        except Exception as e:
            raise PushingToHubError(f"Error pushing model to Hugging Face Hub: {e}")

def train_and_save_model(model: AutoModelForSequenceClassification, tokenizer: AutoTokenizer, model_output_dir: str) -> None:
    """
        Orchestrates the entire process of loading the dataset, tokenizing it, fine-tuning the model, and saving/pushing it to the Hugging Face Hub.
        Params:
            model (AutoModelForSequenceClassification): The pre-trained model to be fine-tuned.
            tokenizer (AutoTokenizer): The tokenizer used for tokenizing the datasets.
            model_output_dir (str): The directory where the model and tokenizer will be saved.
    """
    train_dataset, test_dataset = get_train_test_datasets()
    train_dataset, test_dataset = tokenize_train_test_datasets(train_dataset, test_dataset)

    trainer = fine_tune_model(train_dataset, test_dataset, model, tokenizer)

    save_and_push_model_on_hf_hub(
        trainer,
        tokenizer,
        model_output_dir,
        hub_name="sentiment-analysis-bert-finetuned",
        push_to_hub=True
    )

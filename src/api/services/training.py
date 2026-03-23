from training.train_model import (
    get_train_test_datasets,
    tokenize_train_test_datasets,
    fine_tune_model,
    save_and_push_model_on_hf_hub,
)


def train_and_save_model(app_config: dict) -> dict:
    """
        Orchestrates the entire process of loading the dataset,
        tokenizing it, fine-tuning the model, and saving/pushing it
        to the Hugging Face Hub.
        Params:
            app_config (dict): The application configuration dictionary
                containing all necessary parameters and objects for
                training and saving the model.
        Returns:
            dict: A dictionary containing the evaluation metrics
                (accuracy, f1_macro, loss).
    """
    model = app_config.get("model_object")
    tokenizer = app_config.get("tokenizer_object")
    dataset_name = app_config.get("kaggle_dataset").get("name")
    file_path = app_config.get("kaggle_dataset").get("file_path")
    label2id = app_config.get("hf_model").get("label2id")
    model_output_dir = app_config.get("model_output_dir")
    hub_model_id = app_config.get("hf_hub_model_id")
    quality_thresholds = app_config.get("quality_thresholds")

    train_dataset, test_dataset = get_train_test_datasets(
        dataset_name, file_path
    )
    train_dataset, test_dataset = tokenize_train_test_datasets(
        train_dataset, test_dataset, tokenizer, label2id
    )

    trainer = fine_tune_model(
        train_dataset, test_dataset, model, tokenizer,
        model_output_dir, hub_model_id
    )

    return save_and_push_model_on_hf_hub(
        trainer,
        tokenizer,
        model_output_dir,
        quality_thresholds
    )

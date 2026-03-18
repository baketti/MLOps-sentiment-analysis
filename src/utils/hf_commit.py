def create_hf_hub_commit_message(
        #train_config: dict, 
        metrics: dict) -> str:
    """
        Creates a commit message for pushing the fine-tuned model to the Hugging Face Hub, including the model name and its performance metrics.
        Params:
            train_config (dict): A dictionary containing the training configuration, such as the training dataset size and the number of epochs.
            metrics (dict): A dictionary containing the performance metrics of the model, such as F1 score and accuracy.
        Returns:
            str: A formatted commit message for the Hugging Face Hub.
    """
    commit_message = (
        #f"dataset=tweet_eval | "
        #f"train_size={train_config['train_size']} | "
        #f"epochs={train_config['num_epochs']} | "
        f"f1_macro={metrics['f1_macro']:.2f} | "
        f"accuracy={metrics['accuracy']:.4f}"
    )

    return commit_message
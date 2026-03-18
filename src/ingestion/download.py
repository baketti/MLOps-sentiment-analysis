from datasets import load_dataset, DatasetDict

def load_dataset_for_fine_tuning(config):
    dataset = load_dataset(
        config.get("dataset_name", "tweet_eval"),
        config.get("dataset_split", "sentiment"),
    )

    train = dataset["train"].shuffle(seed=config.get("random_state", 42))
    train = train.select(range(config.get("train_size", 5000)))

    return DatasetDict({
        "train":      train,
        "validation": dataset["validation"],
        "test":       dataset["test"],
    })
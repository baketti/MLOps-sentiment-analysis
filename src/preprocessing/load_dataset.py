import kagglehub
from kagglehub import KaggleDatasetAdapter
from pandas import DataFrame, Series 
from datasets import load_dataset

def load_sentiment_analysis_dataset():
    """
        Loads the sentiment analysis dataset from Kaggle.
        Returns:
            pd.DataFrame: A DataFrame containing the sentiment analysis dataset.
    """
    file_path: str = "sentiment_analysis.csv"

    df: DataFrame = kagglehub.load_dataset(
        KaggleDatasetAdapter.PANDAS,
        "mdismielhossenabir/sentiment-analysis",
        file_path
    )

    return df


def get_sentiment_analysis_dataset(df: DataFrame) -> tuple[Series, Series]:
    """
        Prepares the dataset for model training by splitting it into features and labels.
        Params:
            df (pd.DataFrame): The DataFrame containing the sentiment analysis dataset.
        Returns:
            tuple: A tuple containing the features (X: pd.Series) and labels (y: pd.Series).
    """
    X: Series = df['text']
    y: Series = df['sentiment']

    return (X, y)


def load_and_get_sentiment_analysis_dataset() -> tuple[Series, Series]:
    """
        Loads the sentiment analysis dataset and prepares it for model training.
        Returns:
            tuple: A tuple containing the features (X: pd.Series) and labels (y: pd.Series).
    """
    df = load_sentiment_analysis_dataset()
    return get_sentiment_analysis_dataset(df)

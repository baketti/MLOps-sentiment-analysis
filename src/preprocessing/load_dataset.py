import kagglehub
from kagglehub import KaggleDatasetAdapter

def load_sentiment_analysis_dataset():
    """
        Loads the sentiment analysis dataset from Kaggle.
        Returns:
            pd.DataFrame: A DataFrame containing the sentiment analysis dataset.
    """
    file_path = "sentiment_analysis.csv"

    df = kagglehub.load_dataset(
        KaggleDatasetAdapter.PANDAS,
        "mdismielhossenabir/sentiment-analysis",
        file_path
    )

    return df

def get_sentiment_analysis_dataset(df):
    """
        Prepares the dataset for model training by splitting it into features and labels.
        Params:
            df (pd.DataFrame): The DataFrame containing the sentiment analysis dataset.
        Returns:
            tuple: A tuple containing the features (X: pd.Series) and labels (y: pd.Series).
    """
    X = df['text']
    y = df['sentiment']

    return (X, y)

def load_and_get_sentiment_analysis_dataset():
    """
        Loads the sentiment analysis dataset and prepares it for model training.
        Returns:
            tuple: A tuple containing the features (X: pd.Series) and labels (y: pd.Series).
    """
    df = load_sentiment_analysis_dataset()
    return get_sentiment_analysis_dataset(df)
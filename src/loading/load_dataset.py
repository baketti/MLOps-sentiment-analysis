import kagglehub
from kagglehub import KaggleDatasetAdapter
from pandas import DataFrame, Series 


class KaggleDatasetLoader:
    """
        A class to load dataset from Kaggle using the kagglehub library.
        
        Attributes:
            dataset_name (str): The name of the Kaggle dataset to load.
            file_path (str): The path to the specific file within the Kaggle dataset to load
    """

    def __init__(self, dataset_name: str, file_path: str):
        """
            Initializes the KaggleDatasetLoader with the specified dataset name and file path.
        """
        self.dataset_name = dataset_name
        self.file_path = file_path


    def _load_sentiment_analysis_dataset(self) -> DataFrame:
        """
            Loads the sentiment analysis dataset from Kaggle.
            Returns:
                pd.DataFrame: A DataFrame containing the sentiment analysis dataset.
        """
        df: DataFrame = kagglehub.load_dataset(
            KaggleDatasetAdapter.PANDAS,
            self.dataset_name,
            self.file_path
        )

        return df


    def _get_sentiment_analysis_dataset(self, df: DataFrame) -> tuple[Series, Series]:
        """
            Prepares the dataset for model training by splitting it into features and labels.
            Params:
                df (pd.DataFrame): The DataFrame containing the sentiment analysis dataset.
            Returns:
                tuple: A tuple containing the features (X: pd.Series) and labels (y: pd.Series).
        """
        X: Series = df['text']
        y: Series = df['sentiment']

        return X, y


    def load_and_get_sentiment_analysis_dataset(self) -> tuple[Series, Series]:
        """
            Loads the sentiment analysis dataset and prepares it for model training.
            Returns:
                tuple: A tuple containing the features (X: pd.Series) and labels (y: pd.Series).
        """
        df = self._load_sentiment_analysis_dataset()
        return self._get_sentiment_analysis_dataset(df)

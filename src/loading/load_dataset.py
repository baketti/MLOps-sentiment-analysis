import kagglehub
from kagglehub import KaggleDatasetAdapter
from pandas import DataFrame, Series
from utils.exceptions import InvalidDatasetStructureError, LoadingDatasetError


class KaggleDatasetLoader:
    """
        A class to load dataset from Kaggle using the kagglehub library.

        Attributes:
            dataset_name (str): The name of the Kaggle dataset to load.
            file_path (str): The path to the specific file within the
                Kaggle dataset to load.
    """

    def __init__(self, dataset_name: str, file_path: str):
        """
            Initializes the KaggleDatasetLoader with the specified
            dataset name and file path.
        """
        self.dataset_name = dataset_name
        self.file_path = file_path

    def _load_sentiment_analysis_dataset(self) -> DataFrame:
        """
            Loads the sentiment analysis dataset from Kaggle.
            Returns:
                pd.DataFrame: A DataFrame containing the dataset.
        """
        try:
            df: DataFrame = kagglehub.load_dataset(
                KaggleDatasetAdapter.PANDAS,
                self.dataset_name,
                self.file_path
            )

        except Exception as e:
            print(f"Error loading dataset: {e}")
            raise LoadingDatasetError(f"Error loading dataset: {e}")

        return df

    def _get_sentiment_analysis_dataset(
        self, df: DataFrame
    ) -> tuple[Series, Series]:
        """
            Prepares the dataset for model training by splitting it
            into features and labels.
            Params:
                df (pd.DataFrame): The DataFrame containing the dataset.
            Returns:
                tuple: A tuple containing the features (X: pd.Series)
                    and labels (y: pd.Series).
            Raises:
                InvalidDatasetStructureError: If the required columns
                    'text' and 'sentiment' are not present.
        """
        if 'text' not in df.columns or 'sentiment' not in df.columns:
            raise InvalidDatasetStructureError(
                "Dataset must contain 'text' and 'sentiment' columns."
            )

        X: Series = df['text']
        y: Series = df['sentiment']

        return X, y

    def load_and_get_sentiment_analysis_dataset(
        self,
    ) -> tuple[Series, Series]:
        """
            Loads the sentiment analysis dataset and prepares it
            for model training.
            Returns:
                tuple: A tuple containing the features (X: pd.Series)
                    and labels (y: pd.Series).
            Raises:
                InvalidDatasetStructureError: If the required columns
                    are not present in the DataFrame.
                LoadingDatasetError: If there is an error loading or
                    preparing the dataset.
        """
        try:
            df = self._load_sentiment_analysis_dataset()
            return self._get_sentiment_analysis_dataset(df)
        except InvalidDatasetStructureError:
            raise
        except LoadingDatasetError:
            raise
        except Exception as e:
            raise LoadingDatasetError(
                f"Unexpected error loading/preparing dataset. {e}"
            )

from sklearn.model_selection import train_test_split
from preprocessing.load_dataset import load_and_get_sentiment_analysis_dataset

X, y = load_and_get_sentiment_analysis_dataset()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training set size:", len(X_train))
print("Test set size:", len(X_test))
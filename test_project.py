import pandas as pd
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
from project import load_data
from project import data_preprocessing
from project import train_model
from project import make_prediction

def main():
    test_load_data()
    test_data_preprocessing_with_missing_values()
    test_train_model()
    test_make_predictions()

def test_load_data():
    assert load_data("asd.csv") == None
    assert load_data("asd.pdf") == None
    assert load_data("asd.doc") == None
    assert load_data("asd.txt") == None

def test_data_preprocessing_with_missing_values():
    data = pd.DataFrame([[1, None, 3]], columns=["col1", "col2", "col3"])
    processed_data = data_preprocessing(data.copy())  # Avoid modifying original data
    pd.options.mode.chained_assignment = None  # Suppress chained assignment warning
    pd.set_option('future.no_silent_downcasting', True)
    assert processed_data.isnull().sum().sum() == 1

def test_train_model():
    X_train, y_train = make_classification(n_samples=100, n_features=10, n_classes=2)
    model = train_model(X_train, y_train)
    assert isinstance(model, DecisionTreeClassifier)

def test_make_predictions():
    X_train, y_train = make_classification(n_samples=100, n_features=10, n_classes=2)
    model = train_model(X_train, y_train)
    data = pd.DataFrame([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])  # Sample data without scaling
    predictions = make_prediction(model, data)
    assert isinstance(predictions, pd.Series)
    assert predictions.name == "predicted_failure_type"


if __name__ == "__main__":
    main()

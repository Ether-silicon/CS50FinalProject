# Import library
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import seaborn as sn
import matplotlib.pyplot as plt

def main():
    # Import data
    file_path = "predictive_maintenance.csv"
    data = load_data(file_path)

    # clean data
    data = data_preprocessing(data)

    # Preprocess data
    target_variable = "Failure Type"
    X = data.drop(target_variable, axis=1)  # Features
    X = X.drop("UDI", axis = 1)
    X = X.drop("Product ID", axis = 1)
    X = X.drop("Type", axis = 1)
    X.columns = X.columns.astype(str)
    y = data[target_variable]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Train model
    model = train_model(X_train, y_train)

    # Predictions
    predictions = make_prediction(model, X_test)  # Store the predictions

    # Model evaluation & output result
    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions, average="weighted")  # Weighted average for multi-class problems
    cm = confusion_matrix(y_test, predictions)  # Confusion matrix calculation
    

    # Print output
    print("Loaded Model:", model)
    print("Predictions:", predictions, sep="\n")
    print("Accuracy:", accuracy)
    print("F1-score:", f1)
    print("Confusion Matrix:\n", cm)
    classes = ["Heat Dissipation Failure", "No failure", "Overstrain Failure", "Power Failure", "Tool Wear Failure"]
    df_cm = pd.DataFrame(cm, index = classes, columns=classes)
    plt.figure(figsize = (10,7))
    cm_plot = sn.heatmap(df_cm, annot = True)
    cm_plot.set_title("Confusion matrix")
    plt.show()


def load_data(data_path):
    # Load csv file only
    if data_path.endswith(".csv"):
        try:
            data = pd.read_csv(data_path)
        except FileNotFoundError:
            print(f"Data file not found at path: {data_path}")
            return None

    # Handle not csv file
    else:
        print("Not valid data file")
        return None

    # Popout for notice NaN value in dataset
    if data.isnull().values.any():
        print("This file contain missing values.")
    return data

def data_preprocessing(data):
    # Check for missing value
    if data.isnull().values.any():
        print("This dataset contains null value")
        # Handle missing data
        data = handle_missingdata(data)
    else:
        # Other pre-processing tools
        data = data_formatting(data)
        data = clean_ProductID(data)
        data = drop_column(data)
    return data

def handle_missingdata(data, strategy="mean"):
    if strategy == "mean":
        pd.options.mode.chained_assignment = None  # Suppress chained assignment warning
        pd.set_option('future.no_silent_downcasting', True)
        data = data.fillna(data.mean())
    else:
        print(f"Invalid strategy: {strategy}. Using mean imputation")
    return data

def data_formatting(data):
    data['Tool wear [min]'] = data['Tool wear [min]'].astype('float64')
    data['Rotational speed [rpm]'] = data['Rotational speed [rpm]'].astype('float64')
    data.rename(mapper={'Air temperature [K]': 'Air temperature',
                    'Process temperature [K]': 'Process temperature',
                    'Rotational speed [rpm]': 'Rotational speed',
                    'Torque [Nm]': 'Torque',
                    'Tool wear [min]': 'Tool wear'}, axis=1, inplace=True)
    return data

def clean_ProductID(data):
    data['Product ID'] = data['Product ID'].apply(lambda x: x[1:])
    data['Product ID'] = pd.to_numeric(data['Product ID'])
    return data

def drop_column(data):
    # Remove Random Failure that does not help in training data
    idx_RNF = data.loc[data['Failure Type']=='Random Failures'].index
    data.drop(index=idx_RNF, inplace=True)
    idx_ambiguous = data.loc[(data['Target']==1) &
                       (data['Failure Type']=='No Failure')].index
    data.drop(index=idx_ambiguous, inplace=True)
    data.reset_index(drop=True, inplace=True)
    return data

def train_model(X_train, y_train):
    # model chosen is DecisionTree
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    return model

def make_prediction(model, data, scaler=None):
    # Preprocess data if scaling is necessary
    if scaler:
        data = scaler.transform(data)

    # Make predictions using model.predict
    predictions = model.predict(data)
    return pd.Series(predictions, name="predicted_failure_type")

if __name__ == "__main__":
    main()

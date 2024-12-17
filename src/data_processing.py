import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(file_path):
    """
    Load CSV file and return a pandas DataFrame.
    :param file_path: Path to the CSV file.
    :return: DataFrame with loaded data.
    """
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        raise ValueError(f"Error loading file: {e}")

def preprocess_data(data):
    """
    Preprocess the dataset by cleaning and preparing the data.
    :param data: Input DataFrame.
    :return: Preprocessed DataFrame.
    """
    # Drop rows with missing values
    data = data.dropna()
    
    # Drop duplicates
    data = data.drop_duplicates()

    # Ensure necessary columns are present
    required_columns = ['Subject', 'Message', 'Spam/Ham']
    if not all(col in data.columns for col in required_columns):
        raise ValueError(f"Dataset must contain columns: {required_columns}")

    # Combine Subject and Message into a single text column
    data['Text'] = data['Subject'].fillna('') + ' ' + data['Message'].fillna('')

    return data[['Text', 'Spam/Ham']]

def split_data(data, test_size=0.2, random_state=42):
    """
    Split the dataset into training and testing sets.
    :param data: Input DataFrame.
    :param test_size: Proportion of the dataset to include in the test split.
    :param random_state: Random seed for reproducibility.
    :return: X_train, X_test, y_train, y_test
    """
    X = data['Text']
    y = data['Spam/Ham']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

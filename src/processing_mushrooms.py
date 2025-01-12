import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import LabelEncoder

def load_and_clean_mushrooms_data():
    """
    Loads and preprocesses the mushrooms dataset
    Returns:
        pd.DataFrame: Preprocessed dataframe
    """
    # Fetch dataset
    dataset = fetch_ucirepo(id=73)

    # Convert dotdict to DataFrame
    X = pd.DataFrame(dataset.data.features)
    y = pd.DataFrame(dataset.data.targets)

    # Combine features and target
    df = pd.concat([X, y], axis=1)

    # Handle missing values
    df.fillna(method='ffill', inplace=True)

    # Encode categorical variables
    label_encoders = {}
    for column in df.columns:
        if df[column].dtype == object:
            le = LabelEncoder()
            df[column] = le.fit_transform(df[column])
            label_encoders[column] = le

    return df

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_and_drop_columns(filepath, sheet_name, drop_columns):
    """Load dataset from Excel and drop irrelevant columns."""
    data = pd.read_excel(filepath, sheet_name=sheet_name)
    data = data.drop(columns=drop_columns, errors='ignore')
    return data

def filter_two_classes(data):
    """Keep only rows from the first two unique categories in 'Category'."""
    two_classes = data['Category'].dropna().unique()[:2]
    data_two = data[data['Category'].isin(two_classes)].copy()
    return data_two

def encode_categorical(data, exclude_col='Category'):
    """Fill missing categorical values and encode categorical columns except target."""
    categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
    categorical_cols = [col for col in categorical_cols if col != exclude_col]

    label_encoders = {}
    for col in categorical_cols:
        data[col] = data[col].fillna("Unknown")
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        label_encoders[col] = le
    return data, label_encoders

def fill_numeric_nans(data):
    """Fill missing values in numeric columns with column mean."""
    for col in data.select_dtypes(include=['number']).columns:
        data[col] = data[col].fillna(data[col].mean())
    return data

def drop_remaining_nans(data):
    """Drop any remaining rows with NaN values."""
    return data.dropna()

def prepare_features_labels(data, target_col='Category'):
    """Prepare feature matrix X and label vector y."""
    X = data.drop(columns=[target_col])
    y = data[target_col]
    y_encoder = LabelEncoder()
    y = y_encoder.fit_transform(y)
    return X, y, y_encoder

def split_data(X, y, test_size=0.3, random_state=42):
    """Split data into train and test sets with stratification."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    return X_train, X_test, y_train, y_test

def print_summary(y_encoder, X_train, X_test, y_train):
    """Print dataset summary and sample labels."""
    print(f"Classes: {list(y_encoder.classes_)}")
    print("Train set size:", X_train.shape[0])
    print("Test set size:", X_test.shape)
    print("\nSample y_train labels (encoded):", y_train[:10])

def main():
    filepath = "dataset.xlsx"
    sheet_name = "National_Health_Interview_Surve"
    drop_columns = [
        'LocationAbbr', 'LocationDesc', 'DataSource', 'Data_Value_Unit',
        'Data_Value_Type', 'Data_Value_Footnote_Symbol', 'Data_Value_Footnote',
        'Numerator', 'LocationID', 'DataValueTypeID', 'GeoLocation',
        'Geographic Level', 'StateAbbreviation'
    ]

    data = load_and_drop_columns(filepath, sheet_name, drop_columns)
    data_two = filter_two_classes(data)
    data_two, label_encoders = encode_categorical(data_two)
    data_two = fill_numeric_nans(data_two)
    data_two = drop_remaining_nans(data_two)
    X, y, y_encoder = prepare_features_labels(data_two)
    X_train, X_test, y_train, y_test = split_data(X, y)

    print_summary(y_encoder, X_train, X_test, y_train)

if __name__ == "__main__":
    main()

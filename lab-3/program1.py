import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def load_data(filepath, sheet_name, drop_columns):
    data = pd.read_excel(filepath, sheet_name=sheet_name)
    df = data.drop(columns=drop_columns, errors='ignore')
    return df


def filter_non_null_category(df):
    return df[df['Category'].notna()]


def get_two_classes(df):
    unique_categories = df['Category'].unique()
    return unique_categories[:2]


def get_categorical_columns(df, exclude_col='Category'):
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    return [col for col in categorical_cols if col != exclude_col]


def encode_categorical_columns(df, categorical_cols):
    df[categorical_cols] = df[categorical_cols].fillna("Unknown")
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le  # store encoders if needed later
    return df, label_encoders


def drop_na(df):
    return df.dropna()


def filter_two_classes(df, class1_name, class2_name):
    return df[df['Category'].isin([class1_name, class2_name])]


def extract_class_data(df, class_name):
    return df[df['Category'] == class_name].drop(columns=['Category']).values


def calculate_centroid(data):
    return np.mean(data, axis=0)


def calculate_spread(data):
    return np.std(data, axis=0)


def calculate_distance(centroid1, centroid2):
    return np.linalg.norm(centroid1 - centroid2)


def print_results(class1_name, class2_name, centroid1, centroid2, spread1, spread2, distance):
    print(f"Comparing classes: '{class1_name}' and '{class2_name}'\n")
    print("Class 1 centroid (mean vector):", centroid1)
    print("Class 2 centroid (mean vector):", centroid2)
    print("\nClass 1 spread (std dev vector):", spread1)
    print("Class 2 spread (std dev vector):", spread2)
    print("\nDistance between centroids:", distance)


def main():
    drop_columns = [
        'LocationAbbr', 'LocationDesc', 'DataSource', 'Data_Value_Unit',
        'Data_Value_Type', 'Data_Value_Footnote_Symbol', 'Data_Value_Footnote',
        'Numerator', 'LocationID', 'DataValueTypeID', 'GeoLocation',
        'Geographic Level', 'StateAbbreviation'
    ]

    df = load_data("dataset.xlsx", "National_Health_Interview_Surve", drop_columns)
    df = filter_non_null_category(df)
    class1_name, class2_name = get_two_classes(df)
    categorical_cols = get_categorical_columns(df)
    df, label_encoders = encode_categorical_columns(df, categorical_cols)
    df = drop_na(df)
    df_two_classes = filter_two_classes(df, class1_name, class2_name)

    class1_data = extract_class_data(df_two_classes, class1_name)
    class2_data = extract_class_data(df_two_classes, class2_name)

    centroid1 = calculate_centroid(class1_data)
    centroid2 = calculate_centroid(class2_data)

    spread1 = calculate_spread(class1_data)
    spread2 = calculate_spread(class2_data)

    distance_between_centroids = calculate_distance(centroid1, centroid2)

    print_results(class1_name, class2_name, centroid1, centroid2, spread1, spread2, distance_between_centroids)


if __name__ == "__main__":
    main()

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder


def preprocess_data(df):
    """
    Preprocess the user profiling dataset:
    - Handle missing values
    - Encode categorical features
    - Scale numerical features
    """

    # Identify feature types
    categorical_features = df.select_dtypes(include=["object"]).columns.tolist()
    numerical_features = df.select_dtypes(exclude=["object"]).columns.tolist()

    # Numerical pipeline
    num_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # Categorical pipeline
    cat_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    # Combine pipelines
    preprocessor = ColumnTransformer(transformers=[
        ("num", num_pipeline, numerical_features),
        ("cat", cat_pipeline, categorical_features)
    ])

    # Fit & transform
    X_processed = preprocessor.fit_transform(df)

    return X_processed, preprocessor

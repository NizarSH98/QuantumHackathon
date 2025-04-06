import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

def clean_and_prepare_data(df):
    """
    Cleans, encodes, and splits the dataset for ML training.
    - Converts European-style decimals (`,` to `.`) for numeric columns.
    - One-hot encodes categorical features.
    - Returns train/test split.
    """
    try:
        # Column groups
        numeric_cols = [col for col in df.columns if col.startswith(('X', 'P'))]
        clinical_cols = ['Sex', 'Site', 'Tcategory', 'Ncategory', 'Stage', 'Histology']
        target_col = 'Status'

        # Ensure numeric formatting is correct (convert commas to dots)
        for col in numeric_cols:
            df[col] = df[col].astype(str).str.replace(',', '.', regex=False)
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Drop rows with missing or invalid entries
        df = df.dropna(subset=numeric_cols + clinical_cols + [target_col])

        # Encode binary target
        df[target_col] = df[target_col].apply(lambda x: 1 if 'Pathologic' in str(x) else 0)

        # Extract features and labels
        X = df[numeric_cols + clinical_cols]
        y = df[target_col]

        # Define column transformer (scale numeric, one-hot categorical)
        preprocessor = ColumnTransformer([
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), clinical_cols)
        ])

        X_processed = preprocessor.fit_transform(X)
        return train_test_split(X_processed, y, test_size=0.2, random_state=42)

    except Exception as e:
        raise ValueError(f"Data cleaning failed: {e}")

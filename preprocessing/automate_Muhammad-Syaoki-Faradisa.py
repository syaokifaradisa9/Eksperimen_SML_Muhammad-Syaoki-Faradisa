import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.base import BaseEstimator, TransformerMixin
from imblearn.over_sampling import SMOTENC
import numpy as np

class IQR_OutlierRemover(BaseEstimator, TransformerMixin):
    def __init__(self, factor=1.5):
        self.factor = factor
        self.lower_bounds_ = {}
        self.upper_bounds_ = {}
        self.numerical_features_ = None

    def fit(self, X, y=None):
        self.numerical_features_ = X.select_dtypes(include=np.number).columns.tolist()

        for col in self.numerical_features_:
            Q1 = X[col].quantile(0.25)
            Q3 = X[col].quantile(0.75)
            IQR = Q3 - Q1
            self.lower_bounds_[col] = Q1 - self.factor * IQR
            self.upper_bounds_[col] = Q3 + self.factor * IQR
        return self

    def transform(self, X):
        df_transformed = X.copy()
        if not isinstance(df_transformed, pd.DataFrame):
            pass

        for col in self.numerical_features_:
            if col in df_transformed.columns:
                df_transformed[col] = df_transformed[col].clip(
                    lower=self.lower_bounds_[col], upper=self.upper_bounds_[col]
                )
        return df_transformed

def preprocess_data(df, target_column, project_path, random_state=42, test_size=0.2, smotenc_categorical_features_indices=None):
    # Memisahkan data menjadi feature dan target
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    # Menentukan fitur numerik dan kategori
    numerical_features = ['time', 'age', 'wtkg', 'karnof', 'preanti', 'cd40', 'cd420', 'cd80', 'cd820']
    categorical_features = ['trt', 'hemo', 'homo', 'drugs', 'oprior', 'z30', 'race', 'gender', 'str2', 'strat', 'symptom', 'treat', 'offtrt', 'infected']

    # Split data menjadi train dan test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # SMOTENC pada data training
    smote = SMOTENC(random_state=random_state,categorical_features=smotenc_categorical_features_indices)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    # Membuat pipeline untuk data numerik
    numerical_transformer = Pipeline(steps=[
        ('outlier_remover', IQR_OutlierRemover(factor=1.5)),
        ('scaler', MinMaxScaler())
    ])

    # Membuat preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
        ],
        remainder='passthrough'
    )

    # --- Save Preprocessor Pipeline ---
    os.makedirs(project_path, exist_ok=True)
    pipeline_save_path = f"{project_path}/preprocessor.joblib"
    dump(preprocessor, pipeline_save_path)

    # Terapkan pipeline preprocessing pada train dan test
    X_train_processed = preprocessor.fit_transform(X_train_res)
    X_test_processed = preprocessor.transform(X_test)

    # Persiapan export dataset
    processed_numerical_cols = [f'scaled_{col}' for col in numerical_features]
    remaining_cols = [col for col in X_train_res.columns if col not in numerical_features]
    processed_columns = processed_numerical_cols + remaining_cols

    X_train_processed = pd.DataFrame(X_train_processed, columns=processed_columns, index=X_train_res.index)
    X_test_processed = pd.DataFrame(X_test_processed, columns=processed_columns, index=X_test.index)

    train_df_final = X_train_processed.copy()
    train_df_final[target_column] = y_train_res

    test_df_final = X_test_processed.copy()
    test_df_final[target_column] = y_test

    train_file_path = f"{project_path}/aids_preprocessing/train.csv"
    test_file_path = f"{project_path}/aids_preprocessing/test.csv"

    train_df_final.to_csv(train_file_path, index=False)
    test_df_final.to_csv(test_file_path, index=False)

    return X_train_processed, X_test_processed, y_train_res, y_test

# Program Utama
if __name__ == "__main__":
    DATA_INPUT_PATH = '../AIDS_Classification.csv'
    df = pd.read_csv(DATA_INPUT_PATH)
    TARGET_COLUMN = 'infected'
    OUTPUT_FOLDER_PATH = 'aids_preprocessing'

    X_temp = df.drop(TARGET_COLUMN, axis=1)
    smotenc_cat_indices = [1, 4, 5, 6, 8, 9, 11, 12, 13, 14, 15, 16, 17]

    X_train_processed, X_test_processed, y_train_res, y_test = preprocess_data(
        df=df,
        target_column=TARGET_COLUMN,
        project_path=OUTPUT_FOLDER_PATH,
        smotenc_categorical_features_indices=smotenc_cat_indices
    )

    print("\n--- Preprocessing Complete ---")
    print(f"Shape of X_train_processed  : {X_train_processed.shape}")
    print(f"Shape of X_test_processed   : {X_test_processed.shape}")
    print(f"Value counts of y_train_res :\n{y_train_res.value_counts()}")
    print(f"Value counts of y_test      :\n{y_test.value_counts()}")
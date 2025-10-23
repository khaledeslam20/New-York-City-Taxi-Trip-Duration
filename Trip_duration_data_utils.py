from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from Feature_engineering import *
import joblib
from pathlib import Path
import json
def load_data (train_path, val_path=None,val_size=0.2,target_col="log_trip_duration",
               feature_engineering=True) :
    train_df = pd.read_csv(train_path)
    if val_path is not None:
        val_df = pd.read_csv(val_path)
        if feature_engineering :
            train_df = apply_feature_engineering(train_df, is_training=True)
            val_df = apply_feature_engineering(val_df,is_training=False)


            train_df = train_df.dropna(subset=[target_col])
            val_df = val_df.dropna(subset=[target_col])

            train_df, val_df = train_df.align(val_df, join='left', axis=1, fill_value=0)
    else :
        if feature_engineering :
            train_df = apply_feature_engineering(train_df, is_training=True)
            train_df = train_df.dropna(subset=[target_col])

        train_df,val_df = train_test_split(train_df,test_size=val_size,shuffle=True, random_state=42)

    x_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]
    x_val = val_df.drop(columns=[target_col])
    y_val = val_df[target_col]

    return x_train, y_train, x_val, y_val

def load_test_data(test_path, target_col='log_trip_duration', feature_engineering=True):
    test_df = pd.read_csv(test_path)
    if feature_engineering:
        test_df = apply_feature_engineering(test_df, is_training=False)
    if target_col in test_df.columns:
        x_test = test_df.drop(columns=[target_col])
        y_test = test_df[target_col]
        return x_test, y_test
    return test_df, None

def get_scaler( scaler="minmax"):
    if scaler.lower() == "standard":
        return StandardScaler()
    elif scaler.lower() == "minmax":
        return MinMaxScaler()
    else:
        raise ValueError(f"Unknown scaler: {scaler}. Use 'minmax' or 'standard'")

def save_model_with_columns(model, scaler, feature_columns, model_path, scaler_path, columns_path):
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    Path(scaler_path).parent.mkdir(parents=True, exist_ok=True)
    Path(columns_path).parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)

    # Save feature columns for safe prediction
    Path(columns_path).write_text(json.dumps(list(feature_columns)))

    print(f"Model saved to: {model_path}")
    print(f"Scaler saved to: {scaler_path}")
    print(f"Feature columns saved to: {columns_path}")

def load_model(model_path, scaler_path, columns_path):
    if not Path(model_path).exists() or not Path(scaler_path).exists():
        raise FileNotFoundError(f"Model file {model_path} or scaler file {scaler_path} not found")
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    feature_columns = json.loads(Path(columns_path).read_text())

    print(f"Model loaded from: {model_path}")
    print(f"Scaler loaded from: {scaler_path}")
    return model, scaler, feature_columns
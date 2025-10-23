import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, PolynomialFeatures
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

def apply_polynomial_features (degree, x_train=None, x_val=None, features_to_expand=None) :
    """
        Apply polynomial features to selected numerical features

        Args:
            degree: Polynomial degree (1 = no expansion, 2+ = polynomial features)
            x_train: Training features DataFrame
            x_val: Validation features DataFrame (optional)
            features_to_expand: List of features to apply polynomial expansion to
                               If None, uses default important features

        Returns:
            tuple: (x_train_poly, x_val_poly) or x_train_poly if no x_val
    """
    if degree < 1 or degree > 7:
        raise ValueError("Polynomial degree should be between 1 and 3")

    if features_to_expand is None:
        features_to_expand = [
            'haversine_distance', 'start_hour'
        ]

    if degree == 1:
        if x_val is not None:
            return x_train, x_val
        else:
            return x_train

    important_features = [f for f in features_to_expand if f in x_train.columns]

    if not important_features:
        print("Warning: No specified features found for polynomial expansion")
        if x_val is not None:
            return x_train, x_val
        else:
            return x_train

    print(f"Applying polynomial features (degree {degree}) to: {important_features}")
    # Create polynomial transformer
    poly = PolynomialFeatures(degree=degree, include_bias=False)

    # Only apply polynomial expansion to selected features
    x_train_poly_part = poly.fit_transform(x_train[important_features])

    # Get the new feature names
    poly_feature_names = poly.get_feature_names_out(important_features)

    # Convert to DataFrames to align columns properly
    df_train_poly = pd.DataFrame(x_train_poly_part, columns=poly_feature_names, index=x_train.index)

    # Drop original selected features from the base data
    x_train_remaining = x_train.drop(columns=important_features)

    # Combine the polynomial features with the remaining features
    x_train_poly = pd.concat([x_train_remaining, df_train_poly], axis=1)


    if  x_val is not None:
        x_val_poly_part = poly.transform(x_val[important_features])
        df_val_poly = pd.DataFrame(x_val_poly_part, columns=poly_feature_names, index=x_val.index)
        x_val_remaining = x_val.drop(columns=important_features)
        x_val_poly = pd.concat([x_val_remaining, df_val_poly], axis=1)

        return x_train_poly, x_val_poly
    else :
        return x_train_poly


def save_model_with_columns(model, scaler, feature_columns, model_path, scaler_path, columns_path):
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    Path(scaler_path).parent.mkdir(parents=True, exist_ok=True)
    Path(columns_path).parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)

    # Save feature columns for safe prediction
    import json
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
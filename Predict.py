import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from Trip_duration_data_utils import *
from sklearn.metrics import mean_squared_error, r2_score

def main () :
    parser = argparse.ArgumentParser(description="Predict or evaluate with a saved model")
    parser.add_argument('--data_path', type=str, required=True, help='CSV with unseen or test data')
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--scaler_path', type=str, required=True)
    parser.add_argument('--columns_path', type=str, required=True)

    parser.add_argument('--target_col', type=str, default='log_trip_duration', help="Target column name")

    parser.add_argument('--no_feature_engineering', action='store_true')
    parser.add_argument('--output_path', type=str, default=None, help='Optional CSV to save predictions')
    args = parser.parse_args()

    model, scaler, feature_cols = load_model(args.model_path, args.scaler_path, args.columns_path)
    x, y = load_test_data(args.data_path, args.target_col, feature_engineering=not args.no_feature_engineering)

    if feature_cols is not None:
        x = x.reindex(columns=feature_cols, fill_value=0)
        # print(f"Aligned test data to {x.shape[1]} features")

    x_scaled = scaler.transform(x)
    y_pred_log = model.predict(x_scaled)
    y_pred = np.expm1(y_pred_log)

    if y is not None:
        mse = mean_squared_error(y, y_pred_log)
        r2 = r2_score(y, y_pred_log)
        print(f"MSE (log target): {mse:.4f}  R²: {r2:.4f}")

    if args.output_path:
        out = pd.DataFrame({'predicted_trip_duration': y_pred})
        Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(args.output_path, index=False)
        print(f"Predictions saved to: {args.output_path}")

if __name__ == "__main__":
    main()




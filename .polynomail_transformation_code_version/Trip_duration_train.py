import argparse
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score

from Trip_duration_data_utils import *
from pathlib import Path


def evaluate_model(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f"MSE: {mse:.4f}")
    print(f"R²: {r2:.4f}")


def main() :
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path',type = str, default = r'D:\programming\ML\modeling and data concept\project 1\full data\train.csv')
    parser.add_argument('--val_path',type = str, default =  r'D:\programming\ML\modeling and data concept\project 1\full data\val.csv')
    parser.add_argument('--test_path', type=str, help='Path to test CSV file (optional)')
    parser.add_argument('--alpha', type=float, default=1.0, help='Ridge regression alpha parameter')
    parser.add_argument('--scaler', type=str, default='minmax', choices=['minmax', 'standard'], help='Scaler type')
    parser.add_argument('--target_col', type=str, default='log_trip_duration', help="Target column name")
    parser.add_argument('--polynomial_feature_degree',type=int,default=2, help='choose the degree of polynomial features it must be greater than 1')
    parser.add_argument('--poly_features', type=str, nargs='+',
                        default=['haversine_distance'],
                        help='Features to apply polynomial expansion to')

    parser.add_argument('--val_size', type=float, default=0.2, help='Validation split size if no val_path')
    parser.add_argument('--model_dir', type=str, default='models', help='Directory to save model and scaler')
    # parser.add_argument('--run_test', action='store_true', help='Evaluate on test set')
    # parser.add_argument('--save_model', action='store_true', help='Save model and scaler')

    args = parser.parse_args()

    # Validate file paths
    if not Path(args.train_path).exists():
        raise FileNotFoundError(f"Training file not found: {args.train_path}")
    if args.val_path and not Path(args.val_path).exists():
        raise FileNotFoundError(f"Validation file not found: {args.val_path}")


    x_train, y_train,x_val,y_val = load_data(args.train_path,args.val_path,args.val_size,args.target_col)
    print(f"Training set shape: {x_train.shape}")
    print(f"Validation set shape: {x_val.shape}")
    x_train_poly, x_val_poly = apply_polynomial_features(args.polynomial_feature_degree, x_train,
                                                         x_val,args.poly_features)

    scaler = get_scaler(args.scaler)
    x_train_scaled = scaler.fit_transform(x_train_poly)
    x_val_scaled = scaler.transform(x_val_poly)

    ridge = Ridge(alpha=1.0)
    ridge.fit(x_train_scaled,y_train)

    cv_scores = cross_val_score(ridge, x_train_scaled, y_train, cv=5, scoring='r2')
    print("\nCross-Validation Results:")
    print(f"R² Scores: {cv_scores}")
    print(f"Mean R²: {cv_scores.mean():.4f}")
    print(f"Variance of R²: {cv_scores.var():.6f}")

    y_train_pred = ridge.predict(x_train_scaled)
    print("Training Results:")
    evaluate_model(y_train,y_train_pred)

    y_val_pred = ridge.predict(x_val_scaled)
    print("\nValidation Results:")
    evaluate_model(y_val,y_val_pred)


    # un comment to save the model
    # model_path = Path(args.model_dir) / 'ridge_model.joblib'
    # scaler_path = Path(args.model_dir) / 'scaler.joblib'
    # column_path = Path(args.model_dir) / 'feature_columns.json'
    # save_model_with_columns(ridge, scaler, x_train_poly.columns, model_path, scaler_path,column_path)
    # print(f"\nModel and scaler automatically saved to: {args.model_dir}")

if __name__ == "__main__":
    main()

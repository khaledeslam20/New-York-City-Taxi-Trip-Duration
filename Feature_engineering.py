# Baseline 4

import pandas as pd
import numpy as np
def apply_feature_engineering(df, is_training=True) :
  df=df.copy()

  # ================================================================
  # 1. Basic time features
  # ================================================================

  df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'], errors='coerce')

  df['start_hour'] = df['pickup_datetime'].dt.hour
  df['day_of_week'] = df['pickup_datetime'].dt.dayofweek
  df['weekend_day']=df['day_of_week'].isin([5,6]).astype(int)
  df['month']=df['pickup_datetime'].dt.month


  def get_time(hour):
      if 6 <= hour < 10:
          return 'morning_rush'
      elif 10 <= hour < 16:
          return 'midday'
      elif 16 <= hour < 20:
          return 'evening_rush'
      elif 20 <= hour < 24:
          return 'evening'
      else:
          return 'night'

  df['which_part_of_day'] = df['start_hour'].apply(get_time)

  df = pd.get_dummies(df, columns=['which_part_of_day'], drop_first=True)

  # ================================================================
  # 2. Distance calculations
  # ================================================================


  def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate Haversine distance (in km) between two sets of GPS coordinates.
    """
    R = 6371  # Earth's radius in km

    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)

    a = np.sin(delta_phi / 2) ** 2 + \
        np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2) ** 2

    c = 2 * np.arcsin(np.sqrt(a))

    return R * c

  df['haversine_distance'] = haversine_distance(
        df['pickup_latitude'],
        df['pickup_longitude'],
        df['dropoff_latitude'],
        df['dropoff_longitude']
    )

  # Manhattan distance approximation
  # df['manhattan_distance'] = (
  #     abs(df['pickup_latitude'] - df['dropoff_latitude']) +
  #     abs(df['pickup_longitude'] - df['dropoff_longitude'])
  # ) * 111  # Convert to km approximately
  df['log_distance'] = np.log1p(df['haversine_distance'])

  # ================================================================
  # 7. Trip duration transformation (training only)
  # ================================================================

  if 'trip_duration' in df.columns:
      if is_training:
          # df['trip_duration'] = df['trip_duration'].clip(lower=150, upper=22000)
          df['log_trip_duration'] = np.log1p(df['trip_duration'])

      else:
          df['log_trip_duration'] = np.log1p(df['trip_duration'])

  # ================================================================
  # 9. Drop unused columns
  # ================================================================

  cols_to_drop = ['id', 'vendor_id', 'pickup_datetime', 'passenger_count',
                    'pickup_longitude', 'pickup_latitude', 'dropoff_longitude',
                    'dropoff_latitude', 'store_and_fwd_flag', 'trip_duration']

  df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])

  print(df.columns)

  return df



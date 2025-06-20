import pickle
import numpy as np
import pandas as pd
import argparse
import os


with open('model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)

categorical = ['PULocationID', 'DOLocationID']

def read_data(year, month):
    filename = f"https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet"
    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    #print(df.shape)
    return df


def predict_mean_duration(year, month):
    df = read_data(year, month)
    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = model.predict(X_val)
    
    # Derive and log mean and standard deviation
    mean_pred = np.mean(y_pred)
    print(f"Predicted mean duration for {year:04d}-{month:02d} is : {mean_pred:.2f} minutes")
    print("-----------------------------------------------------------------------------")
    save_predictions(year, month, df, y_pred)


def save_predictions(year, month, df, y_pred):
    # 1. Define the output directory
    output_dir = "output"
    
    # 2. Create the directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")
    
    # 3. Set the output file path
    output_file = os.path.join(output_dir, f"yellow_tripdata_{year:04d}-{month:02d}_predictions.parquet")

    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')
    df_result = pd.DataFrame({'ride_id': df['ride_id'], 'predicted_duration': y_pred}) #Create results DataFrame

    #Save dataframe as parquet:
    df_result.to_parquet(
        output_file,
        engine='pyarrow',
        compression=None,
        index=False
    )
    # verify the data in the saved file
    print(f"successfully saved the file :  {output_file}")
    df_check = pd.read_parquet(output_file, engine='pyarrow')
    print(f" Shape of data in the file {df_check.shape}\n")         # Should match (3316216, 2)
    print(f" Null values in the dataframe {df_check.isnull().sum()}") # Check for unexpected nulls


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run the script to predict yellow taxi trip durations.")
    parser.add_argument(
        "--year",
        type=int,
        required=True,
        help="Four-digit year, e.g., 2023"
    )
    parser.add_argument(
        "--month",
        type=int,
        required=True,
        choices=range(1, 13),
        help="Month as a number (1-12)"
    )
    args = parser.parse_args()

    # Call predict_mean_duration function
    predict_mean_duration(args.year, args.month)

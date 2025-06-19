import os
import requests
from urllib.parse import urlparse
import argparse
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType
from prefect import flow, task

# -----------------------------------------------------------------------------
# Data extraction
# -----------------------------------------------------------------------------
@task(retries=3, retry_delay_seconds=5)
def download_data(url: str, output_dir: str = "data"):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract filename from the URL
    parsed_url = urlparse(url)
    filename = os.path.basename(parsed_url.path)
    
    # Set full output path
    target_path = os.path.join(output_dir, filename)
    
    # Download the file
    response = requests.get(url, stream=True)
    response.raise_for_status()

    with open(target_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    print(f"Data Downloaded to {target_path}")

    return target_path
 

# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------
@task(retries=5, retry_delay_seconds=3)
def read_dataframe(filename):
    df = pd.read_parquet(filename, columns=[
        "tpep_pickup_datetime",
        "tpep_dropoff_datetime",
        "PULocationID",
        "DOLocationID",
    ]) # lesser columns read necessary for evading out-of-memory error
    print(f"We initially loaded {df.shape[0]} number of records")

    return df


# -----------------------------------------------------------------------------
# Feature engineering and Data Transformation
# -
@task(retries=3, retry_delay_seconds=2)
def transform_data(df):
    # Compute trip duration in minutes
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df.duration = df.duration.dt.total_seconds() / 60

    # Filter out outliers
    df = df[(df.duration >= 1) & (df.duration <= 60)]

    print(f"After processing(transformation) we are left with {df.shape[0]} number of records")

    return df

 
    
# -----------------------------------------------------------------------------
#  Data preprocessing
# -----------------------------------------------------------------------------
@task
def preparing_data_for_ML(df):
      
    # Prepare features and target
    categorical = ['PULocationID', 'DOLocationID']
    # Cast categorical columns to string type for vectorization
    df[categorical] = df[categorical].astype(str)
    
    # initialize DictVectorizer
    dv = DictVectorizer()

    # setting the target vector
    y_train = df["duration"].values

    
    train_dicts = df[categorical].to_dict(orient="records")

    del df  # drop reference to the full DataFrame for evading OOM error: free up memory before vectorizing
    X_train = dv.fit_transform(train_dicts)
    del train_dicts  # free up that list too for evading OOM error
   
    return X_train, y_train


# -----------------------------------------------------------------------------
# Model training
# -
@task(log_prints=True)
def train_model(X_train, y_train):
    
    # Train and log the linear regression model
    with mlflow.start_run():
      try:
        model = LinearRegression()
        model.fit(X_train, y_train)
        print("Model trained  successfully.")
      except Exception as e:
        print(f"Training failed with error: {e}")
        raise 

    return model

# -----------------------------------------------------------------------------
# Model registration
# -----------------------------------------------------------------------------
@task(log_prints=True)
def register_model(experiment_name: str):
    client = MlflowClient()

    # Fetch the experiment where the run was logged
    experiment = client.get_experiment_by_name(experiment_name)
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        run_view_type=ViewType.ACTIVE_ONLY
    )

    if not runs:
        print("No runs found to register.")
        return

    # Use the most recent run to register its model
    latest_run = runs[0]
    model_uri = f"runs:/{latest_run.info.run_id}/model"
    try:
        mlflow.register_model(
            model_uri=model_uri,
            name="mlops-zoomcamp-2025-assignment3-orchestration-model"
        )
        print(f"Model registered successfully: {model_uri}")
    except Exception as e:
        # Handle registration errors (e.g., name conflicts, network issues)
        print(f"Failed to register model: {e}")


@flow
def main_flow(url: str):
    
    # Experiment configuration
    EXPERIMENT_NAME = "mlops-zoomcamp-2025-assignment3-orchestrated"
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment(EXPERIMENT_NAME)
    # Enable automatic logging of sklearn parameters, metrics, and model
    mlflow.sklearn.autolog()

    # Download data
    filename = download_data(url)

    # Load
    df = read_dataframe(filename)

    # Transform
    df = transform_data(df)

    # Feature Engineering and Preprocess
    X_train, y_train = preparing_data_for_ML(df)

    # Train
    model = train_model(X_train, y_train)
    print(f"The model intercept is {round(model.intercept_, 2)}")

    # Register the trained model
    register_model(EXPERIMENT_NAME)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run the ML pipeline with optional URL input.")
    parser.add_argument(
        "--url",
        type=str,
        default="https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-03.parquet",
        help="URL to the parquet file (default: March 2023 yellow taxi trip data)"
    )
    
    args = parser.parse_args()
    main_flow(url=args.url)

import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType

# Experiment configuration
EXPERIMENT_NAME = "mlops-zoomcamp-2025-assignment-3"
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment(EXPERIMENT_NAME)
# Enable automatic logging of sklearn parameters, metrics, and model
mlflow.sklearn.autolog()

# -----------------------------------------------------------------------------
# Data loading and preprocessing
# -----------------------------------------------------------------------------
def read_dataframe(filename):
    df = pd.read_parquet(filename, columns=[
        "tpep_pickup_datetime",
        "tpep_dropoff_datetime",
        "PULocationID",
        "DOLocationID",
    ]) # lesser columns read necessary for evading out-of-memory error
    print(f"We initially loaded {df.shape[0]} number of records")

    # Compute trip duration in minutes
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df.duration = df.duration.dt.total_seconds() / 60

    # Filter out outliers
    df = df[(df.duration >= 1) & (df.duration <= 60)]

    # Cast categorical columns to string type for vectorization
    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)
    
    print(f"After processing(transformation) we are left with {df.shape[0]} number of records")
    return df

# -----------------------------------------------------------------------------
# Feature engineering and model training
# -----------------------------------------------------------------------------
def preprocess_and_train_model(df):
    # Prepare features and target
    categorical = ['PULocationID', 'DOLocationID']
    dv = DictVectorizer()

    y_train = df["duration"].values
    train_dicts = df[categorical].to_dict(orient="records")

    del df  # drop reference to the full DataFrame for evading OOM error: free up memory before vectorizing
    X_train = dv.fit_transform(train_dicts)
    del train_dicts  # free up that list too for evading OOM error
   

    # Train and log the linear regression model
    with mlflow.start_run():
        model = LinearRegression()
        model.fit(X_train, y_train)

    return dv, model

# -----------------------------------------------------------------------------
# Model registration
# -----------------------------------------------------------------------------
def register_model():
    client = MlflowClient()

    # Fetch the experiment where the run was logged
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
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
            name="mlops-zoomcamp-2025-assignment3-model"
        )
        print(f"Model registered successfully: {model_uri}")
    except Exception as e:
        # Handle registration errors (e.g., name conflicts, network issues)
        print(f"Failed to register model: {e}")

# -----------------------------------------------------------------------------
# Main pipeline execution (without Prefect)
# -----------------------------------------------------------------------------
def run_ML_pipeline():
    # Path to parquet file containing NYC taxi trip data
    filename = "./data/yellow_tripdata_2023-03.parquet"

    # Load and preprocess data
    df = read_dataframe(filename)

    # Train model
    dv, model = preprocess_and_train_model(df)
    print(f"The model intercept is {round(model.intercept_, 2)}")

    # Register the trained model
    register_model()


if __name__ == '__main__':
    run_ML_pipeline()

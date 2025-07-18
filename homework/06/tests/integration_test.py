from datetime import datetime
import pandas as pd
import s3fs
import batch
import localstack_operations_utilities
import read_from_localstack

year = 2023
month = 1

def dt(hour, minute, second=0):
    return datetime(year, month, 1, hour, minute, second)

def create_input_dataframe():
   
   data = [
        (None, None, dt(1, 1), dt(1, 10)),
        (1, 1, dt(1, 2), dt(1, 10)),
        (1, None, dt(1, 2, 0), dt(1, 2, 59)),
        (3, 4, dt(1, 2, 0), dt(2, 2, 1)),      
   ]
   columns = ['PULocationID', 'DOLocationID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime']
   df = pd.DataFrame(data, columns=columns)   

   return df


def test_save_dataframe():
    # Create the DataFrame and save it to S3
    df = create_input_dataframe()
    input_file = localstack_operations_utilities.get_input_file(year, month)
    df.to_parquet(input_file, engine='pyarrow', index=False, compression=None, storage_options=localstack_operations_utilities.get_storage_options())   

    fs = s3fs.S3FileSystem(
        key="foo123",
        secret="bar456",
        client_kwargs={
            "endpoint_url": "http://localhost:4566"
        }
    )       
    expected_file_size = 3215 # Expected file size in bytes
    # Check if the file exists and has the expected size
    assert fs.exists(input_file), f"Input file {input_file} does not exist"
    assert fs.info(input_file)['size'] == expected_file_size, f"Actual Fize size  {fs.info(input_file)['size']} does not match expected size {expected_file_size}"

def test_read_dataframe():
    input_file = localstack_operations_utilities.get_input_file(year, month)
    expected_df = create_input_dataframe()
    actual_df = read_from_localstack.read_data(input_file)
    # Compare the DataFrame read from S3 with the expected DataFrame
    pd.testing.assert_frame_equal(actual_df, expected_df)  


def test_save_read_integration():
    df = create_input_dataframe()
    output_file = localstack_operations_utilities.get_output_path(year, month)
    categorical = ['PULocationID', 'DOLocationID']
    batch.save_data(df, output_file, categorical)

    expected_value = 36.28  # Expected sum of predicted durations
    df_read = read_from_localstack.read_data(output_file)
    assert round(df_read['predicted_duration'].sum(), 2) == expected_value, f"Actual Sum of Predicted duration {round(df_read['predicted_duration'].sum(), 2)}does not match expected value {expected_value}"

  
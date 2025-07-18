import sys
import os
import pickle
import pandas as pd

def read_and_process_data(filename, categorical):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df
    

def get_input_path(year, month):
    default_input_pattern = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet'
    INPUT_FILE_PATTERN = ''
    input_pattern = os.getenv(INPUT_FILE_PATTERN, default_input_pattern)
    return input_pattern.format(year=year, month=month)


def get_input_file(year, month):
    input_pattern = os.getenv('INPUT_FILE_PATTERN')
    return input_pattern.format(year=year, month=month)


def get_output_path(year, month):
    default_output_pattern = f's3://nyc-duration-prediction-alexey/taxi_type=fhv/year={year:04d}/month={month:02d}/predictions.parquet'
    output_pattern = os.getenv('OUTPUT_FILE_PATTERN', default_output_pattern)
    return output_pattern.format(year=year, month=month)

def get_storage_options():
    S3_ENDPOINT_URL = os.getenv('S3_ENDPOINT_URL', 'http://localhost:4566')
    options = {
        'client_kwargs': {
            'endpoint_url': S3_ENDPOINT_URL
        }
    }
    return options



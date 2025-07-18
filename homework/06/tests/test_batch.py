from datetime import datetime
import pandas as pd
import batch


def dt(hour, minute, second=0):
    return datetime(2023, 1, 1, hour, minute, second)

def test_prepare_data():
   
   data = [
        (None, None, dt(1, 1), dt(1, 10)),
        (1, 1, dt(1, 2), dt(1, 10)),
        (1, None, dt(1, 2, 0), dt(1, 2, 59)),
        (3, 4, dt(1, 2, 0), dt(2, 2, 1)),      
   ]
   columns = ['PULocationID', 'DOLocationID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime']
   df = pd.DataFrame(data, columns=columns)   

   categorical = ['PULocationID', 'DOLocationID']
   actual_df_result = batch.prepare_data(df, categorical)
   
   expected_data = [
      ('-1', '-1', dt(1,1), dt(1,10), 9.0),
      ('1', '1', dt(1,2), dt(1,10), 8.0)          
   ]
   expected_columns = ['PULocationID', 'DOLocationID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime', 'duration']
   expected_df_result = pd.DataFrame(expected_data, columns=expected_columns)

   assert actual_df_result.equals(expected_df_result), "DataFrames do not match"
   assert actual_df_result.shape == expected_df_result.shape, "Shapes of dataframes do not match"
   assert actual_df_result['duration'].min() >= 1, "Duration values below 1 minute  found"
   assert actual_df_result['duration'].max() <= 60, "Duration values above 60 minutes found"
   assert actual_df_result[categorical].dropna().shape[0] == actual_df_result.shape[0],  "Dropping missing rows reduce the count, i.e., NaNs exist"

import sys
import os
import pickle
import pandas as pd
import localstack_operations_utilities
import read_from_localstack


def save_original_data(df, input_file, options):
    df.to_parquet(input_file, engine='pyarrow', index=False, compression=None, storage_options=options)
    print(f'Original Data frame saved to {input_file}')


def save_and_write(year, month):
    input_file = localstack_operations_utilities.get_input_path(year, month)
    output_file = localstack_operations_utilities.get_output_path(year, month)

    print (f'Input file: {input_file}')
    print (f'Output file: {output_file}')

    options = localstack_operations_utilities.get_storage_options()
    
    categorical = ['PULocationID', 'DOLocationID']
    df = localstack_operations_utilities.read_and_process_data(input_file, categorical)
    localstack__original_dataframe_path = localstack_operations_utilities.get_input_file(year, month)
    save_original_data(df, localstack__original_dataframe_path, options)

    df_result = get_prediction_dataframe(localstack__original_dataframe_path, year, month, categorical)
    write_predictions_to_localstack(df_result, output_file)
   
    
    
def get_prediction_dataframe(df_path, year, month, categorical):
    df = read_from_localstack.read_data(df_path)

    with open('model.bin', 'rb') as f_in:
      dv, lr = pickle.load(f_in)
    
    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')
    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = lr.predict(X_val)

    print('predicted mean duration:', y_pred.mean())

    df_result = pd.DataFrame()
    df_result['ride_id'] = df['ride_id']
    df_result['predicted_duration'] = y_pred

    return df_result


def write_predictions_to_localstack(df_result, output_file):
    df_result.to_parquet(
        output_file,
        engine="pyarrow",
        index=False,
        compression=None,
        storage_options=localstack_operations_utilities.get_storage_options()
    )
    print(f'Predictions written to {output_file}')


if __name__ == '__main__':
   save_and_write(int(sys.argv[1]), int(sys.argv[2]))


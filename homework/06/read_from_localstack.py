import sys
import os
import pickle
import pandas as pd
import localstack_operations_utilities


def read_data(file_name): 
    
    options = localstack_operations_utilities.get_storage_options()

    df = pd.read_parquet(file_name, storage_options=options)
   
    print(df.head())
    print('Data read successfully')
    print('Number of rows:', len(df))
    print('Columns:', df.columns.tolist())

    return df


if __name__ == '__main__':
   read_data(localstack_operations_utilities.get_output_path(int(sys.argv[1]), int(sys.argv[2])))


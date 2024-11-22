import os
import pandas as pd

def clean_csv(file_path):
    df = pd.read_csv(file_path)
    df.dropna(inplace=True)
    df = df.round(2)
    df.to_csv(file_path, index=False)
    
    # Append cleaned data to 3080Ti_WSL_Compression_Data.csv
    dataset_name = os.path.basename(os.path.dirname(file_path))
    df['Dataset'] = dataset_name
    output_file = '3080Ti_WSL_Compression_Data.csv'
    if not os.path.isfile(output_file):
        df.to_csv(output_file, index=False)
    else:
        df.to_csv(output_file, mode='a', header=False, index=False)

def process_output_directory(output_dir):
    for root, _, files in os.walk(output_dir):
        for file in files:
            if file == 'report_summary_compress.csv':
                file_path = os.path.join(root, file)
                clean_csv(file_path)

if __name__ == "__main__":
    output_directory = 'output'
    process_output_directory(output_directory)

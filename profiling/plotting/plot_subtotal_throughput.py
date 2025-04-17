import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV data
data = pd.read_csv('/home/skyler/sz_compression/cusz-dev/profiling/combined_kernel_data.csv')

# Define datasets
datasets = ['HACC', 'HURR', 'CESM', 'EXAALT']

peak_through = {}

peak_through['A100'] = 1555
peak_through['A4000'] = 448
peak_through['3080Ti'] = 912

# Create a histogram for each file in each dataset
for dataset in datasets:
    dataset_data = data[data['Dataset'].str.contains(dataset)]
    unique_files = dataset_data['File'].unique()
    
    for file in unique_files:
        file_data = dataset_data[dataset_data['File'] == file]
        plt.figure(figsize=(10, 6))
        plt.bar(file_data['Card'], file_data['(subtotal)_throughput'])
        plt.xlabel('Card')
        plt.ylabel('Subtotal Throughput')
        plt.title(f'Subtotal Throughput for {file}')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'/home/skyler/sz_compression/cusz-dev/profiling/plotting/{dataset}/{file}_throughput_histogram.png')
        plt.close()

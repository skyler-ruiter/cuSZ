import pandas as pd

def combine_kernel_data():
    # Load CSV files
    a4000_data = pd.read_csv('/home/skyler/sz_compression/cusz-dev/profiling/A4000_combined_kernel_data.csv')
    a100_data = pd.read_csv('/home/skyler/sz_compression/cusz-dev/profiling/A100_combined_kernel_data.csv')
    ti3080_data = pd.read_csv('/home/skyler/sz_compression/cusz-dev/profiling/3080Ti_combined_kernel_data.csv')

    # Add a new field for the card
    a4000_data['Card'] = 'A4000'
    a100_data['Card'] = 'A100'
    ti3080_data['Card'] = '3080Ti'

    # Combine the data
    combined_data = pd.concat([a4000_data, a100_data, ti3080_data])

    # Save the combined data to a new CSV file
    combined_data.to_csv('/home/skyler/sz_compression/cusz-dev/profiling/combined_kernel_data.csv', index=False)

if __name__ == "__main__":
    combine_kernel_data()

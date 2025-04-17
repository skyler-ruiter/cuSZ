import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV data
data = pd.read_csv('/home/skyler/sz_compression/cusz-dev/profiling/combined_kernel_data.csv')

# Define datasets
#datasets = ['HACC', 'HURR', 'CESM', 'EXAALT']
datasets = ['HACC', 'EXAALT']

# filter out all columns with "throughput"
data = data[data.columns.drop(list(data.filter(regex='throughput')))]

# Create a bar graph for each file in each dataset
for dataset in datasets:
    dataset_data = data[data['Dataset'].str.contains(dataset)]
    unique_files = dataset_data['File'].unique()
    
    for file in unique_files:
        file_data = dataset_data[dataset_data['File'] == file]
        cards = file_data['Card'].unique()
        
        # Calculate percentage time contribution for each kernel per card
        kernel_columns = file_data.columns[2:-2]  # Adjust based on your CSV structure
        file_data = file_data[['Card'] + list(kernel_columns)]
        file_data = file_data.melt(id_vars=['Card'], var_name='Kernel', value_name='Time (ms)')
        file_data = file_data[~file_data['Kernel'].isin(['(subtotal)_time', '(total)_time'])]
        file_data['time_percentage'] = (
            file_data.groupby('Card')['Time (ms)'].transform(lambda x: x / x.sum() * 100)
        )
        
        # Pivot the data for plotting
        plot_data = file_data.pivot(index='Card', columns='Kernel', values='time_percentage')
        
        # Create the stacked bar plot
        plt.figure(figsize=(10, 6))
        plot_data.plot(kind='bar', stacked=True, colormap='viridis')
        
        # Add plot labels and title
        plt.title(f'Kernel Time Breakdown as % of Total Time for {file}', fontsize=14)
        plt.xlabel('Card', fontsize=14)
        plt.ylabel('Percentage of Total Time (%)', fontsize=14)
        plt.legend(title='Kernel')
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(f'/home/skyler/sz_compression/cusz-dev/profiling/plotting/{dataset}/{file}_percentage_time_breakdown.png')



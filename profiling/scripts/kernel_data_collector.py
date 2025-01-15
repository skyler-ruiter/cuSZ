import os
import csv
import numpy as np

# define the datasets and their directories
datasets = ["CESM", "EXAALT", "HACC_M", "HURR"]
base_dir = "../profiling/output"

skipped_data = []

def gatherKernelStatistics(dataset, file):
    data_dir = os.path.join(base_dir, dataset)
    
    # compression (5 replicates)
    comp_times = {}
    
    # decompression (5 replicates)
    decomp_times = {}
    
    # open the five files and add the data to the dictionary
    for i in range(1, 6):
        rep_file_name = f"{file}_rep{i}.txt"
        comp_file = os.path.join(data_dir, rep_file_name)
        decomp_file = os.path.join(data_dir, f"{file}_decomp_rep{i}.txt")
        
        kernel_data = {}
        
        # open the compression file data and add to dictionary
        try:
            with open(comp_file, 'r') as comp_file:
                # skip first 4 lines
                lines = comp_file.readlines()[4:]
                
                # get the kernel name, time, and throughput
                for line in lines:
                    kernel, time, throughput = line.split()
                    kernel_data[kernel] = {
                        'time': float(time),
                        'throughput': float(throughput)
                    }
                    # if (total) is kernel, end of kernel data
                    if kernel == "(total)":
                        break
            comp_times[f"rep{i}"] = kernel_data
        except FileNotFoundError:
            skipped_data.append(comp_file)
    
    return comp_times

######################################################

for data in datasets:
    # Initialize an empty dictionary to hold the combined data
    combined_data = {}
    
    data_dir = os.path.join(base_dir, data) 
    data_dir = data_dir
    
    print(data_dir)
    
    # get list of unqiue file basenames
    files = []
    for root, _, file_list in os.walk(data_dir):
        for file in file_list:
            if file.endswith('.txt'):
                if "_decomp" in file:
                    # print(file)
                    file = file.split("_decomp")[0]
                    files.append(file)
    # get unique files
    files = list(set(files))
    
    # loop through each file
    for file in files:
      # print(file)
      temp = gatherKernelStatistics(data, file)
      combined_data[file] = temp
      
    averaged_data = {}
    
    # average the data
    for file, rep_data in combined_data.items():
        averaged_data[file] = {}
        replicate_count = len(rep_data)
        
        for rep, kernels in rep_data.items():
            for kernel, metrics in kernels.items():
                if kernel not in averaged_data[file]:
                    averaged_data[file][kernel] = {'time': 0, 'throughput': 0}
                averaged_data[file][kernel]['time'] += metrics['time']
                averaged_data[file][kernel]['throughput'] += metrics['throughput']
                
        for kernel, metrics in averaged_data[file].items():
            metrics['time'] /= replicate_count
            metrics['throughput'] /= replicate_count
    
    # write the averaged data to a csv file
    with open(f"{data}_averaged_data.csv", 'w') as write_file:
        writer = csv.writer(write_file)
        writer.writerow(["File", "Kernel", "Time", "Throughput"])
        
        for data_file, avg_data in averaged_data.items():
            for kernel, metrics in avg_data.items():
                writer.writerow([data_file, kernel, metrics['time'], metrics['throughput']])
                
    print(f"{data}_averaged_data.csv has been written.")
                                
                                
# for each averaged dataset csv combine each file into a single entry and make headers the kernels
# then write to a new csv file

dataset_csvs = ["CESM_averaged_data.csv", "EXAALT_averaged_data.csv", "HACC_M_averaged_data.csv", "HURR_averaged_data.csv"]

combined_data = {}

# Read and combine data from all dataset CSVs
for dataset_csv in dataset_csvs:
    dataset = dataset_csv.split("_")[0]
    with open(dataset_csv, 'r') as read_file:
        reader = csv.reader(read_file)
        next(reader)
        
        for row in reader:
            file = row[0]
            kernel = row[1]
            time = float(row[2])
            throughput = float(row[3])
            
            if file not in combined_data:
                combined_data[file] = {'dataset': dataset}
                
            combined_data[file][kernel] = {
                'time': time,
                'throughput': throughput
            }

# Write combined data to a new CSV file
if combined_data:
    with open("kernel_data.csv", 'w') as write_file:
        writer = csv.writer(write_file)
        
        # Write headers
        headers = ["File", "Dataset"]
        sample_file = next(iter(combined_data.values()))
        for kernel in sample_file:
            if kernel != 'dataset':
                headers.append(f"{kernel}_time")
                headers.append(f"{kernel}_throughput")
        writer.writerow(headers)
        
        # Write data rows
        for file, data in combined_data.items():
            row = [file, data['dataset']]
            for kernel in sample_file:
                if kernel != 'dataset':
                    row.append(round(data[kernel]['time'], 3))
                    row.append(round(data[kernel]['throughput'], 3))
            writer.writerow(row)

    print("kernel_data.csv has been written.")
else:
    print("No data to write to kernel_data.csv.")
    
# remove the individual dataset csvs
for dataset_csv in dataset_csvs:
    os.remove(dataset_csv)
    print(f"{dataset_csv} has been removed.")

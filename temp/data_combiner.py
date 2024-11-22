import os
import pandas as pd

# Define the datasets and their directories
datasets = [("3080ti", "wsl"), ("A100", "bigred200"), ("A4000", "hipdac"), ("H100", "quartz")]
base_dir = "results"
output_file = "combined_dataset.csv"

# Initialize an empty DataFrame to hold the combined data
combined_df = pd.DataFrame()

# Loop through each dataset and read the [dataset]_[machine]_comp_results.csv file
for dataset, machine in datasets:
    file_path = os.path.join(base_dir, f"{dataset}_{machine}_comp_results.csv")
    df = pd.read_csv(file_path)
    df["card"] = dataset
    df["machine"] = machine
    combined_df = pd.concat([combined_df, df], ignore_index=True)

# Save the combined DataFrame to a CSV file
combined_df.to_csv(output_file, index=False)

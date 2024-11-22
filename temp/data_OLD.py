import os
import sys
import csv

def parse_report(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # remove first 3 lines
    lines = lines[4:]
    
    # get file name
    file_name = os.path.basename(file_path)
    file_name = file_name[:-4]
    
    print(file_name)
    
    report = {}
    for line in lines:
        
        kernel = line.split()[0]
        time_ms = float(line.split()[1])
        gib_s = float(line.split()[2])
        report[kernel] = {
            'time_ms': time_ms,
            'gib_s': gib_s
        }
    return report

def collect_data(output_dir):
    reports = {}
    for root, _, files in os.walk(output_dir):
        for file in files:
            if file.endswith('.txt'):
                file_path = os.path.join(root, file)
                report = parse_report(file_path)
                reports[file] = report
    return reports

def save_to_csv(reports, output_dir):
    csv_file = os.path.join(output_dir, 'report_summary.csv')
    with open(csv_file, 'w', newline='') as csvfile:
        fieldnames = ['file', 'kernel', 'time_ms', 'gib_s']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for file, report in reports.items():
            for kernel, data in report.items():
                writer.writerow({
                    'file': file,
                    'kernel': kernel,
                    'time_ms': data['time_ms'],
                    'gib_s': data['gib_s']
                })

def main():
    if len(sys.argv) != 2:
        print("Usage: python3 data.py <output_dir>")
        sys.exit(1)
    
    output_dir = sys.argv[1]
    reports = collect_data(output_dir)
    save_to_csv(reports, output_dir)

if __name__ == "__main__":
    main()

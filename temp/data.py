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

def average_reports(reports):
    averaged_reports = {}
    for file, report in reports.items():
        base_name = file.rsplit('_rep', 1)[0]
        if base_name not in averaged_reports:
            averaged_reports[base_name] = {}
        
        for kernel, metrics in report.items():
            if kernel not in averaged_reports[base_name]:
                averaged_reports[base_name][kernel] = {'time_ms': 0, 'gib_s': 0, 'count': 0}
            
            averaged_reports[base_name][kernel]['time_ms'] += metrics['time_ms']
            averaged_reports[base_name][kernel]['gib_s'] += metrics['gib_s']
            averaged_reports[base_name][kernel]['count'] += 1
    
    for base_name, kernels in averaged_reports.items():
        for kernel, metrics in kernels.items():
            metrics['time_ms'] /= metrics['count']
            metrics['gib_s'] /= metrics['count']
            del metrics['count']
    
    return averaged_reports

def save_to_csv(reports, output_dir):
    csv_file = os.path.join(output_dir, 'report_summary_compress.csv')
    with open(csv_file, 'w', newline='') as csvfile:
        # Collect all unique kernels
        kernels = set()
        for report in reports.values():
            kernels.update(report.keys())
        kernels = sorted(kernels)
        
        # Create fieldnames dynamically
        fieldnames = ['file'] + [f'{kernel}_time_ms' for kernel in kernels] + [f'{kernel}_gib_s' for kernel in kernels]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for file, report in reports.items():
            row = {'file': file}
            for kernel in kernels:
                if kernel in report:
                    row[f'{kernel}_time_ms'] = report[kernel]['time_ms']
                    row[f'{kernel}_gib_s'] = report[kernel]['gib_s']
                else:
                    row[f'{kernel}_time_ms'] = ''
                    row[f'{kernel}_gib_s'] = ''
            writer.writerow(row)

def main():
    if len(sys.argv) != 2:
        print("Usage: python3 data.py <output_dir>")
        sys.exit(1)
    
    output_dir = sys.argv[1]
    reports = collect_data(output_dir)
    averaged_reports = average_reports(reports)
    save_to_csv(averaged_reports, output_dir)

if __name__ == "__main__":
    main()

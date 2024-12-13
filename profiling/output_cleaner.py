import re
import os

def remove_ansi_escape_sequences(file_path):
    with open(file_path, 'r') as file:
        content = file.read()

    # Regular expression to match ANSI escape sequences
    ansi_escape = re.compile(r'\x1B[@-_][0-?]*[ -/]*[@-~]')
    cleaned_content = ansi_escape.sub('', content)

    # Check if the cleaned content is empty or contains "*** exit on failure"
    if not cleaned_content.strip() or "*** exit on failure" in cleaned_content:
        os.remove(file_path)
        print(f"Removed file: {file_path}")
    else:
        # Overwrite the original file with cleaned content
        with open(file_path, 'w') as file:
            file.write(cleaned_content)
        print(f"Cleaned file: {file_path}")

def process_directory(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.txt'):  # Adjust the file extension if needed
                file_path = os.path.join(root, file)
                remove_ansi_escape_sequences(file_path)

# Example usage
directory = '../profiling/output'  # Replace with your directory path
process_directory(directory)
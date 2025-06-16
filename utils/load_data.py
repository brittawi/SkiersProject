import json
import os

# Function to load JSON data
def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)
    
def load_summary_json(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            return json.load(f)
    return {}
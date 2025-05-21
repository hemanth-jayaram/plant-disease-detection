import os
import zipfile
from pathlib import Path

def create_zip():
    # Define the files and directories to include
    files_to_include = [
        'plant_disease_detector.py',
        'predict.py',
        'requirements.txt',
        'README.md',
        'plant_disease_detector.h5'
    ]
    
    # Create a zip file
    with zipfile.ZipFile('Plant_Disease_Detection.zip', 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Add individual files
        for file in files_to_include:
            if os.path.exists(file):
                zipf.write(file)
        
        # Add the dataset directory
        dataset_dir = 'Plant disease detection Dataset'
        if os.path.exists(dataset_dir):
            for root, _, files in os.walk(dataset_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, start=os.path.dirname(__file__))
                    zipf.write(file_path, arcname)
    
    print("Successfully created Plant_Disease_Detection.zip")

if __name__ == "__main__":
    create_zip()

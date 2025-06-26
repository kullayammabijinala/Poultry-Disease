# create_dataset_folders.py

import os

# List of poultry diseases
diseases = [
    "fowlpox",
    "coccidiosis",
    "newcastle",
    "infectious bronchitis",
    "avian influenza",
    "marek's disease",
    "salmonellosis",
    "aspergillosis"
]

base_path = "poultry_dataset"

# Create dataset directories
for disease in diseases:
    folder_name = disease.lower().replace(" ", "_")
    path = os.path.join(base_path, folder_name)
    os.makedirs(path, exist_ok=True)
    print(f"✅ Created folder: {path}")

import os
import yaml

# Define the base directory structure
base_dir = "/home/pace/AlzheimerClassification/data/Alzheimer/rgb_in_air_on_paper/TASK_08"
folds = ["Fold1", "Fold2", "Fold3", "Fold4", "Fold5"]
categories = ["Test", "Train", "Val"]
subcategories = ["HC", "PT"]

# Initialize the structure for storing split divisions
split_division = {fold: {category: [] for category in categories} for fold in folds}

# Iterate through the folder structure to populate the split division
for fold in folds:
    for category in categories:
        for subcategory in subcategories:
            folder_path = os.path.join(base_dir, fold, category, subcategory)
            if os.path.exists(folder_path):
                # List all files in the folder
                files = os.listdir(folder_path)
                # Add files to the corresponding split division
                split_division[fold][category].extend(files)

# Save the split division to a YAML file
yaml_file_path = "split_division.yaml"
with open(yaml_file_path, "w") as yaml_file:
    yaml.dump(split_division, yaml_file, default_flow_style=False)

print(f"Split division saved to {yaml_file_path}")
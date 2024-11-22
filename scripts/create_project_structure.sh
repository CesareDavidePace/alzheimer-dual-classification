#!/bin/bash

# Create the main project directory and subdirectories
mkdir -p image_classification_project/{data,models,scripts}

# Navigate into the project directory
cd image_classification_project

# Create empty files
touch main.py data_module.py requirements.txt README.md models/custom_model.py

echo "Project structure created successfully."
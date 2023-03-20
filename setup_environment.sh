#!/bin/bash

# Create a new virtual environment
python -m venv AGN_env

# Activate the virtual environment
source AGN_env/Scripts/activate

# Install the requirements from the requirements.txt file
pip install -r requirements.txt

# Print a success message
echo "Virtual environment setup complete!"
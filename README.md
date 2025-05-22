# Shapefinder: Replication Data

This repository contains the replication data for the paper "Accounting for variability in conflict dynamics: A pattern-based predictive model" introducing the **Shape Finder**, a shape-based model designed to predict conflict fatalities.

## Overview
The repository includes the necessary scripts and data to replicate the results presented in the paper. The primary script, `comapre.py`, executes the Shape Finder model and generates output files that visualize and store prediction results.

## Requirements
- **Python version:** 3.8.5
- Required libraries: Install dependencies using:
  ```bash
  pip install -r requirements.txt

## Running the Model
To reproduce the results, execute the following command in your terminal:
```bash
python comapre.py
```
## Expected Runtime
The script should take approximately 5 minutes to complete.

## Directory Structure
- comapre.py: Main script.
- shape.py: Functions needed to run the model. 
- Datasets/: Contains input data required.
- out/: Stores the generated images.
- results/: Contains data produced by the model.

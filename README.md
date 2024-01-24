# MedVisionaries Age Prediction Application 

## Overview

Developed by the students of the Computer Science department at Heinrich Heine university Düsseldorf, the MedVisionaries Age Prediction Application represents a cutting-edge intersection of technology and medical science. This Python-based tool harnesses the power of machine learning models to accurately estimate an individual's age using eye fundus images.
## Team Members

- Mazen Al Hamidi
- Smmon Sillah
- Birol Yildiz
- Jaeyong Shin
- Wilke Saathoff



## Requirements
- NumPy
- Pandas
- Opencv-python
- Scikit-learn

## How to Start the App

1. **Navigate to the Hachathlon Folder**:  
   Using the command line, navigate to the Hachathlon folder. You can do this by running a command like `cd path_to_Hachathlon`, where `path_to_Hachathlon` is the path to your Hachathlon folder.

2. **Install the Conda Environment**:  
   Run the following command to install the Conda environment:
   ```bash
   conda env create -p ./env -f environment.yml
3. **Add the Images folder to the MedVisionaries folder to avoid path conflicts**.
4. **Run the App**
## Application Structure

The application is organized into several Python scripts:

- **main.py**: The entry point of the application.
- **model.py**: Contains the logic for the age prediction model.
- **preprocessing.py**: Includes functions for image preprocessing.


## Key Functions
- `Preprocessor(image_path)`: Processes the image for model input.
- `prepare(image_folder)`: Prepares images in the specified folder.
- `resize_images(image_folder, target_width, target_height)`: Resizes images to a specified size.


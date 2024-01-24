#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Original Template Author: Jan Ruhland, modified by MedVisionaries


import numpy as np
import random as rd
import pandas as pd
import cv2
import os
import random
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, explained_variance_score, \
    median_absolute_error, max_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer  # Corrected import
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, explained_variance_score, \
    median_absolute_error, max_error
from preprocessing import Preprocessor, prepare, process_and_display_images


# Feature extraction function
def extract_features(image_path):
    img = cv2.imread(image_path)
    if img is not None:
        return np.mean(img)  # PCA + Logistic Regression
    else:
        return np.nan  # Return NaN if the image can't be loaded


# Update the image paths and apply the feature extraction
def update_path_and_extract(row, col_name):
    updated_path = 'resized_images/' + row[col_name]  # Update the path
    return extract_features(updated_path)


def read_adjust_csv():
    df = pd.read_csv('annotations.csv')
    print(df.info())

    # Create new lines with modified information
    new_lines = pd.concat(
        [df[['ID', 'Age', 'Sex', 'Right_Fundus', 'N', 'D', 'G', 'C', 'A', 'H', 'M', 'O', 'Right_Diagnosis']],
         df[['ID', 'Age', 'Sex', 'Left_Fundus', 'N', 'D', 'G', 'C', 'A', 'H', 'M', 'O', 'Left_Diagnosis']]])

    # Reset the index of the new lines
    new_lines.reset_index(drop=True, inplace=True)
    # Iterate over each row in the dataframe
    for index, row in new_lines.iterrows():
        # Check if the diagnosis contains the word "normal"
        if not pd.isna(row['Right_Diagnosis']) and 'normal' in row['Right_Diagnosis']:
            # Set N to 1
            new_lines.at[index, 'N'] = 1
            # Set G, C, A, H, M to 0
            new_lines.at[index, 'G'] = 0
            new_lines.at[index, 'C'] = 0
            new_lines.at[index, 'A'] = 0
            new_lines.at[index, 'H'] = 0
            new_lines.at[index, 'M'] = 0
        elif not pd.isna(row['Left_Diagnosis']) and 'normal' in row['Left_Diagnosis']:
            # Set N to 1
            new_lines.at[index, 'N'] = 1
            # Set G, C, A, H, M to 0
            new_lines.at[index, 'G'] = 0
            new_lines.at[index, 'C'] = 0
            new_lines.at[index, 'A'] = 0
            new_lines.at[index, 'H'] = 0
            new_lines.at[index, 'M'] = 0
    new_lines['Fundus'] = new_lines['Right_Fundus'].fillna(new_lines['Left_Fundus'])
    new_lines['Diagnosis'] = new_lines['Right_Diagnosis'].fillna(new_lines['Left_Diagnosis'])
    return new_lines


def print_evaluation_metrics(y_pred, y_test, y):
    # Evaluate the model for each output variable
    for i, target_variable in enumerate(y.columns):  # Corrected variable name
        mae = mean_absolute_error(y_test[target_variable], y_pred[:, i])
        mse = mean_squared_error(y_test[target_variable], y_pred[:, i])
        r2 = r2_score(y_test[target_variable], y_pred[:, i])
        explained_variance = explained_variance_score(y_test[target_variable], y_pred[:, i])
        medae = median_absolute_error(y_test[target_variable], y_pred[:, i])
        max_err = max_error(y_test[target_variable], y_pred[:, i])

        print(f'Evaluation metrics for {target_variable}:')
        print(f'Mean Absolute Error: {mae}')
        print(f'Mean Squared Error: {mse}')
        # print(f'R2 Score: {r2}')
        # print(f'Explained Variance Score: {explained_variance}')
        # print(f'Median Absolute Error: {medae}')
        # print(f'Max Error: {max_err}')


def train_model_v2():
    prepare('.\MedVisionaries\Images')
    df = read_adjust_csv()
    X = df[['Fundus']]
    y = df[['Age', 'N', 'D', 'G', 'C', 'A', 'H', 'M', 'O']]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train['Fundus'] = X_train.apply(lambda row: update_path_and_extract(row, 'Fundus'), axis=1)
    X_test['Fundus'] = X_test.apply(lambda row: update_path_and_extract(row, 'Fundus'), axis=1)

    # Feature Scaling and Imputation
    numeric_features = ['Fundus']
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    # Combine transformers using ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features)
        ])

    # Define the model
    model = MultiOutputRegressor(RandomForestRegressor(random_state=42))

    # Create a pipeline with preprocessor and model
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('model', model)])

    # Train the model
    pipeline.fit(X_train, y_train)

    # Predict on the test set
    y_pred = pipeline.predict(X_test)
    print_evaluation_metrics(y_pred, y_test, y)
    return pipeline


def train_model():
    # Load the dataset
    df = pd.read_csv(file_path)

    # Split the data into features (X) and target variables (y)
    X = df[['Left_Fundus', 'Right_Fundus', 'Sex']]
    y = df[['N', 'D', 'G', 'C', 'A', 'H', 'M', 'O']]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train['Left_Fundus'] = X_train.apply(lambda row: update_path_and_extract(row, 'Left_Fundus'), axis=1)
    X_train['Right_Fundus'] = X_train.apply(lambda row: update_path_and_extract(row, 'Right_Fundus'), axis=1)
    X_test['Left_Fundus'] = X_test.apply(lambda row: update_path_and_extract(row, 'Left_Fundus'), axis=1)
    X_test['Right_Fundus'] = X_test.apply(lambda row: update_path_and_extract(row, 'Right_Fundus'), axis=1)

    # Handle categorical variables using Label Encoding
    label_encoder = LabelEncoder()
    X_train['Sex'] = label_encoder.fit_transform(X_train['Sex'])
    X_test['Sex'] = label_encoder.transform(X_test['Sex'])

    # Feature Scaling and Imputation
    numeric_features = ['Left_Fundus', 'Right_Fundus']
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    # Combine transformers using ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features)
        ])

    # Insert CNN here

    # Define the model
    model = MultiOutputRegressor(RandomForestRegressor(random_state=42))

    # Create a pipeline with preprocessor and model
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('model', model)])

    # Train the model
    pipeline.fit(X_train, y_train)

    # Predict on the test set
    y_pred = pipeline.predict(X_test)

    # Evaluate the model for each output variable
    for i, target_variable in enumerate(y.columns):  # Corrected variable name
        mae = mean_absolute_error(y_test[target_variable], y_pred[:, i])
        mse = mean_squared_error(y_test[target_variable], y_pred[:, i])
        r2 = r2_score(y_test[target_variable], y_pred[:, i])
        explained_variance = explained_variance_score(y_test[target_variable], y_pred[:, i])
        medae = median_absolute_error(y_test[target_variable], y_pred[:, i])
        max_err = max_error(y_test[target_variable], y_pred[:, i])

        print(f'Evaluation metrics for {target_variable}:')
        print(f'Mean Absolute Error: {mae}')
        print(f'Mean Squared Error: {mse}')
        # print(f'R2 Score: {r2}')
        # print(f'Explained Variance Score: {explained_variance}')
        # print(f'Median Absolute Error: {medae}')
        # print(f'Max Error: {max_err}')
    return pipeline


def pre_check(file_path):
    img = cv2.imread(file_path)
    assert img is not None, "Image not found"


def model(file_path):
    """
    The model takes an image as input and returns the predicted age.

    Parameters
    ----------
    image : str
        The file path of the input image.

    Returns
    -------
    int
        The predicted age.

    """
    # Preprocess the image
    pre_check(file_path)
    pipeline = train_model_v2()
    preprocessed_image = Preprocessor(file_path)
    df = pd.DataFrame({'Fundus': [np.mean(preprocessed_image)]})
    pred = pipeline.predict(df)
    return pred[0][0]
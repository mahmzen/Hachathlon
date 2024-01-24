#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Original Template Author: Jan Ruhland, modified by MedVisionaries

import numpy as np
import cv2
import numpy as np
import os
import random


def Preprocessor(image_path):
    """
    This function resizes and normalizes the input image.

    Parameters
    ----------
    image : string
        Path to the file.

    Returns
    -------
    image : np array
        Resized and normalized image.

    """
    # Example: Process and display images for the first person in the dataset
    img = cv2.imread(image_path)
    avg_width, avg_height = calculate_average_dimensions('./resized_images')
    aspect_ratio = avg_width / avg_height

    # Decide on a target height (or width) and calculate the other dimension
    # For example, if you choose a target height of 512
    target_height = 512
    target_width = int(target_height * aspect_ratio)
    resized_image = resize_image(img, target_width, target_height)
    return resized_image


def calculate_average_dimensions(image_folder, sample_size=1000):
    print(os.getcwd())
    files = [file for file in os.listdir(image_folder) if file.endswith('.jpg')]
    sampled_files = random.sample(files, min(sample_size, len(files)))

    widths, heights = [], []
    for filename in sampled_files:
        img_path = os.path.join(image_folder, filename)
        img = cv2.imread(img_path)
        h, w, _ = img.shape
        heights.append(h)
        widths.append(w)

    avg_width = int(np.mean(widths))
    avg_height = int(np.mean(heights))

    return avg_width, avg_height


def prepare(image_folder):
    print("Preparing images...")
    avg_width, avg_height = calculate_average_dimensions(image_folder)
    aspect_ratio = avg_width / avg_height
    target_height = 512
    target_width = int(target_height * aspect_ratio)
    resize_images(image_folder, target_width, target_height)


def resize_images(image_folder, target_width, target_height):
    output_folder = 'resized_images'

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # If Output folder is not empty, return early
    if os.listdir(output_folder):
        print("Output folder is not empty. Skipping resizing.")
        return

    for filename in os.listdir(image_folder):
        if filename.endswith('.jpg'):
            img_path = os.path.join(image_folder, filename)
            img = cv2.imread(img_path)

            # Calculate the ratio and resize
            h, w, _ = img.shape
            scaling_factor = min(target_width / w, target_height / h)
            new_dimensions = (int(w * scaling_factor), int(h * scaling_factor))

            resized_img = cv2.resize(img, new_dimensions, interpolation=cv2.INTER_AREA)

            # Save the resized image in the output folder
            print(f"Saving {filename}... {output_folder}/{filename}")
            cv2.imwrite(os.path.join(output_folder, filename), resized_img)


def resize_image(image, target_width, target_height):
    # TODO: use sklearn.preprocessing.StandardScaler to normalize the image
    # Calculate the ratio and resize
    h, w, _ = image.shape
    scaling_factor = min(target_width / w, target_height / h)
    new_dimensions = (int(w * scaling_factor), int(h * scaling_factor))

    resized_img = cv2.resize(image, new_dimensions, interpolation=cv2.INTER_AREA)

    return resized_img


def process_and_display_images(row, image_folder, aspect_ratio):
    """
    Process and display left and right fundus images for a given dataset row.

    :param row: Row of the pandas dataframe.
    :param image_folder: Folder where images are stored.
    """
    # Check if resized images already exist
    left_resized_path = os.path.join('resized_images', f"resized_left_{row['ID']}.jpg")
    right_resized_path = os.path.join('resized_images', f"resized_right_{row['ID']}.jpg")

    if os.path.exists(left_resized_path) and os.path.exists(right_resized_path):
        print(f"Resized images for ID {row['ID']} already exist.")
        return

    # Use average dimensions to calculate target dimensions

    # Decide on a target height (or width) and calculate the other dimension
    # For example, if you choose a target height of 512
    target_height = 512
    target_width = int(target_height * aspect_ratio)

    # Resize images using custom function
    resize_images(image_folder, target_width, target_height)

    # Load resized images
    left_image_resized = cv2.imread(left_resized_path)
    right_image_resized = cv2.imread(right_resized_path)

    if left_image_resized is not None and right_image_resized is not None:
        # Display images
        cv2.imshow(f"Left Eye of ID {row['ID']}", left_image_resized)
        cv2.imshow(f"Right Eye of ID {row['ID']}", right_image_resized)

        # Wait for a key press and then close all windows
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print(f"Resized images for ID {row['ID']} not found.")
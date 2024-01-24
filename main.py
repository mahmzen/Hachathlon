#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Original Template Author: Jan Ruhland, modified by MedVisionaries

from model import model

if __name__ == "__main__":
    pathToFile = './MedVisionaries/1_left.jpg'
    age = model(pathToFile)
    print(f'The person corresponding to image {pathToFile} seems to ba around {age} years old.')
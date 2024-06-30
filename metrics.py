# This file will store functions that will present a performance metric of whatever image processing is performed
#
#
# TASKS:-
# Implement metrics as required

import torch
import numpy as np
import cv2

def calc_psnr(img1, img2):
    return cv2.PSNR(img1.numpy(), img2.numpy(), R=1)
# This code allows you to view files using a uniform image renderer
# Command to run code:
# python viewer.py --img-file "path-to-image"

import cv2
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--img-file', type=str, required=True)
args = parser.parse_args()

cv2.imshow("image" ,cv2.imread(args.img_file, cv2.IMREAD_COLOR))
cv2.waitKey(0)
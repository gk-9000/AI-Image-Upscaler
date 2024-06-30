# This file contains helper functions necessary for processing HDF5 data
# Use it to convert the image data to a .h5 file
# Sample Command to run:
# python preprocess.py -o my.h5 -i "testdata/SampleData" --stride 32 --scale 2 --patch-size 32

import h5py
import numpy as np
import glob
import PIL.Image as pil_image
import argparse


def convert_rgb_to_y(img):
    # Converts rgb image to y
    if type(img) == np.ndarray:
        y = img[:, :, 0]*0.299 +  img[:, :, 1]*0.587 + img[:, :, 2]*0.114
        return y
    else:
        raise Exception('Unknown Type', type(img))


def convert_rgb_to_ycbcr(img):
    # Converts an rgb image to ycbcr
    if type(img) == np.ndarray:
        xform = np.array([[.299, .587, .114], [-.1687, -.3313, .5], [.5, -.4187, -.0813]])
        ycbcr = img.dot(xform.T)
        ycbcr[:,:,[1,2]] += 128
        return ycbcr
    else:
        raise Exception('Unknown Type', type(img))


def convert_ycbcr_to_rgb(img):
    # Converts an image from ycbcr format to rgb
    if type(img) == np.ndarray:
        xform = np.array([[1, 0, 1.402], [1, -0.34414, -.71414], [1, 1.772, 0]])
        rgb = img.astype(np.float32)
        rgb[:,:,[1,2]] -= 128
        rgb = rgb.dot(xform.T)
        np.putmask(rgb, rgb > 255, 255)
        np.putmask(rgb, rgb < 0, 0)
        return rgb
    else:
        raise Exception('Unknown Type', type(img))

def preprocess(args):

    # args is an object returned by an argument parser
    # Required attributes and sample values are provided
    # --output_path path/
    # --images_dir path/
    # --stride 1 
    # --patch size 4 
    # --scale 2  

    h5_file = h5py.File(args.output_path, 'w')

    lr_patches = []
    hr_patches = []

    for image_path in sorted(glob.glob('{}/*'.format(args.images_dir))):
        hr = pil_image.open(image_path).convert('RGB')
        hr_width = (hr.width // args.scale) * args.scale
        hr_height = (hr.height // args.scale) * args.scale
        hr = hr.resize((hr_width, hr_height), resample=pil_image.BICUBIC)
        lr = hr.resize((hr_width // args.scale, hr_height // args.scale), resample=pil_image.BICUBIC)
        lr = lr.resize((lr.width * args.scale, lr.height * args.scale), resample=pil_image.BICUBIC)
        hr = np.array(hr).astype(np.float32)
        lr = np.array(lr).astype(np.float32)
        hr = convert_rgb_to_y(hr)
        lr = convert_rgb_to_y(lr)

        for i in range(0, lr.shape[0] - args.patch_size + 1, args.stride):
            for j in range(0, lr.shape[1] - args.patch_size + 1, args.stride):
                lr_patches.append(lr[i:i + args.patch_size, j:j + args.patch_size])
                hr_patches.append(hr[i:i + args.patch_size, j:j + args.patch_size])

    lr_patches = np.array(lr_patches)
    hr_patches = np.array(hr_patches)

    h5_file.create_dataset('lr', data=lr_patches)
    h5_file.create_dataset('hr', data=hr_patches)

    h5_file.close()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="preprocess the data")
    parser.add_argument('-o','--output-path', type=str, metavar="", required=True, help='output h5 file')
    parser.add_argument('-i', '--images-dir', type=str, metavar="", required=True, help='image directory')
    parser.add_argument('--stride', type=int, metavar="", default=1, help='stride')
    parser.add_argument('--patch-size', type=int, metavar="", default=4, help='patch size')
    parser.add_argument('-s', '--scale', type=int, metavar="", required=True, help='scale')

    args = parser.parse_args()

    preprocess(args)
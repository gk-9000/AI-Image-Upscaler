# This script will be used to manually test model on given image files
# Use appropriate command line arguments and conditions
# python test.py -w "outputs/x4/epoch50.pth" -i "testdata/SampleData/LowRes2.jpg" -s 4 -o testdata/SampleData/sr.png
# python test.py -w "outputs/x4/epoch50.pth" -i "testdata/SampleData/LowRes2.jpg" -s 4

import argparse
from model import Net
import PIL.Image as pil_image
import numpy as np
import torch
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('-w', "--weights-file", type=str, required=True, metavar="", help="model weights file to be used")
parser.add_argument('-i', "--image-file", type=str, required=True, metavar="", help="Image file to be super resolved")
parser.add_argument('-s', "--scale", type=int, required=True, metavar="", help="Upscaling factor")
parser.add_argument('-o', "--output-file", type=str, required=False, metavar="", help="Output file name")

args = parser.parse_args()

def rgb2ycbcr(im):
    xform = np.array([[.299, .587, .114], [-.1687, -.3313, .5], [.5, -.4187, -.0813]])
    ycbcr = im.dot(xform.T)
    ycbcr[:,:,[1,2]] += 128
    return np.uint8(ycbcr)

def ycbcr2rgb(im):
    xform = np.array([[1, 0, 1.402], [1, -0.34414, -.71414], [1, 1.772, 0]])
    rgb = im.astype(np.float32)
    rgb[:,:,[1,2]] -= 128
    rgb = rgb.dot(xform.T)
    np.putmask(rgb, rgb > 255, 255)
    np.putmask(rgb, rgb < 0, 0)
    return np.uint8(rgb)

lr = pil_image.open(args.image_file).convert('RGB')
hr_bi = lr.resize((lr.width * args.scale, lr.height * args.scale), resample=pil_image.BICUBIC)

lr = np.array(lr).astype(np.uint8)
hr_bi = np.array(hr_bi).astype(np.uint8)

lr_ycbcr = rgb2ycbcr(hr_bi)
lr_ycbcr = lr_ycbcr /255.
with torch.no_grad():
    input = torch.from_numpy(lr_ycbcr[:,:,0].astype(np.float32))
    input = input.unsqueeze(-1).unsqueeze(-1).permute([2,3,0,1])   # [h,w] -> [1,1,h,w]
    model = Net()
    model.load_state_dict(torch.load(args.weights_file))
    output = model(input)

    lr_ycbcr[:,:,0] = output[0,0].detach().numpy()
    lr_ycbcr = lr_ycbcr*255
    sr = ycbcr2rgb(lr_ycbcr)

if args.output_file is not None:
    sr_img = pil_image.fromarray(sr)
    sr_img.save(args.output_file)

plt.subplot(1, 3, 1)
plt.title("original image")
plt.imshow(lr)

plt.subplot(1, 3, 2)
plt.title("bicubic interpolation")
plt.imshow(hr_bi)

plt.subplot(1, 3, 3)
plt.title("sr image")
plt.imshow(sr)

plt.show()

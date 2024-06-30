# This script will be used to generate super resolved videos.
# python vsr.py -w "outputs/x3/epoch36.pth" -v "vsr/vid.mp4" -o "vsr/vidsr.avi" -s 3

from model import Net
import PIL.Image as pil_image
import numpy as np
import torch
import cv2
import argparse

parser = argparse.ArgumentParser("Video Super resolution")
parser.add_argument('-w', "--weights-file", type=str, required=True, metavar="", help="model weights file to be used")
parser.add_argument('-v', "--video-file", type=str, required=True, metavar="", help="Video file to be super resolved")
parser.add_argument('-o', "--output-file", type=str, required=False, metavar="", help="Output video file name with .avi format")
parser.add_argument('-s', "--scale", type=int, required=True, metavar="", help="Upscaling factor")
args = parser.parse_args()

vid_capture = cv2.VideoCapture(args.video_file)
frame_width = int(vid_capture.get(3)) * args.scale
frame_height = int(vid_capture.get(4)) * args.scale
frame_size = (frame_width,frame_height)
fps = vid_capture.get(cv2.CAP_PROP_FPS)
out_vid = cv2.VideoWriter(args.output_file, cv2.VideoWriter_fourcc(*'XVID'), fps, frame_size)

while(vid_capture.isOpened()):
    ret, frame = vid_capture.read()
    if ret == True:
        frame_bi = cv2.resize(frame, frame_size, interpolation=cv2.INTER_CUBIC)
        frame_ycc = cv2.cvtColor(frame_bi, cv2.COLOR_BGR2YCR_CB)
        y, _, _ = cv2.split(frame_ycc)
        input = torch.from_numpy(y.astype(np.float32))
        input = input.unsqueeze(-1).unsqueeze(-1).permute([2,3,0,1])
        input = input/255.0

        model = Net()
        model.load_state_dict(torch.load(args.weights_file))
        output = model(input)
        out_y = output[0,0].detach().numpy()
        out_y *= 255.0
        out_y = out_y.clip(0, 255)
        frame_ycc[:, :, 0] = np.uint8(out_y)

        frame_sr = cv2.cvtColor(frame_ycc, cv2.COLOR_YCR_CB2BGR)
        out_vid.write(frame_sr)
    else:
        break
vid_capture.release()
out_vid.release()
cv2.destroyAllWindows()

# Script to evaluate model based on testdataset 
# Sample Command to run:
# python eval.py -w "outputs/x4/epoch40.pth" -e "testdata/Set5_x4.h5"

from model import Net
from datasets import EvalDataset
import metrics
import torch
import argparse
 
parser = argparse.ArgumentParser("Model Evaluation")
parser.add_argument('-w', "--weights-file", type=str, required=True, metavar="", help="model weights file to be used")
parser.add_argument('-e', "--eval-file", type=str, required=True, metavar="", help="Evaluation h5 file")
args = parser.parse_args()

test_loader = torch.utils.data.DataLoader(dataset=EvalDataset(args.eval_file), 
                                           batch_size=1,
                                           shuffle=False)

model = Net()
model.load_state_dict(torch.load(args.weights_file))

with torch.no_grad():
    n_imgs = len(test_loader)
    tot_psnr_srcnn = 0
    tot_psnr_bicubic = 0
    for i, (lr_img, hr_img) in enumerate(test_loader):
        out_img = model(lr_img)

        psnr_bicubic = metrics.calc_psnr(lr_img, hr_img)
        psnr_srcnn = metrics.calc_psnr(out_img, hr_img)
        print (f'Image [{i+1}/{n_imgs}], psnr_srcnn: {psnr_srcnn} dB, psnr_bicubic: {psnr_bicubic} dB')

        tot_psnr_bicubic+=psnr_bicubic
        tot_psnr_srcnn+=psnr_srcnn

    avg_psnr_srcnn = tot_psnr_srcnn/n_imgs
    avg_psnr_bicubic = tot_psnr_bicubic/n_imgs

    print(f'Average psnr value for srcnn over test set {args.eval_file}: {avg_psnr_srcnn} dB')
    print(f'Average psnr value for bicubic over test set {args.eval_file}: {avg_psnr_bicubic} dB')
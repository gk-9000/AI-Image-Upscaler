# This scripts trains the model
# Use appropriate command line arguments and conditions
# python train.py -t "traindata/HDF5_format/91-image_x4.h5" -o "outputs/" -s 4 -n 2 -e 30

from model import Net
from datasets import TrainDataset
import torch
import argparse

parser = argparse.ArgumentParser("Model training")
parser.add_argument('-t', '--train-file', type=str, required=True, help="H5 file of traindata")
parser.add_argument('-o', '--outputs-dir', type=str, required=True, help="Output diretory for storing model")
parser.add_argument('-s', '--scale', type=int, required=True, help="Upscaling factor")
parser.add_argument('-n', '--num-epochs', type=int, required=True, help="number of epochs to run")
parser.add_argument('-e', '--epoch-done', type=int, required=True, help="number of epochs already done")
args = parser.parse_args()

lr = 1e-5
batch_size = 8
num_workers = 2

train_loader = torch.utils.data.DataLoader(dataset=TrainDataset(args.train_file), 
                                           batch_size=batch_size, 
                                           num_workers=num_workers,
                                           shuffle=True)

model = Net()
model.load_state_dict(torch.load(args.outputs_dir + f"x{args.scale}/epoch{args.epoch_done}.pth"))

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

n_total_steps = len(train_loader)

for epoch in range(args.num_epochs):
    for i, (lr_img, hr_img) in enumerate(train_loader):  

        # Forward pass
        out_img = model(lr_img)
        loss = criterion(out_img, hr_img)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 400 == 0:
            print (f'Epoch [{epoch+1}/{args.num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.6f}')
    if(epoch+1)%2 ==0:
        torch.save(model.state_dict(), args.outputs_dir + f"x{args.scale}/epoch{epoch+args.epoch_done+1}.pth" )
print('Finished Training')

torch.save(model.state_dict(), args.outputs_dir + f"x{args.scale}/epoch{args.num_epochs + args.epoch_done}.pth" )
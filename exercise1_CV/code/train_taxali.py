import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
import torch
import torch.nn as nn

from model.model import ResNetModel
from model.data import get_data_loader
from utils.plot_util import plot_keypoints
from run_forward import normalize_keypoints

import argparse
import pickle

PATH = "model/i_taxali"
OPATH = "model/o_taxali.pth"

epochs=10
training_errors = []
val_loss=[]
validation_errors = []
mean_pixel_errors = []

# initialising device and model to the device
print("initializing model, cuda ...")
cuda = torch.device('cuda')
model = ResNetModel(pretrained=False)
#model.load_state_dict(torch.load(PATH_TO_CKPT))
model.to(cuda)


# get data loaders
train_loader = get_data_loader(batch_size=32, is_train=True, single_sample=True)
val_loader = get_data_loader(batch_size=32, is_train=False)

#Defining optimiyer and loss
optimizer = optim.Adam(model.parameters(), lr=1e-4)
loss_fn = torch.nn.MSELoss(reduction='none')

print("training ...")
for e in range (epochs):
  train_loss=0
  for idx, (img, keypoints, weights) in enumerate(val_loader):
    optimizer.zero_grad()
    img = img.to(cuda)
    keypoints = keypoints.to(cuda)
    weights = weights.to(cuda)
    #print(weights.shape)
    keypoints = normalize_keypoints(keypoints, img.shape)
    output = model(img, '')
    #print(weights.repeat_interleave(2,dim=1).shape)
    loss = loss_fn(output, keypoints)*(weights.repeat_interleave(2,dim=1).float())
    loss = torch.sum(loss)
    train_loss += loss.item()
    loss.backward()
    optimizer.step()

  training_errors.append(train_loss/len(train_loader))
  print("Epoch: {}/{}..: avg. training loss = {}".format(e+1, epochs, training_errors[-1]))    
  if epochs % 5 == 0: 
        with torch.no_grad():
            model.eval()
            val_loss = 0
            mpjpe = 0
            for idx, (img, keypoints, weights) in enumerate(val_loader):
                img = img.to(cuda)
                keypoints = keypoints.to(cuda)
                weights = weights.to(cuda)
                output = model(img, '')
                loss = loss_fn(output, keypoints)*(weights.repeat_interleave(2,dim=1).float())
                visible = torch.sum(weights>0.5).item()
                mpjpe += (torch.sum(torch.sqrt(loss))/visible).item()
                val_loss += torch.sum(loss).item()
            validation_errors.append(val_loss/len(val_loader))
            mean_pixel_errors.append(mpjpe/len(val_loader))
            # add later : val_loss/len(val_loader)
            print("validation loss : {}, MPJPE : {} pixels".format(validation_errors[-1] ,mean_pixel_errors[-1]))

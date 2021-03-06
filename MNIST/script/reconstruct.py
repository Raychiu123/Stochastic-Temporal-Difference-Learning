import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import sys
from model_one_layer_3T_v3 import *
from prep_data import *
import time
import os

timestr = time.strftime("%Y%m%d-%H%M")

""" After training the model, we can try to use the model to do
jumpy predictions.
"""

#### load trained model
pt = "SunMay12_model_onelayer_3T_v3_epoch_7899.pt"
pth = "../output/model_ckpt/model_onelayer_3T_v3/"
p_model = pth + pt

img_dir = "model_onelayer_3T_v3"

if not os.path.isdir(pth + img_dir):
    os.mkdir(pth + img_dir)

checkpoint = torch.load(p_model)
input_size = 784
processed_x_size = 784
belief_state_size = 50
state_size = 8
tdvae = TD_VAE(input_size, processed_x_size, belief_state_size, state_size)
optimizer = optim.Adam(tdvae.parameters(), lr = 0.0005)

tdvae.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

#### load dataset 
with open("../data/MNIST.pkl", 'rb') as file_handle:
    MNIST = pickle.load(file_handle)
tdvae.eval()
tdvae = tdvae.cuda()

data = MNIST_Dataset(MNIST['train_image'], binary = False)
batch_size = 6
data_loader = DataLoader(data,
                         batch_size = batch_size,
                         shuffle = True)
idx, images = next(enumerate(data_loader))

images = images.cuda()

## calculate belief
tdvae.forward(images)

## jumpy rollout
t1, t2 = 11, 15
rollout_images = tdvae.reconstrcut(images, t1, t2)

#### plot results
fig = plt.figure(0, figsize = (t2+2,batch_size))
#fig = plt.figure(0, figsize = (12,4))

#fig = plt.figure(0)
fig.clf()
gs = gridspec.GridSpec(batch_size,t2+2)
gs.update(wspace = 0.05, hspace = 0.05)
for i in range(batch_size):
    for j in range(t1):
        axes = plt.subplot(gs[i,j])
        axes.imshow(1-images.cpu().data.numpy()[i,j].reshape(28,28),
                    cmap = 'binary')
        axes.axis('off')

    for j in range(t1,t2+1):
        axes = plt.subplot(gs[i,j+1])
        axes.imshow(1-rollout_images.cpu().data.numpy()[i,j-t1].reshape(28,28),
                    cmap = 'binary')
        axes.axis('off')

timestr = timestr + '_reconstruct_result.png'
fig.savefig(pth + img_dir + timestr)
plt.show()
sys.exit()

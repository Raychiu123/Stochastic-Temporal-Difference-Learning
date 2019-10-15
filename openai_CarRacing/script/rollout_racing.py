import pickle
from sklearn.externals import joblib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import sys
#from model import *
from model_one_layer_3T import *
from prep_data import *
import time
import os

timestr = time.strftime("%Y%m%d-%H%M%S")
batch_size = 6
""" After training the model, we can try to use the model to do
jumpy predictions.
"""

#### load trained model
pt = "SunAug18_model_onelayer_3T_epoch_8100.pt"
pth = "../output/model_ckpt/model_onelayer_3T_racing/"
p_model = pth + pt

img_dir = "model_onelayer_3T_racing/"

if not os.path.isdir(pth + img_dir):
    os.mkdir(pth + img_dir)

checkpoint = torch.load(p_model)
input_size = 200*200*3
processed_x_size = 784
belief_state_size = 50
state_size = 20
tdvae = TD_VAE(input_size, processed_x_size, belief_state_size, state_size).cuda()
optimizer = optim.Adam(tdvae.parameters(), lr = 0.0005)

tdvae.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

#### load dataset 
with open("../logs/racing_norm.pkl", 'rb') as file_handle:
    OPENAI = joblib.load(file_handle)
    print("dataset loaded!")

#OPENAI = OPENAI[:,:,150:350,100:300,:]
print(OPENAI.shape)
OPENAI = OPENAI.reshape(27,480,3,200,200)

opai = []
for i in range(24):
    opai.append(OPENAI[:,20*i:20*i+20,:,:,:])

print(np.array(opai).shape)
OPENAI = np.array(opai).reshape(648,20,3,200,200)
del opai

print(OPENAI.shape)
data_loader = DataLoader(OPENAI,
                         batch_size = 6,
                         shuffle = True)
idx, images = next(enumerate(data_loader))

images = images.view(-1,3,200,200).type(torch.FloatTensor).cuda()

## calculate belief
tdvae.forward(images)

## jumpy rollout
t1, t2 = 11, 15
rollout_images = tdvae.rollout(images, t1, t2).view(6,-1,200,200,3)
images = images.view(-1,20,200,200,3)
#### plot results
fig = plt.figure(0, figsize = (t2+2,6))
#fig = plt.figure(0, figsize = (12,4))

#fig = plt.figure(0)
fig.clf()
gs = gridspec.GridSpec(6,t2+2)
gs.update(wspace = 0.05, hspace = 0.05)
for i in range(6):
    for j in range(t1):
        axes = plt.subplot(gs[i,j])
        axes.imshow((images.cpu().data.numpy()[i,j,:,:,:]*255).astype(np.uint8))
        axes.axis('off')

    for j in range(t1,t2+1):
        axes = plt.subplot(gs[i,j+1])
        axes.imshow((rollout_images.cpu().data.numpy()[i,j-t1,:,:,:]*255).astype(np.uint8))
        axes.axis('off')

timestr = timestr+ '_rollout_result.png'
fig.savefig(pth + img_dir + timestr)
plt.show()
sys.exit()

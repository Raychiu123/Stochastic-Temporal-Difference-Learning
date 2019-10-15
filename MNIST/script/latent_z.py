import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.manifold import TSNE
import sys
from model_5T import *
from prep_data import *
import os
import scipy.io as sio

#### load trained model
pt = "TueOct8_model_5T_epoch_8999.pt"
pth = "../output/model_ckpt/model_5T/"
p_model = pth + pt

#img_dir = "model_3T_vb2/"

#if not os.path.isdir(pth + img_dir):
#    os.mkdir(pth + img_dir)

checkpoint = torch.load(p_model)
input_size = 784
processed_x_size = 784
belief_state_size = 50
state_size = 8
stdl = STDL(input_size, processed_x_size, belief_state_size, state_size)

stdl.load_state_dict(checkpoint['model_state_dict'])

#### load dataset
with open("../data/MNIST.pkl", 'rb') as file_handle:
    MNIST = pickle.load(file_handle)
stdl.eval()
stdl = stdl.cuda()

data = MNIST_Dataset(MNIST['train_image'], MNIST['train_label'], binary = False)
data_lable = MNIST['train_label']
#print(data_lable)

batch_size = 2048
data_loader = DataLoader(data,
                         batch_size = batch_size,
                         shuffle = False)

idx, (images, labels) = next(enumerate(data_loader))

images = images.cuda()
labels = labels.cuda()

color_list = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
label_list = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

z_jump = []
l_jump = []

z = stdl.latent_z(images=images).view(-1, state_size).data.cpu().numpy()
z_jump = stdl.latent_z(images=images[0].view(1,20,784), rollout=True).view(-1, state_size).data.cpu().numpy()

#print(z.shape)
#print(z_jump.shape)

z = np.concatenate((z, z_jump))
#print(labels[:2,:])
labels = labels.view(-1,1)
#print(labels[:40])

print("TSNE...")
z_2d = TSNE(n_components=2, perplexity = 30 , random_state = 0).fit_transform(z)
print("TSNE done!!!")
#print(z_2d.shape)
z_jump = z_2d[-6:,:]
z_2d = z_2d[:-6,:]

#print(z_jump[0][0])
#z_jump.append(z_2d[0])
#l_jump.append(labels[0,i])

for i in range(10):
	plt.scatter(0,0, c = color_list[i], s = 2.5, label=label_list[i])
plt.scatter(0,0, c='white', alpha=1,s = 2.5, edgecolors = 'none')

for i in range(batch_size*20):
	if i%batch_size ==0:
		print("Number %d" %(i/batch_size))
	color = color_list[labels[i,0]]
	plt.scatter(z_2d[i,0], z_2d[i,1], c = color, s = 1.5)

for i in range(6):
	plt.scatter(z_jump[i][0],z_jump[i][1], c = color_list[labels[i,0]], s = 1.5)
	plt.annotate("X", (z_jump[i][0],z_jump[i][1]))

plt.legend()
file = str(batch_size) + 'lnt_5T.png'
plt.savefig(file)

#sio.savemat("results_z_tdvae.mat", z_2d)
#sio.savemat("results_z_jump_tdvae.mat", z_jump)

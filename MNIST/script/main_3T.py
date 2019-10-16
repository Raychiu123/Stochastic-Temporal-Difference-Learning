### markov transition + auxiliary cost
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from model_3T import *
from prep_data import *
import sys
import time
import os
import matplotlib.pyplot as plt
from matplotlib import gridspec


pth = "../output/"
img_dir = "model_3T/"

if not os.path.isdir(pth + img_dir):
    os.mkdir(pth + img_dir)

p_dir = "model_ckpt/"
if not os.path.isdir(pth + p_dir + img_dir):
    os.mkdir(pth + p_dir + img_dir)

#### preparing dataset
with open("../data/MNIST.pkl", 'rb') as file_handle:
    MNIST = pickle.load(file_handle)

data = MNIST_Dataset(MNIST['train_image'], MNIST['train_label'])
data_test = MNIST_Dataset(MNIST['test_image'], MNIST['test_label'])

batch_size = 512*16
data_loader = DataLoader(data,
                         batch_size = batch_size,
                         shuffle = True)
data_loader_t = DataLoader(data_test, batch_size = 6,
                         shuffle = True)

del data
del data_test

#### build a TD-VAE model
input_size = 784
processed_x_size = 784
belief_state_size = 50
state_size = 8
stdl = STDL(input_size, processed_x_size, belief_state_size, state_size)
stdl = stdl.cuda()

path1 = time.asctime(time.localtime(time.time()))
path1 = path1[0:11].replace(" ","")
path1 = "../log/" + path1 + "_loginfo_model_3T.txt"
#### training
optimizer = optim.Adam(stdl.parameters(), lr = 0.0005)
num_epoch = 9000
log_file_handle = open(path1, 'w')

loss_pkl = []

for epoch in range(num_epoch):
    for idx, (images,_) in enumerate(data_loader):   
        images = images.cuda()       
        stdl.forward(images)
        t_1 = np.random.choice(12)
        t_2 = t_1 + np.random.choice([1,2,3,4])
        t_3 = t_2 + np.random.choice([1,2,3,4])
        loss = stdl.calculate_loss(t_1, t_2, t_3)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print("epoch: {:>4d}, idx: {:>4d}, loss: {:.2f}".format(epoch, idx, loss.item()),
              file = log_file_handle, flush = True)
        
    print("epoch: {:>4d}, loss: {:.2f}".format(epoch, loss.item()))
    loss_pkl.append(loss.item())

    if (epoch + 1) % 200 == 0:
        path = time.asctime(time.localtime(time.time()))
        path = path[0:11].replace(" ","")
        path = pth + p_dir + img_dir + path + "_model_3T_epoch_{}.pt"
        torch.save({
            'epoch': epoch,
            'model_state_dict': stdl.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss
        }, path.format(epoch))
        stdl.eval()

        idx, images = next(enumerate(data_loader_t))
        images = images.cuda()

        ## calculate belief
        stdl.forward(images)

        ## jumpy rollout
        t1, t2 = 11, 15
        rollout_images = stdl.rollout(images, t1, t2)
        reconstruct_images = stdl.reconstruct(images, t1, t2)

        #### plot results
        fig = plt.figure(0, figsize = (t2+2,6))
        fig.clf()
        gs = gridspec.GridSpec(6,t2+2)
        gs.update(wspace = 0.05, hspace = 0.05)
        for i in range(6):
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

        path = time.asctime(time.localtime(time.time()))
        path = path[0:11].replace(" ","")
        path = str(epoch) + path + '_rollout_result.png'
        fig.savefig(pth + img_dir + path)

        fig1 = plt.figure(0, figsize = (t2+2,6))
        fig1.clf()
        gs1 = gridspec.GridSpec(6,t2+2)
        gs1.update(wspace = 0.05, hspace = 0.05)
        for i in range(6):
            for j in range(t1):
                axes = plt.subplot(gs1[i,j])
                axes.imshow(1-images.cpu().data.numpy()[i,j].reshape(28,28),
                    cmap = 'binary')
                axes.axis('off')

            for j in range(t1,t2+1):
                axes = plt.subplot(gs1[i,j+1])
                axes.imshow(1-reconstruct_images.cpu().data.numpy()[i,j-t1].reshape(28,28),
                    cmap = 'binary')
                axes.axis('off')

        path = time.asctime(time.localtime(time.time()))
        path = path[0:11].replace(" ","")
        path = str(epoch) + path + '_reconstruct_result.png'
        fig1.savefig(pth + img_dir + path)        

        stdl.train()
        
torch.save(loss_pkl, "../log/loss.pkl")

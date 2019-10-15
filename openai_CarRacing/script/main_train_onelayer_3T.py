import pickle
from sklearn.externals import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from model_one_layer_3T import *
import time
import os
import matplotlib.pyplot as plt
from matplotlib import gridspec

batch_size = 32

pth = "../output/"
img_dir = "model_onelayer_3T_racing/"

if not os.path.isdir(pth + img_dir):
    os.mkdir(pth + img_dir)

p_dir = "model_ckpt/"
if not os.path.isdir(pth + p_dir + img_dir):
    os.mkdir(pth + p_dir + img_dir)

#### preparing dataset
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
data_loader1 = DataLoader(OPENAI[51:171,:,:,:,:],
                         batch_size = 6,
                         shuffle = True)

#_, test_images = next(enumerate(data_loader))
#test_images = test_images.view(-1,3,400,400).type(torch.FloatTensor).cuda()

data_loader = DataLoader(OPENAI,
                         batch_size = batch_size,
                         shuffle = True)

#data_loader_t = DataLoader(OPENAI,
#                         batch_size = 6,
#                         shuffle = True)


#### build a TD-VAE model
input_size = 200*200*3
processed_x_size = 784
belief_state_size = 50
state_size = 20
tdvae = TD_VAE(input_size, processed_x_size, belief_state_size, state_size)
tdvae = tdvae.cuda()
print(tdvae)

path1 = time.asctime(time.localtime(time.time()))
path1 = path1[0:11].replace(" ","")
path1 = "../logs/" + path1 + "_loginfo_model_onelayer_3T.txt"
#### training
optimizer = optim.Adam(tdvae.parameters(), lr = 0.00005)
num_epoch = 9000
log_file_handle = open(path1, 'w')
for epoch in range(num_epoch):
    for idx, images in enumerate(data_loader):
        ## binarize MNIST images
        #tmp = np.random.rand(28,28)
        #images = tmp <= images
        #images = images.astype(np.float32)
        #print("image size, ",images.size())

        images = images.view(-1,3,200,200).type(torch.FloatTensor).cuda()
        
        tdvae.forward(images)
        t_1 = np.random.choice(12)
        t_2 = t_1 + np.random.choice([1,4])
        t_3 = t_2 + np.random.choice([1,4])
        loss = tdvae.calculate_loss(t_1, t_2, t_3)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print("epoch: {:>4d}, idx: {:>4d}, loss: {:.2f}".format(epoch, idx, loss.item()),
              file = log_file_handle, flush = True)

    print("epoch: {:>4d}, loss: {:.2f}".format(epoch, loss.item()))

    if (epoch + 1) % 50 == 1:
        path = time.asctime(time.localtime(time.time()))
        path = path[0:11].replace(" ","")
        path = pth + p_dir + img_dir + path + "_model_onelayer_3T_epoch_{}.pt"
        torch.save({
            'epoch': epoch,
            'model_state_dict': tdvae.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss
        }, path.format(epoch))
        tdvae.eval()

        idx, test_images = next(enumerate(data_loader1))
        test_images = test_images.view(-1,3,200,200).type(torch.FloatTensor).cuda()

        ## calculate belief
        #test_images = test_images.view(-1,3,400,400)
        tdvae.forward(test_images)

        ## jumpy rollout
        t1, t2 = 11, 15
        rollout_images = tdvae.rollout(test_images, t1, t2).view(6,-1,200,200,3)
        reconstruct_images = tdvae.reconstruct(test_images, t1, t2).view(6,-1,200,200,3)

        test_images = test_images.view(-1,20,200,200,3)
        #### plot results
        fig = plt.figure(0, figsize = (t2+2,6))
        fig.clf()
        gs = gridspec.GridSpec(6,t2+2)
        gs.update(wspace = 0.05, hspace = 0.05)
        for i in range(6):
            for j in range(t1):
                axes = plt.subplot(gs[i,j])
                axes.imshow((test_images.cpu().data.numpy()[i,j,:,:,:]*255).astype(np.uint8))
                axes.axis('off')

            for j in range(t1,t2+1):
                axes = plt.subplot(gs[i,j+1])
                axes.imshow((rollout_images.cpu().data.numpy()[i,j-t1,:,:,:]*255).astype(np.uint8))
                axes.axis('off')

        path = time.asctime(time.localtime(time.time()))
        path = path[0:11].replace(" ","")
        path = str(epoch) + path + '_rollout_result.png'
        fig.savefig(pth + img_dir + path)
        print("rollout images saved!")

        fig1 = plt.figure(0, figsize = (t2+2,6))
        fig1.clf()
        gs1 = gridspec.GridSpec(6,t2+2)
        gs1.update(wspace = 0.05, hspace = 0.05)
        for i in range(6):
            for j in range(t1):
                axes = plt.subplot(gs1[i,j])
                axes.imshow((test_images.cpu().data.numpy()[i,j,:,:,:]*255).astype(np.uint8))
                axes.axis('off')

            for j in range(t1,t2+1):
                axes = plt.subplot(gs1[i,j+1])
                axes.imshow((reconstruct_images.cpu().data.numpy()[i,j-t1,:,:,:]*255).astype(np.uint8))
                axes.axis('off')

        path = time.asctime(time.localtime(time.time()))
        path = path[0:11].replace(" ","")
        path = str(epoch) + path + '_reconstruct_result.png'
        fig1.savefig(pth + img_dir + path)
        print("reconstructed images saved!")

        tdvae.train()

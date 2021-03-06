#### part of the codes comes from "https://github.com/xqding/TD-VAE"
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from Prob_calculate import *

class DBlock(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DBlock, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(input_size, hidden_size)
        self.fc_mu = nn.Linear(hidden_size, output_size)
        self.fc_logsigma = nn.Linear(hidden_size, output_size)
        
    def forward(self, input):
        t = torch.tanh(self.fc1(input))
        t = t * torch.sigmoid(self.fc2(input))
        mu = self.fc_mu(t)
        logsigma = self.fc_logsigma(t)
        return mu, logsigma

class PreProcess(nn.Module):
    def __init__(self, input_size, processed_x_size):
        super(PreProcess, self).__init__()
        self.input_size = input_size
        self.fc1 = nn.Linear(input_size, processed_x_size)
        self.fc2 = nn.Linear(processed_x_size, processed_x_size)

    def forward(self, input):
        t = torch.relu(self.fc1(input))
        t = torch.relu(self.fc2(t))
        return t

class Decoder(nn.Module):
    def __init__(self, z_size, hidden_size, x_size):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(z_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, x_size)
        
    def forward(self, z):
        t = torch.tanh(self.fc1(z))
        t = torch.tanh(self.fc2(t))
        p = torch.sigmoid(self.fc3(t))
        return p

    
class STDL(nn.Module):
    def __init__(self, x_size, processed_x_size, b_size, z_size):
        super(STDL, self).__init__()
        self.x_size = x_size
        self.processed_x_size = processed_x_size
        self.b_size = b_size
        self.z_size = z_size

        ## input pre-process layer
        self.process_x = PreProcess(self.x_size, self.processed_x_size)
        
        self.lstm = nn.LSTM(input_size = self.processed_x_size,
                            hidden_size = self.b_size,
                            batch_first = True)

        self.b_to_z = DBlock(b_size, 50, z_size)
        
        ## Given belief and state at time t2, infer the state at time t1
        ## infer state
        self.infer_z = DBlock(5*b_size + 4*z_size, 50, z_size)
        
        ## Given the state at time t1, model state at time t2 through state transition
        ## state transition
        self.transition_z = DBlock(z_size, 50, z_size)      

        ## state to observation
        self.z_to_x = Decoder(z_size, 200, x_size)


    def forward(self, images):        
        self.batch_size = images.size()[0]
        self.x = images
        ## pre-precess image x
        self.processed_x = self.process_x(self.x)

        ## aggregate the belief b
        self.b, (h_n, c_n) = self.lstm(self.processed_x)
        
    def calculate_loss(self, t1, t2, t3, t4, t5):
        t5_z_mu, t5_z_logsigma = self.b_to_z(self.b[:, t5, :])
        t5_z_epsilon = torch.randn_like(t5_z_mu)
        t5_z = t5_z_mu + torch.exp(t5_z_logsigma)*t5_z_epsilon

        t4_qs_z_mu, t4_qs_z_logsigma = self.infer_z(
            torch.cat((t5_z.new_zeros(self.b[:,t1,:].size()),
                       t5_z.new_zeros(self.b[:,t2,:].size()),
                       t5_z.new_zeros(self.b[:,t3,:].size()),
                       self.b[:,t4,:],self.b[:,t5,:], 
                       t5_z.new_zeros(t5_z.size()),
                       t5_z.new_zeros(t5_z.size()),
                       t5_z.new_zeros(t5_z.size()),t5_z), dim = -1))
        t4_qs_z_epsilon = torch.randn_like(t4_qs_z_mu)
        t4_qs_z = t4_qs_z_mu + torch.exp(t4_qs_z_logsigma)*t4_qs_z_epsilon

        t4_pb_z_mu, t4_pb_z_logsigma = self.b_to_z(self.b[:, t4, :])
        t4_z_epsilon = torch.randn_like(t4_pb_z_mu)
        t4_z = t4_pb_z_mu + torch.exp(t4_pb_z_logsigma)*t4_z_epsilon

        t3_qs_z_mu, t3_qs_z_logsigma = self.infer_z(
            torch.cat((t4_z.new_zeros(self.b[:,t1,:].size()),
                       t4_z.new_zeros(self.b[:,t2,:].size()),
                       self.b[:,t3,:],self.b[:,t4,:],self.b[:,t5,:],
                       t4_z.new_zeros(t4_z.size()),t4_z.new_zeros(t4_z.size()),t4_z,t5_z), dim = -1))
        t3_qs_z_epsilon = torch.randn_like(t3_qs_z_mu)
        t3_qs_z = t3_qs_z_mu + torch.exp(t3_qs_z_logsigma)*t3_qs_z_epsilon


        ## sample a state at time t3 (belief state)
        t3_pb_z_mu, t3_pb_z_logsigma = self.b_to_z(self.b[:, t3, :])
        t3_z_epsilon = torch.randn_like(t3_pb_z_mu)
        t3_z = t3_pb_z_mu + torch.exp(t3_pb_z_logsigma)*t3_z_epsilon
        
        ## sample a state at time t2
        ## infer state at time t2 based on states at time t3
        t2_qs_z_mu, t2_qs_z_logsigma = self.infer_z(
            torch.cat((t3_z.new_zeros(self.b[:,t1,:].size()),self.b[:,t2,:],self.b[:,t3,:],self.b[:,t4,:],self.b[:,t5,:], 
                       t3_z.new_zeros(t3_z.size()),t3_z,t4_z,t5_z), dim = -1))
        t2_qs_z_epsilon = torch.randn_like(t2_qs_z_mu)
        t2_qs_z = t2_qs_z_mu + torch.exp(t2_qs_z_logsigma)*t2_qs_z_epsilon
        
        ## sample a state at time t2 (belief state)
        t2_pb_z_mu, t2_pb_z_logsigma = self.b_to_z(self.b[:, t2, :])
        t2_z_epsilon = torch.randn_like(t2_pb_z_mu)
        t2_z = t2_pb_z_mu + torch.exp(t2_pb_z_logsigma)*t2_z_epsilon

        ## sample a state at time t1
        ## infer state at time t1 based on states at time t2
        t1_qs_z_mu, t1_qs_z_logsigma = self.infer_z(
            torch.cat((self.b[:,t1,:],self.b[:,t2,:],self.b[:,t3,:],self.b[:,t4,:],self.b[:,t5,:], 
                       t2_z, t3_z, t4_z, t5_z), dim = -1))
        t1_qs_z_epsilon = torch.randn_like(t1_qs_z_mu)
        t1_qs_z = t1_qs_z_mu + torch.exp(t1_qs_z_logsigma)*t1_qs_z_epsilon

        #### After sampling states z from the variational distribution, we can calculate
        #### the loss.

        ## state distribution at time t1 based on belief at time 1
        t1_pb_z_mu, t1_pb_z_logsigma = self.b_to_z(self.b[:, t1, :])        

        ## state distribution at time t2 based on states at time t1 and state transition
        t2_t_z_mu, t2_t_z_logsigma = self.transition_z(t1_qs_z)
        
        ## state distribution at time t3 based on states at time t1, t2 and state transition       
        t3_t_z_mu, t3_t_z_logsigma = self.transition_z(t2_qs_z)

        t4_t_z_mu, t4_t_z_logsigma = self.transition_z(t3_qs_z)

        t5_t_z_mu, t5_t_z_logsigma = self.transition_z(t4_qs_z)
        
        ## observation distribution at time t2 based on state at time t2
        t5_x_prob = self.z_to_x(t5_z)
        #t2_x_prob = self.z_to_x(t2_z) # auxi cost

        #### start calculating the loss
        loss = kl_div_gaussian(t1_qs_z_mu, t1_qs_z_logsigma, t1_pb_z_mu, t1_pb_z_logsigma)
        loss += kl_div_gaussian(t2_qs_z_mu, t2_qs_z_logsigma, t2_pb_z_mu, t2_pb_z_logsigma)        
        loss += kl_div_gaussian(t3_qs_z_mu, t3_qs_z_logsigma, t3_pb_z_mu, t3_pb_z_logsigma)
        loss += kl_div_gaussian(t4_qs_z_mu, t4_qs_z_logsigma, t4_pb_z_mu, t4_pb_z_logsigma)

        loss += gaussian_log_prob(t5_z_mu, t5_z_logsigma, t5_z)
        loss += gaussian_log_prob(t4_pb_z_mu, t4_pb_z_logsigma, t4_z)
        loss += gaussian_log_prob(t3_pb_z_mu, t3_pb_z_logsigma, t3_z)
        loss += gaussian_log_prob(t2_pb_z_mu, t2_pb_z_logsigma, t2_z)
        
        #### variational bottleneck penalty
        loss += -gaussian_log_prob(t5_t_z_mu, t5_t_z_logsigma, t5_z)
        loss += -gaussian_log_prob(t4_t_z_mu, t4_t_z_logsigma, t4_z)
        loss += -gaussian_log_prob(t3_t_z_mu, t3_t_z_logsigma, t3_z)
        loss += -gaussian_log_prob(t2_t_z_mu, t2_t_z_logsigma, t2_z)

        ## observation prob at time t2
        loss += -torch.sum(self.x[:,t5,:]*torch.log(t5_x_prob) + (1-self.x[:,t5,:])*torch.log(1-t5_x_prob), -1)
        #loss += -torch.sum(self.x[:,t2,:]*torch.log(t2_x_prob) + (1-self.x[:,t2,:])*torch.log(1-t2_x_prob), -1)
        loss = torch.mean(loss)
        
        return loss

    def latent_z(self, images, rollout = False):
        self.forward(images)

        if rollout:
            z_list = []
            z_mu, z_logsigma = self.b_to_z(self.b[:,11,:])
            z_epsilon = torch.randn_like(z_mu)
            z = z_mu + torch.exp(z_logsigma)*z_epsilon
            #z_list.append(z)
            current_z = z                       
        
            for k in range(6):
                ## predicting states after time t1 using state transition        
                next_z_mu, next_z_logsigma = self.transition_z(current_z)
                next_z_epsilon = torch.randn_like(next_z_mu)
                next_z = next_z_mu + torch.exp(next_z_logsigma)*next_z_epsilon

                ## generate an observation x_t1 at time t1 based on sampled state z_t1
                z_list.append(next_z)
                current_z = next_z

            z = torch.stack(z_list, dim = 1)

        else:
            z_mu, z_logsigma = self.b_to_z(self.b[:,:,:])
            z_epsilon = torch.randn_like(z_mu)
            z = z_mu + torch.exp(z_logsigma)*z_epsilon

        return z

    def rollout(self, images, t1, t2):
        self.forward(images)
        
        ## at time t1-1, we sample a state z based on belief at time t1-1
        z_mu, z_logsigma = self.b_to_z(self.b[:,t1-1,:])
        z_epsilon = torch.randn_like(z_mu)
        z = z_mu + torch.exp(z_logsigma)*z_epsilon

        current_z = z                       
        rollout_x = []

        for k in range(t2 - t1 + 1):
            ## predicting states after time t1 using state transition        
            next_z_mu, next_z_logsigma = self.transition_z(current_z)
            next_z_epsilon = torch.randn_like(next_z_mu)
            next_z = next_z_mu + torch.exp(next_z_logsigma)*next_z_epsilon

            ## generate an observation x_t1 at time t1 based on sampled state z_t1
            next_x = self.z_to_x(next_z)
            rollout_x.append(next_x)

            current_z = next_z

        rollout_x = torch.stack(rollout_x, dim = 1)
        
        return rollout_x
        
    def reconstruct(self, images, t1, t2):
        self.forward(images)
        rollout_x = []
        for k in range(t2 - t1 + 1):
            ## predicting states after time t1 using state transition        
            next_z_mu, next_z_logsigma = self.b_to_z(self.b[:,t1+k,:])
            next_z_epsilon = torch.randn_like(next_z_mu)
            next_z = next_z_mu + torch.exp(next_z_logsigma)*next_z_epsilon

            ## generate an observation x_t1 at time t1 based on sampled state z_t1
            next_x = self.z_to_x(next_z)
            rollout_x.append(next_x)

        rollout_x = torch.stack(rollout_x, dim = 1)
        
        return rollout_x
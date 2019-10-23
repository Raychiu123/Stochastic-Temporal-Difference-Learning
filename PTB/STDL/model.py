import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from utils import *
import os

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

class Decoder(nn.Module):
    def __init__(self, z_size, x_size, hidden_size):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(z_size + hidden_size, x_size)
        #self.fc2 = nn.Linear(2*hidden_size, 4*hidden_size)
        #self.fc3 = nn.Linear(4*hidden_size, 16*hidden_size)
        #self.fc4 = nn.Linear(16*hidden_size, x_size)

    def forward(self, z):

        t = self.fc1(z)
        #t = torch.tanh(self.fc2(t))
        #p = self.fc3(t)
        #p = torch.tanh(p)
        #p = self.fc4(p)
        p = F.log_softmax(t)
        return p


class STDL(nn.Module):
    def __init__(self, x_size, processed_x_size, b_size, z_size
                , dropout_rate, bos_idx, eos_idx, pad_idx, time_step, time_range
                , lnt_overshooting = False, obs_overshooting = False):
        super(STDL, self).__init__()
        self.x_size = x_size # for vocab_size
        self.processed_x_size = processed_x_size # for embed_size
        self.b_size = b_size # for hidden size
        self.z_size = z_size
        self.dropout_rate = dropout_rate
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx
        self.pad_idx = pad_idx
        self.time_step = time_step
        self.time_range = time_range
        self.lnt_overshooting = lnt_overshooting
        self.obs_overshooting = obs_overshooting
        if self.lnt_overshooting or self.obs_overshooting:
            self.overshooting = True
        else:
            self.overshooting = False

        ## input pre-process layer
        self.process_x = nn.Embedding(self.x_size, self.processed_x_size, padding_idx = pad_idx)

        ## one layer LSTM for aggregating belief states
        self.lstm = nn.LSTM(input_size = self.processed_x_size,
                            hidden_size = self.b_size,
                            num_layers = 1,
                            batch_first = True)
        #self.b_size*=2

        self.b_to_z = DBlock(b_size, 150, z_size)

        ## Given belief and state at time t2, infer the state at time t1
        ## infer state
        self.infer_z = DBlock(b_size + (self.time_step-1)*z_size, 150, z_size)
        ## Given the state at time t1, model state at time t2 through state transition
        ## state transition

        #self.transition_z = nn.LSTM(input_size = self.z_size,
        #                            hidden_size = self.b_size,
        #                            batch_first = True)
        self.transition_z = DBlock(z_size, 50, z_size)

        ## state to observation
        self.z_to_x = Decoder(z_size, x_size, b_size)


    def forward(self, input_data, length, drop):
        self.length = length
        sorted_len, sorted_idx = torch.sort(self.length, descending=True)
        input_data = input_data[sorted_idx]

        self.batch_size = input_data.size()[0]
        self.x = input_data#.float()

        #drop_x = self.dropout(self.x)

        ## embedding x
        self.processed_x = self.process_x(self.x)
        #print(self.processed_x.type())
        #os._exit()
        if drop:
            self.processed_x = F.dropout(self.processed_x, p = self.dropout_rate, training=self.training)

        pack_input = pack_padded_sequence(self.processed_x, sorted_len + 1,
                                          batch_first=True)
        pack_output, _ = self.lstm(pack_input)

        self.b, _= pad_packed_sequence(pack_output, batch_first=True)

        _, reversed_idx = torch.sort(sorted_idx)
        #print("\nreversed_idx:", reversed_idx)
        self.b = self.b[reversed_idx]
        if drop:
            self.b = F.dropout(self.b, p = 0.5, training=self.training)
        #return self.b.size()

    def z_compare(self):
        z_mu, z_logsigma = self.b_to_z(self.b)
        z_epsilon = torch.randn_like(z_mu)
        t_z = z_mu + torch.exp(z_logsigma)*z_epsilon
        return t_z

    def calculate_loss(self, target):

        ## make time list
        t_list = []
        t0 = np.random.choice(self.length.cpu().numpy()-self.time_range*(self.time_step-1))
        t_list.append(t0)
        for i in range(self.time_step-1):
            tmp = t_list[-1] + np.random.choice([1,self.time_range])
            t_list.append(tmp)

        #print("t_list", t_list)
        ## sample a state at different time step
        t_b_z = {'mu': [], 'logsigma': [], 'z': []}
        t_qs_z = {'mu': [], 'logsigma': [], 'z': []}
        t_t_z = {'mu': [], 'logsigma': [], 'z': []}

        for i in range(self.time_step-1):
            z_mu, z_logsigma = self.b_to_z(self.b[:, t_list[-1-i], :])
            z_epsilon = torch.randn_like(z_mu)
            t_z = z_mu + torch.exp(z_logsigma)*z_epsilon

            t_b_z['z'].append(t_z)
            t_b_z['mu'].append(z_mu)
            t_b_z['logsigma'].append(z_logsigma)


            infer_z_input = self.b[:,t_list[-2-i],:]  ######
            for m in range(self.time_step-i-2):
                # padding zeros
                infer_z_input = torch.cat((infer_z_input,
                                              t_b_z['z'][i].new_zeros(t_b_z['z'][i].size())), dim = -1)
            for k in range(i+1):
                # put z input
                infer_z_input = torch.cat((infer_z_input,t_b_z['z'][-1-k]), dim = -1) ######

            #infer_z_input = torch.cat(infer_z_input, dim = -1).view(self.batch_size, -1)
            qs_z_mu, qs_z_logsigma = self.infer_z(infer_z_input)
            qs_z_epsilon = torch.randn_like(qs_z_mu)
            t_q_z = qs_z_mu + torch.exp(qs_z_logsigma)*qs_z_epsilon

            # backward
            t_qs_z['mu'].append(qs_z_mu)
            t_qs_z['logsigma'].append(qs_z_logsigma)
            t_qs_z['z'].append(t_q_z)

        #######################################################################
        z_mu, z_logsigma = self.b_to_z(self.b[:, t_list[0], :])
        z_epsilon = torch.randn_like(z_mu)
        t_z = z_mu + torch.exp(z_logsigma)*z_epsilon
        # backward
        t_b_z['z'].append(t_z)
        t_b_z['mu'].append(z_mu)
        t_b_z['logsigma'].append(z_logsigma)
        #######################################################################

        #print(t_qs_z['z'])
        t_qs_z['z'].reverse()
        t_input = torch.stack(t_qs_z['z'])
        t_qs_z['z'].reverse()

        #h, _ = self.transition_z(t_input.view(self.batch_size, self.time_step-1, -1))
        for i in range(self.time_step-1):
            #t_z_mu, t_z_logsigma = self.b_to_z(h[:,i,:])
            #t_z_epsilon = torch.randn_like(t_z_mu)
            #t_z_z = t_z_mu + torch.exp(t_z_logsigma)*t_z_epsilon
            t_z_mu, t_z_logsigma = self.transition_z(t_input[i])
            t_z_epsilon = torch.randn_like(t_z_mu)
            t_z_z = t_z_mu + torch.exp(t_z_logsigma)*t_z_epsilon

            ## forward
            t_t_z['mu'].append(t_z_mu)
            t_t_z['logsigma'].append(t_z_logsigma)
            t_t_z['z'].append(t_z_z)

        ## reconstruction
        t_x_prob = {'prob':[]}
        tmp = torch.cat((t_b_z['z'][0], self.b[:,t_list[-2],:]), dim = -1).view(-1, self.z_size + self.b_size)
        t_x_prob['prob'].append(self.z_to_x(tmp).view(self.batch_size, -1, self.x_size))

        if self.overshooting:
            t_overshoot_all = []
            t_overshoot = {'mu': [], 'logsigma': [], 'z': []}

            for i in range(self.time_step-1):
                t_overshoot['mu'].append(t_t_z['mu'][i])
                t_overshoot['logsigma'].append(t_t_z['logsigma'][i])
                t_overshoot['z'].append(t_t_z['z'][i])

                for j in range(self.time_step-i-2):
                    t_z_mu, t_z_logsigma = self.transition_z(t_overshoot['z'][-1])
                    t_z_epsilon = torch.randn_like(t_z_mu)
                    t_z_z = t_z_mu + torch.exp(t_z_logsigma)*t_z_epsilon

                    t_overshoot['z'].append(t_z_z)
                    t_overshoot['mu'].append(t_z_mu)
                    t_overshoot['logsigma'].append(t_z_logsigma)

                t_overshoot_all.append(t_overshoot)


                #t_2_x = self.z_to_x(t_t_z['z'][i]).view(self.batch_size, 1, -1)
                #t_x_prob['prob'].append(t_2_x)
                #t_x_prob['prob'].append(self.z_to_x(t_b_z['z'][i]).view(self.batch_size, 1, -1))

        #### start calculating the loss
        loss = 0
        kl_loss = 0
        for i in range(self.time_step-1):
            kl_loss += kl_div_gaussian(t_qs_z['mu'][-1-i], t_qs_z['logsigma'][-1-i],
                                    t_b_z['mu'][-1-i], t_b_z['logsigma'][-1-i])
            loss += kl_div_gaussian(t_qs_z['mu'][-1-i], t_qs_z['logsigma'][-1-i],
                                    t_b_z['mu'][-1-i], t_b_z['logsigma'][-1-i])
            loss += gaussian_log_prob(t_b_z['mu'][i], t_b_z['logsigma'][i], t_b_z['z'][i])

            loss -= gaussian_log_prob(t_t_z['mu'][i], t_t_z['logsigma'][i], t_b_z['z'][i])

        nll = NLL(t_x_prob['prob'][0], target[:,t_list[-1]])
        loss += nll

        over_loss = 0
        if self.overshooting:
            for i in range(self.time_step-1):
                for j in range(self.time_step-i-2):
                    if self.lnt_overshooting:
                        over_loss += kl_div_gaussian(t_overshoot_all[i]['mu'][j],t_overshoot_all[i]['logsigma'][j],
                                            t_b_z['mu'][-2-i-j],t_b_z['logsigma'][-2-i-j])
                    if self.obs_overshooting:
                        tmp = torch.cat((t_overshoot_all[i]['z'][j], self.b[:,t_list[i+j],:])
                                        , dim = -1).view(-1, self.z_size + self.b_size)
                        over_loss += NLL(self.z_to_x(tmp).view(self.batch_size, 1, -1),
                                        target[:,t_list[1+i+j]])
            over_loss = torch.mean(over_loss)

        loss += over_loss
        loss = torch.mean(loss)
        kl_loss = torch.mean(kl_loss)
        return loss, nll, over_loss, kl_loss

    def rollout(self):
        #b_size = self.forward(input_data, length)
        rollout_x = []
        input_z = []

        z_mu, z_logsigma = self.b_to_z(self.b.view(-1, self.b_size))
        z_epsilon = torch.randn_like(z_mu)
        z = z_mu + torch.exp(z_logsigma)*z_epsilon

        next_z_mu, next_z_logsigma = self.transition_z(z)
        next_z_epsilon = torch.randn_like(next_z_mu)
        next_z = next_z_mu + torch.exp(next_z_logsigma)*next_z_epsilon

        input_z = torch.cat((next_z, self.b.view(-1, self.b_size)), dim = -1)
        input_z = F.dropout(input_z, p = self.dropout_rate, training=self.training)

        next_x = self.z_to_x(input_z)#.view(-1, self.z_size + self.b_size))
        next_x = F.dropout(next_x, p = self.dropout_rate, training=self.training)
        next_x = next_x.view(self.batch_size, -1, self.x_size)
        return next_x

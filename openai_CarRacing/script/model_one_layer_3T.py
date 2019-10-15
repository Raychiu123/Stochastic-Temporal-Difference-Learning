### 1 transition(without zero padding) and 1 inference model with padding + auxiliary cost
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from Prob_calculate import *


class DBlock(nn.Module):
    """ A basie building block for parametralize a normal distribution.
    It is corresponding to the D operation in the reference Appendix.
    """
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
    """ The pre-process layer for MNIST image

    """
    def __init__(self, input_size, processed_x_size):
        super(PreProcess, self).__init__()
        self.input_size = input_size
        self.conv1 = nn.Conv2d(3,6,7)
        self.conv2 = nn.Conv2d(6,12,7)

        self.fc1 = nn.Linear(12*10*10, processed_x_size)
        self.fc2 = nn.Linear(processed_x_size, processed_x_size)

    def forward(self, input):
        cv1 = F.max_pool2d(torch.relu(self.conv1(input)), 4)
        #print("cv1 size",cv1.size())
        cv2 = F.max_pool2d(torch.relu(self.conv2(cv1)), 4)
        #print("cv2 size",cv2.size())

        t = cv2.view(-1, 12*10*10)
        t = torch.relu(self.fc1(t))
        #print("t size",t.size())
        t = torch.relu(self.fc2(t))
        #print("t1 size",t.size())
        return t

class Decoder(nn.Module):
    """ The decoder layer converting state to observation.
    Because the observation is MNIST image whose elements are values
    between 0 and 1, the output of this layer are probabilities of
    elements being 1.
    """
    def __init__(self, z_size, hidden_size, x_size):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(z_size, 64)
        #self.fc2 = nn.Linear(hidden_size, hidden_size*8)
        #self.fc3 = nn.Linear(hidden_size*8, hidden_size*64)
        #self.fc4 = nn.Linear(hidden_size*64, hidden_size*64*8)
        #self.fc5 = nn.Linear(hidden_size*64*8, hidden_size*64*64)
        #self.fc6 = nn.Linear(hidden_size*64*8, 3*400*400)

        self.deconv1 = nn.Sequential(nn.ConvTranspose2d(64,64*2,kernel_size=7,stride=5,padding=1),
                                 nn.BatchNorm2d(64*2),
                                 nn.Tanh())

        self.deconv2 = nn.Sequential(nn.ConvTranspose2d(64*2,64*2,kernel_size=7,stride=5,padding=1),
                                 nn.BatchNorm2d(64*2),
                                 nn.Tanh())

        self.deconv3 = nn.Sequential(nn.ConvTranspose2d(64*2,8*8,kernel_size=6,stride=4,padding=1),
                                 nn.BatchNorm2d(8*8),
                                 nn.Tanh())

        self.deconv4 = nn.Sequential(nn.ConvTranspose2d(8*8,3,kernel_size=4,stride=2,padding=1),
                                 nn.Sigmoid())

    def forward(self, z):
        t = torch.tanh(self.fc1(z))
        #print("t size",t.size())
        t = self.deconv1(t.view(t.size()[0],-1,1,1))
        #print("t",t.size())
        t = self.deconv2(t)
        #print("t",t.size())
        t = self.deconv3(t)
        #print("t",t.size())
        t = self.deconv4(t)
        #t = torch.tanh(self.fc3(t))
        #t = torch.tanh(self.fc4(t))
        #t = torch.tanh(self.fc5(t))
        #p = torch.sigmoid(self.fc6(t)).view(3,400,400)
        #print("t",t.size())
        return t


class TD_VAE(nn.Module):
    """ The full TD_VAE model with jumpy prediction.

    First, let's first go through some definitions which would help
    understanding what is going on in the following code.

    Belief: As the model is feed a sequence of observations, x_t, the
      model updates its belief state, b_t, through a LSTM network. It
      is a deterministic function of x_t. We call b_t the belief at
      time t instead of belief state, becuase we call the hidden state z
      state.

    State: The latent state variable, z.

    Observation: The observated variable, x. In this case, it represents
      binarized MNIST images

    """
    def __init__(self, x_size, processed_x_size, b_size, z_size):
        super(TD_VAE, self).__init__()
        self.x_size = x_size
        self.processed_x_size = processed_x_size
        self.b_size = b_size
        self.z_size = z_size

        ## input pre-process layer
        self.process_x = PreProcess(self.x_size, self.processed_x_size)

        ## one layer LSTM for aggregating belief states
        ## One layer LSTM is used here and I am not sure how many layers
        ## are used in the original paper from the paper.
        self.lstm = nn.LSTM(input_size = self.processed_x_size,
                            hidden_size = self.b_size,
                            batch_first = True)

        ## Two layer state model is used. Sampling is done by sampling
        ## higher layer first.
        ## belief to state (b to z)
        ## (this is corresponding to P_B distribution in the reference;
        ## weights are shared across time but not across layers.)
        self.b_to_z = DBlock(b_size, 50, z_size)

        ## Given belief and state at time t2, infer the state at time t1
        ## infer state
        self.infer_z = DBlock(3*b_size + 2*z_size, 50, z_size)

        ## Given the state at time t1, model state at time t2 through state transition
        ## state transition
        self.transition_z = DBlock(z_size, 50, z_size)
        #self.transition_z_1 = DBlock(z_size, 50, z_size)

        ## state to observation
        self.z_to_x = Decoder(z_size, 200, x_size)

    def forward(self, images):
        self.x = images

        ## pre-precess image x
        self.processed_x = self.process_x(self.x)
        #print(self.processed_x.size())
        tmp = self.processed_x.size()[-1]
        self.processed_x = self.processed_x.view(-1,20,tmp)
        self.batch_size = self.processed_x.size()[0]
        #print(self.processed_x.size())
        ## aggregate the belief b
        self.b, (h_n, c_n) = self.lstm(self.processed_x)

    def calculate_loss(self, t1, t2, t3):
        ## Because the loss is based on variational inference, we need to
        ## draw samples from the variational distribution in order to estimate
        ## the loss function.

        ## sample a state at time t3
        t3_z_mu, t3_z_logsigma = self.b_to_z(self.b[:, t3, :])
        #t3_z_logsigma = torch.clamp(t3_z_logsigma, min = -20, max = 20)
        t3_z_epsilon = torch.randn_like(t3_z_mu)
        t3_z = t3_z_mu + torch.exp(t3_z_logsigma)*t3_z_epsilon

        ## sample a state at time t2 (see the reparametralization trick is used)
        t2_qs_z_mu, t2_qs_z_logsigma = self.infer_z(
            torch.cat((t3_z.new_zeros(self.b[:,t2,:].size()),self.b[:,t2,:],self.b[:,t3,:], t3_z.new_zeros(t3_z.size()),t3_z), dim = -1))
        #t2_qs_z_logsigma = torch.clamp(t2_qs_z_logsigma, min = -20, max = 20)
        t2_qs_z_epsilon = torch.randn_like(t2_qs_z_mu)
        t2_qs_z = t2_qs_z_mu + torch.exp(t2_qs_z_logsigma)*t2_qs_z_epsilon

        t2_z_mu, t2_z_logsigma = self.b_to_z(self.b[:, t2, :])
        #t2_z_logsigma = torch.clamp(t2_z_logsigma, min = -20, max = 20)
        t2_z_epsilon = torch.randn_like(t2_z_mu)
        t2_z = t2_z_mu + torch.exp(t2_z_logsigma)*t2_z_epsilon

        ## sample a state at time t1
        ## infer state at time t1 based on states at time t2
        t1_qs_z_mu, t1_qs_z_logsigma = self.infer_z(
            torch.cat((self.b[:,t1,:],self.b[:,t2,:],self.b[:,t3,:], t2_z, t3_z), dim = -1))
        #t1_qs_z_logsigma = torch.clamp(t1_qs_z_logsigma, min = -20, max = 20)
        t1_qs_z_epsilon = torch.randn_like(t1_qs_z_mu)
        t1_qs_z = t1_qs_z_mu + torch.exp(t1_qs_z_logsigma)*t1_qs_z_epsilon

        #### After sampling states z from the variational distribution, we can calculate
        #### the loss.

        ## state distribution at time t1 based on belief at time 1
        t1_pb_z_mu, t1_pb_z_logsigma = self.b_to_z(self.b[:, t1, :])
        #t1_pb_z_logsigma = torch.clamp(t1_pb_z_logsigma, min = -20, max = 20)

        ## state distribution at time t2 based on states at time t1 and state transition
        t2_t_z_mu, t2_t_z_logsigma = self.transition_z(t1_qs_z)
        #t2_t_z_logsigma = torch.clamp(t2_t_z_logsigma, min = -20, max = 20)

        ## state distribution at time t3 based on states at time t1, t2 and state transition
        t3_t_z_mu, t3_t_z_logsigma = self.transition_z(t2_qs_z)
        #t3_t_z_logsigma = torch.clamp(t3_t_z_logsigma, min = -20, max = 20)

        ## observation distribution at time t2 based on state at time t2
        t3_x_prob = self.z_to_x(t3_z).view(self.batch_size, -1) #+ 1e-8
        t2_x_prob = self.z_to_x(t2_z).view(self.batch_size, -1) #+ 1e-8

        #### start calculating the loss

        #### KL divergence between z distribution at time t1 based on variational distribution
        #### (inference model) and z distribution at time t1 based on belief.
        #### This divergence is between two normal distributions and it can be calculated analytically

        ## KL divergence between t1_l2_pb_z, and t1_l2_qs_z
        #loss = 0.5*torch.sum(((t1_pb_z_mu - t1_qs_z)/torch.exp(t1_pb_z_logsigma))**2,-1) + \
               #torch.sum(t1_pb_z_logsigma, -1) - torch.sum(t1_qs_z_logsigma, -1)

        loss = kl_div_gaussian(t1_qs_z_mu, t1_qs_z_logsigma, t1_pb_z_mu, t1_pb_z_logsigma)#.mean()
        a = kl_div_gaussian(t1_qs_z_mu, t1_qs_z_logsigma, t1_pb_z_mu, t1_pb_z_logsigma)
        print("kl loss 1: ", a)

        #### The following four terms estimate the KL divergence between the z distribution at time t2
        #### based on variational distribution (inference model) and z distribution at time t2 based on transition.
        #### In contrast with the above KL divergence for z distribution at time t1, this KL divergence
        #### can not be calculated analytically because the transition distribution depends on z_t1, which is sampled
        #### after z_t2. Therefore, the KL divergence is estimated using samples

        ## state log probabilty at time t2 based on belief
        #loss += torch.sum(-0.5*t2_z_epsilon**2 - 0.5*t2_z_epsilon.new_tensor(2*np.pi) - t2_z_logsigma, dim = -1)

        loss += kl_div_gaussian(t2_qs_z_mu, t2_qs_z_logsigma, t2_t_z_mu, t2_t_z_logsigma)#.mean()
        a = kl_div_gaussian(t2_qs_z_mu, t2_qs_z_logsigma, t2_t_z_mu, t2_t_z_logsigma)
        print("kl loss 2: ", a)

        ## state log probabilty at time t2 based on transition
        #loss += torch.sum(0.5*((t2_z - t2_t_z_mu)/torch.exp(t2_t_z_logsigma))**2 + 0.5*t2_z.new_tensor(2*np.pi) + t2_t_z_logsigma, -1)

        loss += gaussian_log_prob(t3_z_mu, t3_z_logsigma, t3_z)#.mean()
        a = gaussian_log_prob(t3_z_mu, t3_z_logsigma, t3_z)
        print("gaussian loss 1: ", a)

        loss += -gaussian_log_prob(t3_t_z_mu, t3_t_z_logsigma, t3_z)#.mean()
        a = gaussian_log_prob(t3_t_z_mu, t3_t_z_logsigma, t3_z)
        print("gaussian loss 2: ", a)

        #loss -= F.binary_cross_entropy(t3_x_prob, self.x[:,t3,:])

        ## observation prob at time t2
        #print("self.x size: ", self.x.size())
        self.x = self.x.view(self.batch_size, 20, -1)
        
        loss += torch.sum((self.x[:,t3,:]-t3_x_prob)**2, dim = -1)
        a = torch.sum((self.x[:,t3,:]-t3_x_prob)**2, dim = -1)
        print("reconstruct loss 1", a)

        loss += torch.sum((self.x[:,t2,:]-t2_x_prob)**2, dim = -1)
        a = torch.sum((self.x[:,t2,:]-t2_x_prob)**2, dim = -1)
        print("reconstruct loss 2", a)

        #loss += -torch.sum(self.x[:,t3,:]*torch.log(t3_x_prob) + (1-self.x[:,t3,:])*torch.log(1-t3_x_prob), -1)
        #loss += -torch.sum(self.x[:,t2,:]*torch.log(t2_x_prob) + (1-self.x[:,t2,:])*torch.log(1-t2_x_prob), -1)
        #loss += F.binary_cross_entropy(t3_x_prob, self.x[:,t3,:], reduction = 'sum') / self.batch_size
        loss = torch.mean(loss)

        return loss

    def z_compare(self, images):
        self.forward(images)
        z_mu, z_logsigma = self.b_to_z(self.b[:,:,:])
        z_epsilon = torch.randn_like(z_mu)
        z = z_mu + torch.exp(z_logsigma)*z_epsilon

        #t3_z_mu, t3_z_logsigma = self.b_to_z(self.b[:, t3, :])
        #t3_z_epsilon = torch.randn_like(t3_z_mu)
        #t3_z = t3_z_mu + torch.exp(t3_z_logsigma)*t3_z_epsilon

        ## sample a state at time t2 (see the reparametralization trick is used)
        #t2_qs_z_mu, t2_qs_z_logsigma = self.infer_z(
         #   torch.cat((t3_z.new_zeros(self.b[:,t2,:].size()),self.b[:,t2,:],self.b[:,t3,:], t3_z.new_zeros(t3_z.size()),t3_z), dim = -1))

        return z#_mu, z_logsigma, t2_qs_z_mu, t2_qs_z_logsigma

    def rollout1(self, images, t1, t2):
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

    def rollout(self, images, t1, t2):
        self.forward(images)

        ## at time t1-1, we sample a state z based on belief at time t1-1
        z_mu, z_logsigma = self.b_to_z(self.b[:,t1-1,:])
        z_epsilon = torch.randn_like(z_mu)
        z = z_mu + torch.exp(z_logsigma)*z_epsilon

        current_z = z
        rollout_x = []

        for k in range((t2 - t1 + 1)*2):
            ## predicting states after time t1 using state transition
            next_z_mu, next_z_logsigma = self.transition_z(current_z)
            next_z_epsilon = torch.randn_like(next_z_mu)
            next_z = next_z_mu + torch.exp(next_z_logsigma)*next_z_epsilon

            ## generate an observation x_t1 at time t1 based on sampled state z_t1
            next_x = self.z_to_x(next_z)
            if k % 2 == 1:
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

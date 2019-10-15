import torch
from model_3T import *
from model_one_layer import *

input_size = 784
processed_x_size = 784
belief_state_size = 50
state_size = 8

tdvae = TD_VAE(input_size, processed_x_size, belief_state_size, state_size)
stdl = STDL(input_size, processed_x_size, belief_state_size, state_size)

counter1 = 0
for parameter in tdvae.parameters():
    counter1 += parameter.numel()

print("tdvae parameters: ", counter1)

counter2 = 0
for parameter in stdl.parameters():
	counter2 += parameter.numel()  
	  
print("stdl(K=3) parameters: ", counter2)

from model_4T import *
stdl = STDL(input_size, processed_x_size, belief_state_size, state_size)

counter3 = 0
for parameter in stdl.parameters():
	counter3 += parameter.numel()  
	  
print("stdl(K=4) parameters: ", counter3)

from model_5T import *
stdl = STDL(input_size, processed_x_size, belief_state_size, state_size)

counter4 = 0
for parameter in stdl.parameters():
	counter4 += parameter.numel()  
	  
print("stdl(K=4) parameters: ", counter4)
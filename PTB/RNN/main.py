import sys
sys.path.append('..')

import os
import time
import math
import scipy.io as sio
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from multiprocessing import cpu_count

from ptb import PTB
from model import RNN
from utils import *

# device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Penn TreeBank (PTB) dataset
data_path = '../data'
max_len = 96
splits = ['train', 'valid', 'test']
datasets = {split: PTB(root=data_path, split=split) for split in splits}

#print(datasets['train']._symbols['<pad>'])
#os._exit()

# data loader
batch_size = 256
dataloaders = {split: DataLoader(datasets[split],
                                 batch_size=batch_size,
                                 shuffle=split=='train',
                                 num_workers=cpu_count(),
                                 pin_memory=torch.cuda.is_available())
                                 for split in splits}
symbols = datasets['train'].symbols
#####
# RNN model
embedding_size = 300
hidden_size = 256
dropout_rate = 0.5
model = RNN(vocab_size=datasets['train'].vocab_size,
            embed_size=embedding_size,
            time_step=max_len,
            hidden_size=hidden_size,
            dropout_rate=dropout_rate,
            bos_idx=symbols['<bos>'],
            eos_idx=symbols['<eos>'],
            pad_idx=symbols['<pad>'])
model = model.to(device)
print(model)

# initialization
for p in model.parameters():
    p.data.uniform_(-0.1, 0.1)

# folder to save model
save_path = 'model'
if not os.path.exists(save_path):
    os.makedirs(save_path)

# objective function
learning_rate = 0.001
criterion = nn.NLLLoss(size_average=False, ignore_index=symbols['<pad>'])
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# training setting
epoch = 20
print_every = 50

# training interface
step = 0
tracker = {'NLL': []}
start_time = time.time()
for ep in range(epoch):
    # learning rate decay
    if ep >= 10 and ep % 2 == 0:
        learning_rate = learning_rate * 0.5
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate

    for split in splits:
        dataloader = dataloaders[split]
        model.train() if split == 'train' else model.eval()
        totals = {'NLL': 0., 'words': 0}

        for itr, (_, dec_inputs, targets, lengths) in enumerate(dataloader):
            bsize = dec_inputs.size(0)
            dec_inputs = dec_inputs.to(device)
            targets = targets.to(device)
            lengths = lengths.to(device)

            #print("lengths", lengths)
            # forward
            logp = model(dec_inputs, lengths)
            #print("logp.size():", logp.size())

            targets = targets[:, :torch.max(lengths+1).item()].contiguous().view(-1)
            #print("targets.size()", targets.size())
            # calculate loss
            NLL_loss = NLL(logp, targets)
            loss = NLL_loss / bsize

            # cumulate
            totals['NLL'] += NLL_loss.item()
            totals['words'] += torch.sum(lengths).item()
            print(totals)
            #os._exit()
            # backward and optimize
            if split == 'train':
                step += 1
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5)
                optimizer.step()

                # track
                tracker['NLL'].append(loss.item())

                # print statistics
                if itr % print_every == 0 or itr + 1 == len(dataloader):
                    print("%s Batch %04d/%04d, NLL-Loss %.4f, "
                          % (split.upper(), itr, len(dataloader),
                             tracker['NLL'][-1]))

        samples = len(datasets[split])
        print("%s Epoch %02d/%02d, NLL %.4f, PPL %.4f"
              % (split.upper(), ep, epoch, totals['NLL'] / samples,
                 math.exp(totals['NLL'] / totals['words'])))

    # save checkpoint
    checkpoint_path = os.path.join(save_path, "E%02d.pkl" % ep)
    torch.save(model.state_dict(), checkpoint_path)
    print("Model saved at %s\n" % checkpoint_path)
end_time = time.time()
print('Total cost time',
      time.strftime("%H hr %M min %S sec", time.gmtime(end_time - start_time)))

# save learning results
sio.savemat("results.mat", tracker)

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
import torch.nn.utils as utils
import matplotlib.pyplot as plt

from ptb import PTB
from model import *
from utils import *

# device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#device_ids = [0, 1]

# Penn TreeBank (PTB) dataset
data_path = '../data'
#max_len = 96
splits = ['train', 'valid', 'test']
datasets = {split: PTB(root=data_path, split=split) for split in splits}

# data loader
batch_size = 128
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
z_size = 16
dropout_rate = 0.5
time_range = 1
time_step = 4
lnt_overshooting = False
obs_overshooting = True
model = STDL(x_size=datasets['train'].vocab_size,
            processed_x_size=embedding_size,
            b_size=hidden_size,
            z_size=z_size,
            dropout_rate=dropout_rate,
            bos_idx=symbols['<bos>'],
            eos_idx=symbols['<eos>'],
            pad_idx=symbols['<pad>'],
            time_step = time_step,
            time_range = time_range,
            lnt_overshooting = lnt_overshooting,
            obs_overshooting = obs_overshooting)
model = model.to(device)
#torch.backends.cudnn.benchmark = True
#model = model.cuda(device_ids[0])
#model = nn.DataParallel(model, device_ids=device_ids)
print(model)

# initialization
for p in model.parameters():
    p.data.uniform_(-0.1, 0.1)

# folder to save model
save_path = '../output/model_tdvae_v4'
if not os.path.exists(save_path):
    os.makedirs(save_path)

path = time.asctime(time.localtime(time.time()))
path = path[0:11].replace(" ","")

save_path = save_path + '/' + path

if not os.path.exists(save_path):
    os.makedirs(save_path)

save_path = save_path + '/' + "time_step_" + str(time_step)
if not os.path.exists(save_path):
    os.makedirs(save_path)

if lnt_overshooting:
    save_path = save_path + '/' + "with_lnt_overshooting"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
elif obs_overshooting:
    save_path = save_path + '/' + "with_obs_overshooting"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
else:
    save_path = save_path + '/' + "without_overshooting"
    if not os.path.exists(save_path):
        os.makedirs(save_path)


log_file_handle = open(save_path + "/log_file.txt", 'w')
log_file_handle2 = open(save_path + "/log_file_train.txt", 'w')
# objective function
learning_rate = 0.001
criterion = nn.NLLLoss(size_average=False, ignore_index=symbols['<pad>'])
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
#optimizer = nn.DataParallel(optimizer, device_ids=device_ids)

# training setting
epoch = 20
small_epoch = 5
print_every = 10

# training interface
tracker = {'NLL': []}

train_tracker = {'NLL': [], 'PPL': [], 'z_space': []}
valid_tracker = {'NLL': [], 'PPL': [], 'z_space': []}
test_tracker = {'NLL': [], 'PPL': [], 'z_space': []}

best_nll = 1000

start_time = time.time()
for ep in range(epoch):
    # learning rate decay
    #if ep >= 10 and ep % 2 == 0 and learning_rate > 0.00001:
    #    learning_rate = learning_rate * 0.5
    #    for param_group in optimizer.param_groups:
    #        param_group['lr'] = learning_rate

    for split in splits:
        dataloader = dataloaders[split]
        model.train() if split == 'train' else model.eval()
        totals = {'NLL': 0., 'words': 0}

        for itr, (_, dec_inputs, targets, lengths) in enumerate(dataloader):
            bsize = dec_inputs.size(0)
            dec_inputs = dec_inputs.to(device)
            targets = targets.to(device)
            lengths = lengths.to(device)

            model.forward(dec_inputs, lengths)
            rollout_x = model.rollout()
            target = targets[:, :torch.max(lengths+1).item()].contiguous().view(-1)
            NLL_loss = NLL(rollout_x, target)
            loss = NLL_loss / bsize

            # cumulate
            totals['NLL']+=NLL_loss.item()
            totals['words'] += torch.sum(lengths).item()

            if split == 'train':
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                tracker['NLL'].append(loss.item())

                # inner update
                loss = 0
                nll = 0
                over = 0

                #model.eval()
                for _ in range(small_epoch):
                    # forward
                    model(dec_inputs, lengths)
                    t_loss, t_nll, t_over = model.calculate_loss(targets)

                    t_loss = t_loss / bsize
                    t_nll = t_nll / bsize
                    t_over = t_over / bsize

                    optimizer.zero_grad()
                    t_loss.backward()
                    utils.clip_grad_norm_(model.parameters(), 1)
                    optimizer.step()

                    loss += t_loss
                    nll += t_nll
                    over += t_over

                loss/=small_epoch
                nll/=small_epoch
                over/=small_epoch

                if itr % print_every == 0 or itr + 1 == len(dataloader):
                    print("%s Batch %04d/%04d, NLL-Loss %.4f, model_loss %.4f, tdvae_nll %.4f, overshooting_nll %.4f"
                           %(split.upper(), itr, len(dataloader), tracker['NLL'][-1], loss, nll, over))
                    print("Epoch %d, %s Batch %04d/%04d, NLL-Loss %.4f, "
                           %(ep, split.upper(), itr, len(dataloader), tracker['NLL'][-1]),
                           file = log_file_handle2, flush = True)

        samples = len(datasets[split])

        if split == 'valid' and best_nll > (totals['NLL'] / samples):
            best_nll = totals['NLL'] / samples
            print("Best NLL: ", best_nll)

        elif split == 'valid' and learning_rate > 0.00001 and best_nll <= (totals['NLL'] / samples):
            learning_rate = learning_rate * 0.5
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate
                print("learning rate: ",learning_rate)
                print("Best NLL: ", best_nll)


        print("%s Epoch %02d/%02d, NLL %.4f, PPL %.4f"
               % (split.upper(), ep, epoch, totals['NLL'] / samples,
                     math.exp(totals['NLL'] / totals['words'])),
                     file = log_file_handle, flush = True)
        print("%s Epoch %02d/%02d, NLL %.4f, PPL %.4f"
               % (split.upper(), ep, epoch, totals['NLL'] / samples,
                     math.exp(totals['NLL'] / totals['words'])))

        model.eval()
        model.forward(dec_inputs, lengths)
        z = model.z_compare()
        # save checkpoint
        if split == 'train':
            train_tracker['NLL'].append(totals['NLL'] / samples)
            train_tracker['PPL'].append(math.exp(totals['NLL'] / totals['words']))
            train_tracker['z_space'].append(z)

            checkpoint_path = os.path.join(save_path, "E%02d.pkl" % ep)
            torch.save(model.state_dict(), checkpoint_path)
            print("Model saved at %s\n" % checkpoint_path)
        elif split == 'valid':
            valid_tracker['NLL'].append(totals['NLL'] / samples)
            valid_tracker['PPL'].append(math.exp(totals['NLL'] / totals['words']))
            valid_tracker['z_space'].append(z)
        else:
            test_tracker['NLL'].append(totals['NLL'] / samples)
            test_tracker['PPL'].append(math.exp(totals['NLL'] / totals['words']))
            test_tracker['z_space'].append(z)

end_time = time.time()
print('Total cost time',
      time.strftime("%H hr %M min %S sec", time.gmtime(end_time - start_time)))

save_pickle(train_tracker, save_path+"/train_tracker.pkl")
save_pickle(valid_tracker, save_path+"/valid_tracker.pkl")
save_pickle(test_tracker, save_path+"/test_tracker.pkl")
#save_pickle(lnt, save_path+"/test_lnt.pkl")


# save learning results
sio.savemat("results.mat", tracker)
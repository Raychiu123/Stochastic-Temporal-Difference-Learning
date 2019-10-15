import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from utils import *

time_step = 8

pth1 = "../output/model_tdvae_v4/MonSep30/"
pth2 = "time_step_"

pth_over1 = "with_overshooting"
pth_over2 = "without_overshooting"

path = pth1 + pth2 + str(time_step) + '/'
path = path + pth_over1

#train_tracker1 = load_pickle(path+"/train_tracker.pkl")
#valid_tracker1 = load_pickle(path+"/valid_tracker.pkl")
#test_tracker1 = load_pickle(path+"/test_tracker.pkl")

###############################################################
path = pth1 + pth2 + str(time_step) + '/'
path = path + pth_over2

#train_tracker2 = load_pickle(path+"/train_tracker.pkl")
#valid_tracker2 = load_pickle(path+"/valid_tracker.pkl")
test_tracker2 = load_pickle(path+"/test_tracker.pkl")

###############################################################

#plt.style.use('ggplot')
#z_train = train_tracker2['z_space'][19].view(-1, 16)
#z_valid = valid_tracker2['z_space'][19].view(-1, 16)
z_test = test_tracker2['z_space'][19].view(-1, 16)

#z_train_2d = TSNE(n_components=2, perplexity = 30 , random_state = 0).fit_transform(z_train.data.cpu().numpy())
#z_valid_2d = TSNE(n_components=2, perplexity = 30 , random_state = 0).fit_transform(z_valid.data.cpu().numpy())
print("TSNE...")
z_test_2d = TSNE(n_components=2, perplexity = 30 , random_state = 0).fit_transform(z_test.data.cpu().numpy())
print("TSNE done!!!")
#plt.scatter(z_train_2d[:,0], z_train_2d[:,1], c = 'C1', s = 3)
#plt.scatter(z_valid_2d[:,0], z_valid_2d[:,1], c = 'g', s = 1.5)
plt.scatter(z_test_2d[:,0], z_test_2d[:,1], c = 'C0', s = 2.5)

file = str(time_step) + '_lnt_scatter'
plt.savefig(file)

#print(train_tracker2['z_space'][19].shape)

import numpy as np
import time
import argparse
import pickle
import torch
import torch.optim as optim
import torch.nn.functional as F
from scipy.sparse import coo_matrix
from model import MDANet
from utils import get_logger
from utils import data_loader
from utils import multi_data_loader
import uproot, pandas
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument("-n", "--name", help="Name used to save the log file.", type=str, default="ttH")
parser.add_argument("-f", "--frac", help="Fraction of the supervised training data to be used.",
                    type=float, default=0.66)
parser.add_argument("-s", "--seed", help="Random seed.", type=int, default=42)
parser.add_argument("-v", "--verbose", help="Verbose mode: True -- show training progress. False -- "
                                            "not show training progress.", type=bool, default=True)
parser.add_argument("-m", "--model", help="Choose a model to train: [mdan]",
                    type=str, default="mdan")
# The experimental setting of using 48 dimensions of features is according to the papers in the literature.
parser.add_argument("-d", "--dimension", help="Number of features to be used in the experiment",
                    type=int, default=41)
parser.add_argument("-u", "--mu", help="Hyperparameter of the coefficient for the domain adversarial loss",
                    type=float, default=1.2)
parser.add_argument("-e", "--epoch", help="Number of training epochs", type=int, default=300)
parser.add_argument("-b", "--batch_size", help="Batch size during training", type=int, default=5000)
parser.add_argument("-o", "--mode", help="Mode of combination rule for MDANet: [maxmin|dynamic]", type=str, default="maxmin")
parser.add_argument("-l", "--hidden_layers", help="Number of neurons in hidden layers.", nargs='+', type=int, default=[45, 30, 25])
parser.add_argument("-dom", "--data_from", help="Data from domains:[data_src_vs_src1|data_trg_vs_trg1|data_src_vs_trg|data_trg_vs_src]", type=str, default='data_src_vs_trg') 
parser.add_argument("-dev", "--device_name", help="Device to use: [cuda|cpu].", type=str, default='cuda') 
parser.add_argument("-u_mode", "--mu_mode", help="Strategy for 'mu': [const|off_disc]", type=str, default='const')
parser.add_argument("-d_mode", "--d_mode", help="Strategy for discriminator, either pass bkg events from S1 to discriminator or all instances from S1: [bkg_only|all]", type=str, default='bkg_only')
args = parser.parse_args()

N = 1

train_val_loss_dict = pickle.load(open("../train_val_loss_dicts/train_val_loss_dict-{}-{}-{}-{}-epochs_{}-mu_{}-l_{}-data_from_{}.pkl".format(args.name, args.frac, args.model, args.mode, args.epoch, args.mu, args.hidden_layers, args.data_from), "rb"))
lw = 2
plt.rc('font', size=12)  
fig, ax1 = plt.subplots()
ax1.set_ylabel('NLL loss')
ax1.set_xlabel('Epochs')
# {'clf_losses': [], 'discr_losses' : [], 'total_loss_in_epoch': [], 'clf_losses_trg_val': [], 'significance', 'best_setting' : {'best_clf_loss_val': 1e10, 'accuracy': 0, 'epoch': 0} }
'''if args.mu == 0.0:
  for key, col in zip(['clf_losses', 'clf_losses_trg_val'], two_col):
    train_val_loss_dict[key] = np.convolve(train_val_loss_dict[key], np.ones((N,))/N, mode='valid')
    ax1.plot(train_val_loss_dict[key], label=key, color = col)
else:
  for key in ['clf_losses', 'discr_losses', 'clf_losses_trg_val']:
    train_val_loss_dict[key] = np.convolve(train_val_loss_dict[key], np.ones((N,))/N, mode='valid')
    ax1.plot(train_val_loss_dict[key], label=key)
'''

train_val_loss_dict['clf_losses'] = np.convolve(train_val_loss_dict['clf_losses'], np.ones((N,))/N, mode='valid')
ax1.plot(train_val_loss_dict['clf_losses'], label='clf_loss_S_1', color = 'C0', lw = lw, ls = '--')

if args.mu != 0.0:
  train_val_loss_dict['discr_losses'] = np.convolve(train_val_loss_dict['discr_losses'], np.ones((N,))/N, mode='valid')
  ax1.plot(train_val_loss_dict['discr_losses'], label='discr_loss', color = 'C1', lw = lw, ls = '-.')

train_val_loss_dict['clf_losses_trg_val'] = np.convolve(train_val_loss_dict['clf_losses_trg_val'], np.ones((N,))/N, mode='valid')
ax1.plot(train_val_loss_dict['clf_losses_trg_val'], label='clf_loss_S_2', color = 'C2', lw = lw, ls = ':')

ax2 = ax1.twinx()
ax2.set_ylabel('Significance', color='red')
train_val_loss_dict['significance'] = np.convolve(train_val_loss_dict['significance'], np.ones((N,))/N, mode='valid')
ax2.plot(train_val_loss_dict['significance'], label='significance', color='red', lw = lw)

ax1.legend()
ax1.set_ylim([0.5, 1.2])
ax1.set_xlim([0, 300])
plt.savefig("../../Plots/Training_plots/train_val_loss_plot-epochs_{}-mu_{}-l_{}-data_from_{}.png".format(args.epoch, args.mu, args.hidden_layers, args.data_from), dpi = 300)
plt.show()
"""
plt.plot(train_val_loss_dict['clf_losses'], label='clf_losses')
plt.legend()
plt.show()
"""

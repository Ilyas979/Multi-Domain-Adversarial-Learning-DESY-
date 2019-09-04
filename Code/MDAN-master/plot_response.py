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
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc

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

#args.mu = 1.2006896551724138

with open("../pred_scores/pred_scores-{}-{}-{}-{}-epochs_{}-mu_{}-l_{}-data_from_{}.pkl".format(args.name, args.frac, args.model, args.mode, args.epoch, args.mu, args.hidden_layers, args.data_from), "rb") as f:
  y_score_test = pickle.load(f)
  y_test = pickle.load(f)
  y_score_train = pickle.load(f)
  y_train = pickle.load(f)

#Test
signal_test = y_score_test[y_test==1]
bkg_test = y_score_test[y_test==0]
#Train
signal_train = y_score_train[y_train==1]
bkg_train = y_score_train[y_train==0]

lw = 2
plt.rc('font', size=12) 

bins = np.histogram(np.hstack((signal_test, bkg_test, signal_train, bkg_train)), bins=40)[1]
plt.hist(signal_test, bins, alpha=0.5, label='sgnl_S_2', density = True, color = '#1f77b4')
plt.hist(bkg_test, bins, alpha=0.5, label='bkg_S_2', density = True, color = '#ff7f0e')
plt.hist(signal_train, bins, alpha=1, label='sgnl_S_1', density = True, color = '#1f77b4', histtype = 'step', lw = lw)
plt.hist(bkg_train, bins, alpha=1, label='bkg_S_1', density = True, color = '#ff7f0e', histtype = 'step', lw = lw)

plt.xlim((0,1))
plt.legend(loc='upper right')
plt.title("Response_NN")
plt.xlabel('NN response')
plt.ylabel('dN/N')
plt.savefig("../../Plots/Response/Response-epochs_{}-mu_{}-l_{}-data_from_{}.png".format(args.epoch, args.mu, args.hidden_layers, args.data_from))
plt.show()
plt.close()



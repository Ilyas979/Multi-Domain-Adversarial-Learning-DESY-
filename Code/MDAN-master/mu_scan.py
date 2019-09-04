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
import os

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
                    type=int, default=44)
parser.add_argument("-u", "--mu", help="Hyperparameter of the coefficient for the domain adversarial loss",
                    type=float, default=0.5)
parser.add_argument("-e", "--epoch", help="Number of training epochs", type=int, default=300)
parser.add_argument("-b", "--batch_size", help="Batch size during training", type=int, default=5000)
parser.add_argument("-o", "--mode", help="Mode of combination rule for MDANet: [maxmin|dynamic]", type=str, default="maxmin")
parser.add_argument("-l", "--hidden_layers", help="Number of neurons in hidden layers.", nargs='+', type=int, default=[45, 30, 25])
parser.add_argument("-dom", "--data_from", help="Data from domains:[data_src_vs_src1|data_trg_vs_trg1|data_src_vs_trg]", type=str, default='data_src_vs_trg') 
parser.add_argument("-dev", "--device_name", help="Device to use: [cuda|cpu].", type=str, default='cuda') 
parser.add_argument("-d_mode", "--d_mode", help="Strategy for discriminator, either pass bkg events from S1 to discriminator or all instances from S1: [bkg_only|all]", type=str, default='bkg_only')
args = parser.parse_args()

#hid_lay = args.hidden_layers
epochs = args.epoch
#mu = args.mu 
#hidden_layers = args.hidden_layers
#assert(hidden_layers == [45, 30, 25])

#mu_range = [1.2006896551724138]
mu_range = [0.0]
a_range = [45]
b_range = [30]
c_range = [25]
#[a, b, c] in range 
for a in a_range:
  for b in b_range:
    for c in c_range:
      for mu in mu_range:
        os.system("python main_ttH.py -e {} -u {} -l {} {} {}".format(epochs, mu, a, b, c))
        os.system("python plot_train_val_loss.py -e {} -u {} -l {} {} {}".format(epochs, mu, a, b, c))
        os.system("python plot_response.py -e {} -u {} -l {} {} {}".format(epochs, mu, a, b, c))





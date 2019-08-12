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

train_val_loss_dict = pickle.load(open("train_val_loss_dict-ttH-0.66-mdan-maxmin-0.5.pkl", "rb"))
# {'clf_losses': [], 'discr_losses' : [], 'total_loss_in_epoch': [], 'clf_losses_val': []}
for key, values in train_val_loss_dict.items():
  #if key != 'clf_losses_val':  
  plt.plot(values, label=key)

plt.legend()
plt.savefig("../../Plots/train_val_loss_plot")
plt.show()
"""
plt.plot(train_val_loss_dict['clf_losses'], label='clf_losses')
plt.legend()
plt.show()
"""

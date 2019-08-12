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

with open("pred_scores-ttH-0.66-mdan-maxmin-0.5.pkl", "rb") as f:
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

bins = np.histogram(np.hstack((signal_test, bkg_test, signal_train, bkg_train)), bins=40)[1]
plt.hist(signal_test, bins, alpha=0.5, label='sgnl_test', density = True, color = '#1f77b4')
plt.hist(bkg_test, bins, alpha=0.5, label='bkg_test', density = True, color = '#ff7f0e')
plt.hist(signal_train, bins, alpha=1, label='sgnl_train', density = True, color = '#1f77b4', histtype = 'step')
plt.hist(bkg_train, bins, alpha=1, label='bkg_train', density = True, color = '#ff7f0e', histtype = 'step')

plt.xlim((0,1))
plt.legend(loc='upper right')
plt.title("Response_NN")
plt.xlabel('NN response')
plt.ylabel('dN/N')
plt.savefig("../../Plots/Response/Response_mu_0,5")
plt.show()
plt.close()



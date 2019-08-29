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
                    type=int, default=44)
parser.add_argument("-u", "--mu", help="Hyperparameter of the coefficient for the domain adversarial loss",
                    type=float, default=0.5)
parser.add_argument("-e", "--epoch", help="Number of training epochs", type=int, default=300)
parser.add_argument("-b", "--batch_size", help="Batch size during training", type=int, default=5000)
parser.add_argument("-o", "--mode", help="Mode of combination rule for MDANet: [maxmin|dynamic]", type=str, default="maxmin")
parser.add_argument("-l", "--hidden_layers", help="Number of neurons in hidden layers.", nargs='+', type=int, default=[45, 30, 25])
parser.add_argument("-dom", "--data_from", help="Data from domains:[data_src_vs_src1|data_trg_vs_trg1|data_src_vs_trg]", type=str, default='data_src_vs_trg') 
parser.add_argument("-dev", "--device_name", help="Device to use: [cuda|cpu].", type=str, default='cuda') 
args = parser.parse_args()

args.mu = 1.2006896551724138

with open("../pred_scores/pred_scores-{}-{}-{}-{}-epochs_{}-mu_{}-l_{}-data_from_{}.pkl".format(args.name, args.frac, args.model, args.mode, args.epoch, args.mu, args.hidden_layers, args.data_from), "rb") as f:
  y_score_test = pickle.load(f)
  y_test = pickle.load(f)
  y_score_train = pickle.load(f)
  y_train = pickle.load(f)
  train_source_pred_scores = pickle.load(f)
  train_source_labels = pickle.load(f)

with open("../pred_scores/pred_scores-{}-{}-{}-{}-epochs_{}-mu_{}-l_{}-data_from_{}.pkl".format(args.name, args.frac, args.model, args.mode, args.epoch, 0.0, args.hidden_layers, args.data_from), "rb") as f:
  y_score_test_0 = pickle.load(f)
  y_test_0 = pickle.load(f)
  y_score_train_0 = pickle.load(f)
  y_train_0 = pickle.load(f)
  train_source_pred_scores_0 = pickle.load(f)
  train_source_labels_0 = pickle.load(f)
# Compute ROC curve and ROC area for each class


fpr, tpr, _ = roc_curve(y_test, y_score_test)
roc_auc = auc(fpr, tpr)

fpr_0, tpr_0, _ = roc_curve(y_test_0, y_score_test_0)
roc_auc_0 = auc(fpr_0, tpr_0)

fpr_s, tpr_s, _ = roc_curve(train_source_labels, train_source_pred_scores)
roc_auc_s = auc(fpr_s, tpr_s)

fpr_s_0, tpr_s_0, _ = roc_curve(train_source_labels_0, train_source_pred_scores_0)
roc_auc_s_0 = auc(fpr_s_0, tpr_s_0)

plt.figure()
lw = 2

plt.plot(fpr, tpr, color='darkorange', lw=lw, label='mu: 1.2, AUC = %0.2f, S_2' % roc_auc)
plt.plot(fpr_0, tpr_0, color='black',ls='-.', lw=lw, label='mu: 0.0, AUC = %0.2f, S_2' % roc_auc_0)
plt.plot(fpr_s, tpr_s, color='blue', ls = '--', lw=lw, label='mu: 1.2, AUC = %0.2f, S_1' % roc_auc_s)
plt.plot(fpr_s_0, tpr_s_0, color='red', ls = 'dotted', lw=lw, label='mu: 0.0, AUC = %0.2f, S_1' % roc_auc_s_0)

#plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic for ttH(bb)')
plt.legend(loc="lower right")
plt.savefig("../../Plots/ROC/ROC-epochs_{}-mu_{}-l_{}-data_from_{}.png".format(args.epoch, args.mu, args.hidden_layers, args.data_from))
plt.show()

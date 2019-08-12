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

with open("pred_scores-ttH-0.66-mdan-maxmin-0.0.pkl", "rb") as f:
  y_score = pickle.load(f)
  y_test = pickle.load(f)

# Compute ROC curve and ROC area for each class


fpr, tpr, _ = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)

plt.figure()
lw = 1
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic for ttH')
plt.legend(loc="lower right")
plt.savefig("../../Plots/ROC")
plt.show()

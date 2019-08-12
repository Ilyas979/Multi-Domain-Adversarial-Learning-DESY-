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
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt

num_data_sets = 2
num_insts = []
data_from = 'data_src_vs_trg'
#options: 'data_src_vs_src1' data_trg_vs_trg1 data_src_vs_trg

with open('../../Data/'+data_from, 'rb') as f:
  data_insts = pickle.load(f)
  data_labels = pickle.load(f)
  data_name = pickle.load(f)

source_insts = data_insts[0].toarray()[:500]
source_labels = data_labels[0].flatten()[:500]

X = source_insts
y = source_labels.astype(int)

target_names = ['bkg', 'sgnl']

pca = PCA(n_components=2)
X_r = pca.fit(X).transform(X)

lda = LinearDiscriminantAnalysis(n_components=2)
X_r2 = lda.fit_transform(X, y)
#print(X.shape, y.shape, X_r2.shape)
# Percentage of variance explained for each components
print('explained variance ratio (first two components): %s'
      % str(pca.explained_variance_ratio_))

colors = ['red', 'b']
lw = 2

plt.figure()
for color, i, target_name in zip(colors, [0, 1], target_names):
    plt.scatter(X_r[y == i, 0], X_r[y == i, 1], color=color, alpha=.8, lw=lw,
                label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('PCA')
'''
plt.figure()
for color, i, target_name in zip(colors, [0, 1], target_names):
    plt.scatter(X_r2[y == i, 0], X_r2[y == i, 1], alpha=.8, color=color,
                label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('LDA')
'''
plt.show()


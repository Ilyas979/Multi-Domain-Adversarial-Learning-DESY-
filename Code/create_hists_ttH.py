import numpy as np
import time
import pickle

from scipy.sparse import coo_matrix
import uproot, pandas
import matplotlib.pyplot as plt

with open('../Data/data_src_vs_trg', 'rb') as f:
  data_insts = pickle.load(f)
  data_labels = pickle.load(f)
  data_name = pickle.load(f)

unwanted_features = ['weight_mc', 'weight_xsec', 'weight_total', 'OverlapRemove']
ttbar = uproot.open("../Data/pre_selected_tev13_mg5_ttbar_0001.root")["nominalAfterCuts"]
ttbar_df = ttbar.pandas.df().drop(columns = unwanted_features)

feature_names = ttbar_df.columns

for feat_name, i in zip(feature_names, range(len(feature_names))):
  for inst, label, src_name in zip(data_insts, data_labels, data_name):
    a = inst.toarray()[:,i][label.reshape(-1) == 1]
    b = inst.toarray()[:,i][label.reshape(-1) == 0]
    if src_name == 'src':
      bins = np.histogram(np.hstack((a,b)), bins=40)[1]
      plt.hist(a, bins, alpha=0.5, label='sgnl', density = True)
      plt.hist(b, bins, alpha=0.5, label='bkg', density = True)
      x_lim = plt.xlim()
      y_lim = plt.ylim()
      l = list(y_lim)
      l[1] *= 1.1
      y_lim = tuple(l)
    else:
      plt.hist(a, bins, alpha=0.5, label='sgnl', density = True)
      plt.hist(b, bins, alpha=0.5, label='bkg', density = True)
    plt.xlim(x_lim)
    plt.ylim(y_lim)
    plt.legend(loc='upper right')
    plt.title(feat_name + ", " + src_name)
    plt.savefig("../Plots/Histograms/"+feat_name+ "_" + src_name)
    plt.close()

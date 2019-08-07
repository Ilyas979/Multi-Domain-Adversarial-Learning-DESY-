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
    if src_name == 'src':
      b1 = inst.toarray()[:,i][label.reshape(-1) == 0]
      #plt.hist(a, bins, alpha=0.5, label='sgnl', density = True)
    elif src_name == 'target':
      a = inst.toarray()[:,i][label.reshape(-1) == 1]
      b2 = inst.toarray()[:,i][label.reshape(-1) == 0]
      bins = np.histogram(np.hstack((a,b1,b2)), bins=40)[1]
      plt.hist(a, bins, alpha=0.5, label='sgnl', density = True, color = '#1f77b4')
      plt.hist(b1, bins, alpha=0.5, label='bkg_1', density = True, color = '#ff7f0e')
      plt.hist(b2, bins, alpha=0.5, label='bkg_2', density = True, color = '#2ca02c')
      plt.legend(loc='upper right')
      plt.title(feat_name)
      plt.savefig("../Plots/Histograms/Bkgs_sgnl_together/"+feat_name)
      plt.close()

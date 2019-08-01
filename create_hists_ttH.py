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

for inst, label, src_name in zip(data_insts, data_labels, data_name):
  for feat_name, i in zip(feature_names, range(len(feature_names))):
    a = inst.toarray()[:,i][label.reshape(-1) == 1]
    b = inst.toarray()[:,i][label.reshape(-1) == 0]
    bins = np.histogram(np.hstack((a,b)), bins=40)[1]
    plt.hist(a, bins, alpha=0.5, label='sgnl', density = True)
    plt.hist(b, bins, alpha=0.5, label='bkg', density = True)
    plt.legend(loc='upper right')
    plt.title(feat_name + ", " + src_name)
    plt.savefig("../Plots/Histograms/"+feat_name+ "_" + src_name)
    plt.close()

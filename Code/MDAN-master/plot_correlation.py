import pickle
import uproot
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="ticks", color_codes=True)
plt.style.use('ggplot')
data_from = 'data_src_vs_trg'
#options: 'data_src_vs_src1' data_trg_vs_trg1 data_src_vs_trg

unwanted_features = ['weight_mc', 'weight_xsec', 'weight_total', 'OverlapRemove', 'Mbbbb_Max', 'HT_all']
highly_correlated_features = ['Mbj_MaxPt']
unwanted_features.extend(highly_correlated_features)

ttbar = uproot.open("../../Data/pre_selected_tev13_mg5_ttbar_0001.root")["nominalAfterCuts"]
ttbar_df = ttbar.pandas.df().drop(columns = unwanted_features)
feature_names = list(ttbar_df.columns)
if feature_names[1] == 'HT_all':
  feature_names[1] = 'HT_all_maybe_copy'

with open('../../Data/'+data_from, 'rb') as f:
    data_insts = pickle.load(f)
    data_labels = pickle.load(f)
    data_name = pickle.load(f)

source_insts = data_insts[0].toarray()[:1000]
source_labels = data_labels[0].flatten()[:1000]
print(source_insts.shape)
dic = {key: value for key, value in zip(range(1,1+len(feature_names)), feature_names)}
print(dic)
print(np.corrcoef(np.transpose(source_insts)))
X2 = np.transpose(source_insts)
corr = (100 * np.corrcoef(np.transpose(source_insts))).astype(int)
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10,10))
sns.heatmap(corr, ax=axes, xticklabels=range(1, source_insts.shape[1]+1),yticklabels=range(1,source_insts.shape[1]+1), annot=True, fmt = '.3g')
axes.set_xlabel('Features',fontsize=15)
axes.set_ylabel('Features',fontsize=15)
axes.set_title('Correlation',fontsize=20)
plt.show()

'''
for i, f_name_i in zip(range(len(feature_names)), feature_names):
  for j, f_name_j in zip(range(len(feature_names)), feature_names):
    if i > j:
      plt.scatter(source_insts[source_labels==1,i], source_insts[source_labels==1, j], alpha = 0.5, color = 'b', label = 'sgnl')
      plt.scatter(source_insts[source_labels==0,i], source_insts[source_labels==0, j], alpha = 0.5, color = 'red', label ='bkg1')
      plt.xlabel(f_name_i)
      plt.ylabel(f_name_j)
      plt.legend(loc='upper right')
      plt.title(f_name_i+" vs "+f_name_j)
      plt.savefig("../../Plots/Correlations/Bkg1/"+f_name_i+"_vs_"+f_name_j+"with_bkg1")
      plt.close()

'''

import uproot, pandas
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
from scipy.sparse import coo_matrix
from sklearn.preprocessing import StandardScaler

unwanted_features = ['weight_mc', 'weight_xsec', 'weight_total', 'OverlapRemove']

ttH = uproot.open("../Data/pre_selected_tev13_mg5_ttH_0001.root")["nominalAfterCuts"]
ttH_np = ttH.pandas.df().drop(columns = unwanted_features).values

ttbar = uproot.open("../Data/pre_selected_tev13_mg5_ttbar_0001.root")["nominalAfterCuts"]
ttbar_np = ttbar.pandas.df().drop(columns = unwanted_features).values

ttbb = uproot.open("../Data/pre_selected_tev13_ttbb_PP8delphes.root")["nominalAfterCuts"]
ttbb_np = ttbb.pandas.df().drop(columns = ['scalePDF'] + unwanted_features).values

print(ttH_np.shape, ttbar_np.shape, ttbb_np.shape)

#maybe suffle first
np.random.shuffle(ttH_np)
np.random.shuffle(ttbar_np)
np.random.shuffle(ttbb_np)
ratio = 4
N_events=90000//ratio*3

src = coo_matrix(StandardScaler().fit_transform(np.concatenate((ttH_np[:N_events], ttbar_np[:ratio*N_events])))).tocsc()
y_src = np.concatenate((np.ones(N_events),np.zeros(ratio*N_events)))

target = coo_matrix(StandardScaler().fit_transform(np.concatenate((ttH_np[N_events:2*N_events], ttbb_np[:ratio*N_events])))).tocsc()
y_target = np.concatenate((np.ones(N_events),np.zeros(ratio*N_events)))



data_insts, data_labels, num_insts = [], [], []
data_name = ['src', 'target']
num_data_sets = 2
data_insts.append(src)
data_labels.append(y_src)
num_insts.append(y_src.shape[0])

data_insts.append(target)
data_labels.append(y_target)
num_insts.append(y_target.shape[0])

for i in range(num_data_sets):
    r_order = np.arange(num_insts[i])
    np.random.shuffle(r_order)
    data_insts[i] = data_insts[i][r_order, :]
    data_labels[i] = data_labels[i].reshape((data_labels[i].shape[0], 1))
    data_labels[i] = data_labels[i][r_order, :]

with open('../Data/data_src_vs_trg', 'wb') as f:
    pickle.dump(data_insts, f)
    pickle.dump(data_labels, f)
    pickle.dump(data_name, f)

print("Data is saved to 'data_src_vs_trg' in pickle format")
############
src1 = coo_matrix(StandardScaler().fit_transform(np.concatenate((ttH_np[N_events:2*N_events], ttbar_np[ratio*N_events:2*ratio*N_events])))).tocsc()
y_src1 = np.concatenate((np.ones(N_events),np.zeros(ratio*N_events)))

data_insts, data_labels, num_insts = [], [], []
data_name = ['src', 'src1']
num_data_sets = 2

data_insts.append(src)
data_labels.append(y_src)
num_insts.append(y_src.shape[0])

data_insts.append(src1)
data_labels.append(y_src1)
num_insts.append(y_src1.shape[0])

for i in range(num_data_sets):
    r_order = np.arange(num_insts[i])
    np.random.shuffle(r_order)
    data_insts[i] = data_insts[i][r_order, :]
    data_labels[i] = data_labels[i].reshape((data_labels[i].shape[0], 1))
    data_labels[i] = data_labels[i][r_order, :]

with open('../Data/data_src_vs_src1', 'wb') as f:
    pickle.dump(data_insts, f)
    pickle.dump(data_labels, f)
    pickle.dump(data_name, f)

print("Data is saved to 'data_src_vs_src1' in pickle format")

############
target1 = coo_matrix(StandardScaler().fit_transform(np.concatenate((ttH_np[:N_events], ttbb_np[ratio*N_events:2*ratio*N_events])))).tocsc()
y_target1 = np.concatenate((np.ones(N_events),np.zeros(ratio*N_events)))

data_insts, data_labels, num_insts = [], [], []
data_name = ['target', 'target1']
num_data_sets = 2

data_insts.append(target)
data_labels.append(y_target)
num_insts.append(y_target.shape[0])

data_insts.append(target1)
data_labels.append(y_target1)
num_insts.append(y_target1.shape[0])

for i in range(num_data_sets):
    r_order = np.arange(num_insts[i])
    np.random.shuffle(r_order)
    data_insts[i] = data_insts[i][r_order, :]
    data_labels[i] = data_labels[i].reshape((data_labels[i].shape[0], 1))
    data_labels[i] = data_labels[i][r_order, :]

with open('../Data/data_trg_vs_trg1', 'wb') as f:
    pickle.dump(data_insts, f)
    pickle.dump(data_labels, f)
    pickle.dump(data_name, f)

print("Data is saved to 'data_trg_vs_trg1' in pickle format")


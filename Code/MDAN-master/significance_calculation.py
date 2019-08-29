import numpy as np
import time
import argparse
import pickle
from scipy.sparse import coo_matrix
import uproot, pandas
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
import csv
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
                    type=float, default=1.0)
parser.add_argument("-e", "--epoch", help="Number of training epochs", type=int, default=300)
parser.add_argument("-b", "--batch_size", help="Batch size during training", type=int, default=2000)
parser.add_argument("-o", "--mode", help="Mode of combination rule for MDANet: [maxmin|dynamic]", type=str, default="maxmin")
parser.add_argument("-l", "--hidden_layers", help="Number of neurons in hidden layers.", nargs='+', type=int, default=[35, 30, 15])
parser.add_argument("-dom", "--data_from", help="Data from domains:[data_src_vs_src1|data_trg_vs_trg1|data_src_vs_trg]", type=str, default='data_src_vs_trg') 
parser.add_argument("-dev", "--device_name", help="Device to use: [cuda|cpu].", type=str, default='cuda') 
parser.add_argument("-c", "--cut", help="Cut in the response plot to calculate the significance: between 0.0 and 1.0", type=float, default=0.6)
args = parser.parse_args()

'''mus = [0.0,
1.3,
0.0,
1.2448275862068967,
0.0,
1.2779310344827586,
0.0,
1.2558620689655173,
0.0,
1.1786206896551725,
0.0,
1.2006896551724138]

hid_lays = [
[35, 30, 15],
[35, 30, 15],
[40, 25, 25],
[40, 25, 25],
[45, 25, 20],
[45, 25, 20],
[45, 25, 25],
[45, 25, 25],
[45, 30, 15],
[45, 30, 15],
[45, 30, 25],
[45, 30, 25]
]
'''
mus = [1.1786206896551725]
hid_lays = [[45, 30, 15]]
for mu, l in zip(mus, hid_lays):
  with open("./pred_scores/pred_scores-{}-{}-{}-{}-epochs_{}-mu_{}-l_{}-data_from_{}.pkl".format(args.name, args.frac, args.model, args.mode, args.epoch, mu, l, args.data_from), "rb") as f:
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

  #Test
  signal_test_cut = y_score_test[np.logical_and(y_test==1, y_score_test > args.cut)] 
  s = signal_test_cut.size / signal_test.size * 0.05 * 50e3
  bkg_test_cut = y_score_test[np.logical_and(y_test==0, y_score_test > args.cut)] 
  b_test = bkg_test_cut.size / bkg_test.size * 0.95 * 50e3
  #Train
  signal_train_cut = y_score_train[np.logical_and(y_train==1, y_score_train > args.cut)]
  bkg_train_cut = y_score_train[np.logical_and(y_train==0, y_score_train > args.cut)] 
  b_train = bkg_train_cut.size / bkg_train.size * 0.95 * 50e3

  sigma = b_test - b_train
  Z_A = s/np.sqrt(b_test+sigma**2)
  print("s = {}, b = {}, sigma_2 = {}, Significance = {}".format(s, b_test, sigma**2, Z_A))
  with open('../Results/mu_significance.csv','a+') as csv_f:
      fieldnames = ['mu', 'hidden_layers', 'significance', 's', 'b', '\sigma^2']
      writer = csv.DictWriter(csv_f, fieldnames=fieldnames)
      writer.writerow({"mu": mu, 'hidden_layers': l, 'significance': Z_A, 's': s, 'b': b_test, '\sigma^2': sigma**2})


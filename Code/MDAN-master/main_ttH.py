#!/usr/bin/env python
# -*- coding: utf-8 -*-

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
import csv
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

def save_best_setting(old_dic, curr_dic):
  if old_dic['best_clf_loss_val'] < curr_dic['best_clf_loss_val']:
    return curr_dic
  else:
    return curr_dic

def calculate_and_save_significance(y_score_src, y_src, y_score_trg, y_trg, cut = 0.6):
  
  #Source
  signal_src = y_score_src[y_src==1]
  bkg_src = y_score_src[y_src==0]
  #Target
  signal_trg = y_score_trg[y_trg==1]
  bkg_trg = y_score_trg[y_trg==0]
  
  #Source
  signal_src_cut = signal_src[signal_src > cut] 
  bkg_src_cut = bkg_src[bkg_src > cut] 
  b_src = bkg_src_cut.size / bkg_src.size * 0.95 * 50e3
  
  #Target
  signal_trg_cut = signal_trg[signal_trg > cut]
  s = signal_trg_cut.size / signal_trg.size * 0.05 * 50e3
  bkg_trg_cut = bkg_trg[bkg_trg > cut] 
  b_trg = bkg_trg_cut.size / bkg_trg.size * 0.95 * 50e3

  sigma = b_src - b_trg
  if b_trg+sigma**2 != 0.0:
    Z_A = s/np.sqrt(b_trg+sigma**2)
  else:
    Z_A = 0
  return Z_A
  #print("s = {}, b = {}, sigma_2 = {}, Significance = {}".format(s, b_trg, sigma**2, Z_A))
  '''
  with open('../significance_in_training/significance_in_training.csv','a+') as csv_f:
      fieldnames = ['significance', 's', 'b', '\sigma^2']
      writer = csv.DictWriter(csv_f, fieldnames=fieldnames)
      writer.writerow({'significance': Z_A, 's': s, 'b': b_test, '\sigma^2': sigma**2})
  '''

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
                    type=int, default=41)
parser.add_argument("-u", "--mu", help="Hyperparameter of the coefficient for the domain adversarial loss",
                    type=float, default=1.2)
parser.add_argument("-e", "--epoch", help="Number of training epochs", type=int, default=300)
parser.add_argument("-b", "--batch_size", help="Batch size during training", type=int, default=5000)
parser.add_argument("-o", "--mode", help="Mode of combination rule for MDANet: [maxmin|dynamic]", type=str, default="maxmin")
parser.add_argument("-l", "--hidden_layers", help="Number of neurons in hidden layers.", nargs='+', type=int, default=[45, 30, 25])
parser.add_argument("-dom", "--data_from", help="Data from domains:[data_src_vs_src1|data_trg_vs_trg1|data_src_vs_trg|data_trg_vs_src]", type=str, default='data_src_vs_trg') 
parser.add_argument("-dev", "--device_name", help="Device to use: [cuda|cpu].", type=str, default='cuda') 
parser.add_argument("-u_mode", "--mu_mode", help="Strategy for 'mu': [const|off_disc]", type=str, default='const') #doesnt work
parser.add_argument("-d_mode", "--d_mode", help="Strategy for discriminator, either pass bkg events from S1 to discriminator or all instances from S1: [bkg_only|all]", type=str, default='bkg_only')
parser.add_argument("-swap_dom", "--swap_domains", help="If Source and Target domains should be swapped: [True|False]", type=bool, default=False)
parser.add_argument("-opt", "--opt", help="Choose which optimizer to use: [Adadelta|Adagrad|Nesterov|]", type=str, default='Adadelta')
parser.add_argument("-lr", "--lr", help="Learning rate for optimizer", type=float, default=0.1)
# Compile and configure all the model parameters.
args = parser.parse_args()

torch.cuda.empty_cache()
device = torch.device(args.device_name if torch.cuda.is_available() and args.device_name == 'cuda' else 'cpu')
#device2 = torch.device("cpu")

logger = get_logger(args.name)

# Set random number seed.
np.random.seed(args.seed)
torch.manual_seed(args.seed)


# Partition the data into three categories and for each category partition the data set into training and test set.
#maybe suffle first
num_data_sets = 2
num_insts = []
data_from = args.data_from

if num_data_sets == 3:
  pass
elif num_data_sets == 2:
  with open('../../Data/'+data_from, 'rb') as f:
    data_insts = pickle.load(f)
    data_labels = pickle.load(f)
    data_name = pickle.load(f)
  num_insts.append(data_labels[0].shape[0])
  num_insts.append(data_labels[1].shape[0])

###Training on target and testing on source
if args.swap_domains:
   data_insts = list(reversed(data_insts))
   data_labels = list(reversed(data_labels)) 
   data_name = list(reversed(data_name)) 
   num_insts = list(reversed(num_insts)) 


logger.info("Data sets: {}".format(data_name))
logger.info("Number of total instances in the data sets: {}".format(num_insts))
# Partition the data set into training and test parts, following the convention in the ICML-2012 paper, use a fixed
# amount of instances as training and the rest as test.
num_trains = int(num_insts[0] * args.frac)
input_dim = data_insts[0].shape[1]
# The confusion matrix stores the prediction accuracy between the source and the target tasks. The row index the source
# task and the column index the target task.
results = {}
logger.info("Training fraction = {}, number of actual training data instances = {}".format(args.frac, num_trains))
logger.info("-" * 100)

if args.model == "mdan":
    configs = {"input_dim": input_dim, "hidden_layers": args.hidden_layers, "num_classes": 2,
               "num_epochs": args.epoch, "batch_size": args.batch_size, "lr": args.lr, "mu": args.mu, "num_domains":
                   num_data_sets - 1, "mode": args.mode, "gamma": 10.0, "verbose": args.verbose}
    num_epochs = configs["num_epochs"]
    batch_size = configs["batch_size"]
    num_domains = configs["num_domains"]
    lr = configs["lr"]
    mu = configs["mu"]
    gamma = configs["gamma"]
    mode = configs["mode"]
    logger.info("Training with domain adaptation using PyTorch madnNet: ")
    logger.info("Hyperparameter setting = {}.".format(configs))
    error_dicts = {}
    train_val_loss_dict = {'clf_losses': [], 'discr_losses' : [], 'total_loss_in_epoch': [], 'clf_losses_trg_val': [], 'clf_losses_src_val': [], 'significance': []}
    for i in range(num_data_sets):
        if i == 0: continue
        # Build source instances.
        source_insts = []
        source_labels = []
        for j in range(num_data_sets):
            if j != i:
                source_insts.append(data_insts[j][:num_trains, :].todense().astype(np.float32))
                source_labels.append(data_labels[j][:num_trains, :].ravel().astype(np.int64))
        #Build source instances for testing
        test_source_insts_cpu = []
        test_source_labels_cpu = []
        for j in range(num_data_sets):
          if j != i:
            test_source_insts_cpu.append(data_insts[j][num_trains:, :].todense().astype(np.float32))
            test_source_labels_cpu.append(data_labels[j][num_trains:, :].ravel().astype(np.int64))
        # Build target instances.
        target_idx = i
        target_insts = data_insts[i][:num_trains, :].todense().astype(np.float32)
        target_labels = data_labels[i][:num_trains, :].ravel().astype(np.int64)

        test_target_insts_cpu = data_insts[i][num_trains:, :].todense().astype(np.float32)
        test_target_labels_cpu = data_labels[i][num_trains:, :].ravel().astype(np.int64)
        
        # Train DannNet.
        mdan = MDANet(configs).to(device)
        if args.opt == 'Adadelta':
          optimizer = optim.Adadelta(mdan.parameters(), lr=args.lr) #works
        elif args.opt == 'Adagrad':
          optimizer = optim.Adagrad(mdan.parameters(), lr=args.lr) #works
        else:
          optimizer = optim.SGD(mdan.parameters(), lr=args.lr, momentum=0.9) #doesn't work, even momentum doesn't seem to help
        mdan.eval()
        with torch.no_grad():
          test_target_insts = torch.tensor(test_target_insts_cpu, requires_grad=False).to(device)
          test_target_labels = torch.tensor(test_target_labels_cpu).to(device)
          preds_labels = torch.max(mdan.inference(test_target_insts), 1)[1]
          pred_acc = torch.sum(preds_labels == test_target_labels).item() / float(test_target_insts.size(0))
          print("accuracy before training: ", pred_acc)
          del test_target_insts, test_target_labels, preds_labels, pred_acc
          torch.cuda.empty_cache()

        mdan.train()
        # Training phase.
        time_start = time.time()
        for t in range(num_epochs):
            running_loss, sum_in_epoch_losses, sum_in_epoch_domain_losses = 0.0, 0.0, 0.0
            train_loader = multi_data_loader(source_insts, source_labels, batch_size)
            train_loader_size = 0
            for xs, ys in train_loader:
                #These 'tlabels' are labels of being in a particular source. One means that it is source, Zero that it is target.
                if args.d_mode == "bkg_only":
                  bkg_in_batch_size = list(ys[0]).count(0)
                else:
                  bkg_in_batch_size = batch_size
                slabels = torch.ones(bkg_in_batch_size, requires_grad=False).type(torch.LongTensor).to(device)
                tlabels = torch.zeros(bkg_in_batch_size, requires_grad=False).type(torch.LongTensor).to(device)
                #These 'ys' are labels of being in a particular class, i.e. signal or background
                for j in range(num_domains):
                    xs[j] = torch.tensor(xs[j], requires_grad=False).to(device)
                    ys[j] = torch.tensor(ys[j], requires_grad=False).to(device)
                #tinputs = target_insts[ridx, :]
                ridx = np.random.choice(target_insts.shape[0], bkg_in_batch_size)
                tinputs = target_insts[ridx, :]
                tinputs = torch.tensor(tinputs, requires_grad=False).to(device)
                optimizer.zero_grad()
                logprobs, sdomains, tdomains = mdan(xs, tinputs, ys, args.d_mode) #this line only evals probs of being a signal and belonging to a particular source given a current weights of NN
                # Compute prediction accuracy on multiple training sources.
                losses = torch.stack([F.nll_loss(logprobs[j], ys[j]) for j in range(num_domains)])
                domain_losses = torch.stack([F.nll_loss(sdomains[j], slabels) +
                                           F.nll_loss(tdomains[j], tlabels) for j in range(num_domains)])
                # Different final loss function depending on different training modes.
                if mode == "maxmin":
                    loss = torch.max(losses) + 0.5 * mu * torch.min(domain_losses)
                    #what it does actually is: loss = losses[0] + mu * domain_losses[0]
                elif mode == "dynamic":
                    loss = torch.log(torch.sum(torch.exp(gamma * (losses + 0.5 *  mu * domain_losses)))) / gamma
                else:
                    raise ValueError("No support for the training mode on madnNet: {}.".format(mode))
                running_loss += loss.item()
                sum_in_epoch_losses += losses.item()
                sum_in_epoch_domain_losses += domain_losses.item()
                train_loader_size += 1
                loss.backward()
                optimizer.step()

            train_val_loss_dict['clf_losses'].append(sum_in_epoch_losses / train_loader_size)
            train_val_loss_dict['discr_losses'].append(sum_in_epoch_domain_losses / train_loader_size / 2.0)
            train_val_loss_dict['total_loss_in_epoch'].append(running_loss / train_loader_size)
            
            # Evaluate the loss on target domain (on validation data)
            mdan.eval()
            with torch.no_grad():
              test_target_insts = torch.tensor(test_target_insts_cpu, requires_grad=False).to(device)
              test_target_labels = torch.tensor(test_target_labels_cpu).to(device)
              test_target_logprobs = mdan.inference(test_target_insts)
              test_target_pred_scores = torch.exp(test_target_logprobs).to(device)[:,1].detach().cpu().numpy()
              train_val_loss_dict['clf_losses_trg_val'].append(F.nll_loss(test_target_logprobs, test_target_labels).item())
              logger.info("Iteration {}, loss = {}, clf_losses_trg_val = {}".format(t, running_loss / train_loader_size, train_val_loss_dict['clf_losses_trg_val'][-1]))
            # Evaluate the loss on source domain (on validation data)
            with torch.no_grad():
              test_source_insts = torch.tensor(test_source_insts_cpu[0], requires_grad=False).to(device)
              test_source_labels = torch.tensor(test_source_labels_cpu[0]).to(device)
              test_source_logprobs = mdan.inference(test_source_insts)
              test_source_pred_scores = torch.exp(test_source_logprobs).to(device)[:,1].detach().cpu().numpy()
              #train_val_loss_dict['clf_losses_src_val'].append(F.nll_loss(test_source_logprobs, test_source_labels).item())
              #logger.info("Iteration {}, loss = {}, clf_losses_val = {}".format(t, running_loss / train_loader_size, train_val_loss_dict['clf_losses_src_val'][-1]))
              train_val_loss_dict['significance'].append( calculate_and_save_significance(test_source_pred_scores, test_source_labels.detach().cpu().numpy(), test_target_pred_scores, test_target_labels.detach().cpu().numpy()) )
              del tinputs, slabels, tlabels, logprobs, sdomains, tdomains, loss, losses, domain_losses, xs, ys, test_target_insts, test_target_labels, test_target_logprobs, test_source_insts, test_source_labels, test_source_logprobs
              torch.cuda.empty_cache()

            mdan.train() 
          
        time_end = time.time()
        mdan.eval()
        del optimizer
        torch.cuda.empty_cache()
        # Validation on test !source! samples
        with torch.no_grad():
          train_source_insts = torch.tensor(source_insts[0], requires_grad=False).to(device)
          train_source_labels = torch.tensor(source_labels[0]).to(device)
          train_source_pred_scores = torch.exp(mdan.inference(train_source_insts)).to(device)[:,1]
          
          test_source_insts = torch.tensor(test_source_insts_cpu[0], requires_grad=False).to(device)
          test_source_labels = torch.tensor(test_source_labels_cpu[0]).to(device)
          test_source_pred_scores = torch.exp(mdan.inference(test_source_insts)).to(device)[:,1]
        # Validation on test !target! samples. 
          best_epoch = args.epoch-1  
          test_target_insts = torch.tensor(test_target_insts_cpu, requires_grad=False).to(device)
          test_target_labels = torch.tensor(test_target_labels_cpu).to(device)
          preds_labels = torch.max(mdan.inference(test_target_insts), 1)[1]
          pred_scores = torch.exp(mdan.inference(test_target_insts)).to(device)[:,1]
          pred_acc = torch.sum(preds_labels == test_target_labels).item() / float(test_target_insts.size(0))
          logger.info("Prediction accuracy on {} = {}, time used = {} seconds.".
                      format(data_name[i], pred_acc, time_end - time_start))
          results[data_name[i]] = pred_acc
          del mdan, test_source_insts
          torch.cuda.empty_cache()
        #AUCROC
        fpr, tpr, _ = roc_curve(test_target_labels.detach().cpu().numpy(), pred_scores.detach().cpu().numpy())
        auc_roc = auc(fpr, tpr)

    logger.info("Prediction accuracy with multiple source domain adaptation using madnNet: ")
    logger.info(results)

    pickle.dump(train_val_loss_dict, open("../train_val_loss_dicts/train_val_loss_dict-{}-{}-{}-{}-epochs_{}-mu_{}-l_{}-data_from_{}-opt_{}.pkl".format(args.name, args.frac, args.model, args.mode, args.epoch, args.mu, args.hidden_layers, data_from, args.opt), "wb"))
    with open("../pred_scores/pred_scores-{}-{}-{}-{}-epochs_{}-mu_{}-l_{}-data_from_{}-opt_{}.pkl".format(args.name, args.frac, args.model, args.mode, args.epoch, args.mu, args.hidden_layers, data_from, args.opt), "wb") as file_pred_scores:
      pickle.dump(pred_scores.detach().cpu().numpy(), file_pred_scores)
      pickle.dump(test_target_labels.detach().cpu().numpy(), file_pred_scores)
      pickle.dump(test_source_pred_scores.detach().cpu().numpy(), file_pred_scores)
      pickle.dump(test_source_labels.detach().cpu().numpy(), file_pred_scores)
      pickle.dump(train_source_pred_scores.detach().cpu().numpy(), file_pred_scores)
      pickle.dump(train_source_labels.detach().cpu().numpy(), file_pred_scores)
    logger.info("*" * 100)

    with open('../../Results/mu_accuracy.csv','a') as csv_f:
      fieldnames = ['mu','accuracy', 'auc_roc', 'stopped_epoch', 'epochs', 'hidden_layers', 'lr', 'data_from', 'significance', 'opt']
      writer = csv.DictWriter(csv_f, fieldnames=fieldnames)
      writer.writerow({"mu": args.mu, "accuracy": pred_acc, "auc_roc": auc_roc, "stopped_epoch": best_epoch+1,"epochs": args.epoch, "hidden_layers": configs["hidden_layers"], "lr": lr, "data_from": data_from, "significance": train_val_loss_dict['significance'][-1], "opt" : args.opt})
    del test_target_labels, test_target_insts, test_source_labels, test_source_pred_scores, pred_scores, target_insts, target_labels, pred_acc, preds_labels
    torch.cuda.empty_cache()
else:
    raise ValueError("No support for the following model: {}.".format(args.model))


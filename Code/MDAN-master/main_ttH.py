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

def save_best_setting(old_dic, curr_dic):
  if old_dic['best_clf_loss_val'] < curr_dic['best_clf_loss_val']:
    return curr_dic
  else:
    return curr_dic


parser = argparse.ArgumentParser()
parser.add_argument("-n", "--name", help="Name used to save the log file.", type=str, default="ttH")
parser.add_argument("-f", "--frac", help="Fraction of the supervised training data to be used.",
                    type=float, default=0.66)
parser.add_argument("-s", "--seed", help="Random seed.", type=int, default=42)
parser.add_argument("-v", "--verbose", help="Verbose mode: True -- show training progress. False -- "
                                            "not show training progress.", type=bool, default=True)
parser.add_argument("-m", "--model", help="Choose a model to train: [mdan]",
                    type=str, default="mdan")
# The experimental setting of using 40 dimensions of features is according to the papers in the literature.
parser.add_argument("-d", "--dimension", help="Number of features to be used in the experiment",
                    type=int, default=40)
parser.add_argument("-u", "--mu", help="Hyperparameter of the coefficient for the domain adversarial loss",
                    type=float, default=0.5)
parser.add_argument("-e", "--epoch", help="Number of training epochs", type=int, default=15)
parser.add_argument("-b", "--batch_size", help="Batch size during training", type=int, default=1000)
parser.add_argument("-o", "--mode", help="Mode of combination rule for MDANet: [maxmin|dynamic]", type=str, default="maxmin")
parser.add_argument("-l", "--hidden_layers", help="Number of neurons in hidden layers.", nargs='+', type=int, default=[30, 10, 2])
# Compile and configure all the model parameters.
args = parser.parse_args()

torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

device = torch.device("cpu")

logger = get_logger(args.name)

# Set random number seed.
np.random.seed(args.seed)
torch.manual_seed(args.seed)


# Partition the data into three categories and for each category partition the data set into training and test set.
#maybe suffle first
num_data_sets = 2
num_insts = []
data_from = 'data_src_vs_trg'
#options: 'data_src_vs_src1' data_trg_vs_trg1 data_src_vs_trg

if num_data_sets == 3:
  pass
elif num_data_sets == 2:
  with open('../../Data/'+data_from, 'rb') as f:
    data_insts = pickle.load(f)
    data_labels = pickle.load(f)
    data_name = pickle.load(f)
  num_insts.append(data_labels[0].shape[0])
  num_insts.append(data_labels[1].shape[0])

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
               "num_epochs": args.epoch, "batch_size": args.batch_size, "lr": 1e-1, "mu": args.mu, "num_domains":
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
    train_val_loss_dict = {'clf_losses': [], 'discr_losses' : [], 'total_loss_in_epoch': [], 'clf_losses_val': [], 'best_setting' : {'best_clf_loss_val': 1e10, 'accuracy': 0, 'epoch': 0, 'target_logprob': [], 'train_pred_score': []} }
    for i in range(num_data_sets):
        if i == 0: continue
        #train_val_loss_dict = {'clf_losses': [], 'discr_losses' : [], 'total_loss_in_epoch': [], 'clf_losses_val': []}
        # Build source instances.
        source_insts = []
        source_labels = []
        for j in range(num_data_sets):
            if j != i:
                source_insts.append(data_insts[j][:num_trains, :].todense().astype(np.float32))
                source_labels.append(data_labels[j][:num_trains, :].ravel().astype(np.int64))
        # Build target instances.
        target_idx = i
        target_insts = data_insts[i][:num_trains, :].todense().astype(np.float32)
        target_labels = data_labels[i][:num_trains, :].ravel().astype(np.int64)

        test_target_insts = data_insts[i][num_trains:, :].todense().astype(np.float32)
        test_target_labels = data_labels[i][num_trains:, :].ravel().astype(np.int64)
        test_target_insts = torch.tensor(test_target_insts, requires_grad=False).to(device)
        test_target_labels = torch.tensor(test_target_labels).to(device)
        # Train DannNet.
        mdan = MDANet(configs).to(device)
        optimizer = optim.Adadelta(mdan.parameters(), lr=lr)
        #optimizer = optim.Adamax(mdan.parameters())

        mdan.eval()
        
        #target_logprobs = mdan.inference(test_target_insts)
       
        #train_val_loss_dict['clf_losses_val'].append(F.nll_loss(target_logprobs, test_target_labels))

        #target_insts = torch.tensor(target_insts, requires_grad=False).to(device)
        #target_labels = torch.tensor(target_labels)
        #preds_labels = torch.max(mdan.inference(target_insts), 1)[1].cpu().data.squeeze_()
        #pred_acc = torch.sum(preds_labels == target_labels).item() / float(target_insts.size(0))
        #target_insts = target_insts.cpu().numpy()
        #target_labels = target_labels.cpu().numpy()
        #print("accuracy before training: ", pred_acc)
        
        target_insts_from_bkg = target_insts[target_labels == 0,:]
        
        train_source_insts = torch.tensor(source_insts[0], requires_grad=False).to(device)
        train_source_labels = torch.tensor(source_labels[0]).to(device)

        mdan.train()
        # Training phase.
        time_start = time.time()
        for t in range(num_epochs):
            running_loss, sum_in_epoch_losses, sum_in_epoch_domain_losses = 0.0, 0.0, 0.0
            train_loader = multi_data_loader(source_insts, source_labels, batch_size)
            train_loader_size = 0
            for xs, ys in train_loader:
                #These 'tlabels' are labels of being in a particular source. One's means that it is source, Zeros that it is target.
                bkg_in_batch_size = list(ys[0]).count(0)
                #print("bkg in source!!!: ", bkg_in_batch_size)
                slabels = torch.ones(bkg_in_batch_size, requires_grad=False).type(torch.LongTensor).to(device)
                tlabels = torch.zeros(bkg_in_batch_size, requires_grad=False).type(torch.LongTensor).to(device)
                #These 'ys' are labels of being in a particular class, i.e. signal or background
                for j in range(num_domains):
                    xs[j] = torch.tensor(xs[j], requires_grad=False).to(device)
                    ys[j] = torch.tensor(ys[j], requires_grad=False).to(device)
                #tinputs = target_insts[ridx, :]
                ridx = np.random.choice(target_insts_from_bkg.shape[0], bkg_in_batch_size)
                tinputs = target_insts_from_bkg[ridx, :]
                tinputs = torch.tensor(tinputs, requires_grad=False).to(device)
                optimizer.zero_grad()
                logprobs, sdomains, tdomains = mdan(xs, tinputs, ys) #this line only evals probs of being a signal and belonging to a particular source given a current weights of NN
                # Compute prediction accuracy on multiple training sources.
                losses = torch.stack([F.nll_loss(logprobs[j], ys[j]) for j in range(num_domains)]) 
                #print('train_loss: ', losses)
                domain_losses = torch.stack([F.nll_loss(sdomains[j], slabels) +
                                           F.nll_loss(tdomains[j], tlabels) for j in range(num_domains)])
                # Different final loss function depending on different training modes.
                if mode == "maxmin":
                    loss = torch.max(losses) +  mu * torch.min(domain_losses)
                    #what it does actually is: loss = losses[0] + mu * domain_losses[0]
                elif mode == "dynamic":
                    loss = torch.log(torch.sum(torch.exp(gamma * (losses + mu * domain_losses)))) / gamma
                else:
                    raise ValueError("No support for the training mode on madnNet: {}.".format(mode))
                running_loss += loss.item()
                sum_in_epoch_losses += losses.item()
                sum_in_epoch_domain_losses += domain_losses.item()
                train_loader_size += 1
                loss.backward()
                optimizer.step()

            train_val_loss_dict['clf_losses'].append(sum_in_epoch_losses / train_loader_size)
            train_val_loss_dict['discr_losses'].append(mu*sum_in_epoch_domain_losses / train_loader_size)
            train_val_loss_dict['total_loss_in_epoch'].append(running_loss / train_loader_size)
            
            # Evaluate the loss on target domain
            mdan.eval()
            target_logprobs = mdan.inference(test_target_insts)
            train_val_loss_dict['clf_losses_val'].append(F.nll_loss(target_logprobs, test_target_labels).item())
            # Validation on train and test target samples
            preds_labels = torch.max(target_logprobs, 1)[1]

            train_source_insts = torch.tensor(source_insts[0], requires_grad=False).to(device)
            train_pred_scores = torch.exp(mdan.inference(train_source_insts)).to(device)[:,1]


            current_setting = {'best_clf_loss_val': train_val_loss_dict['clf_losses_val'][-1], 'accuracy' : torch.sum(preds_labels == test_target_labels).item() / float(test_target_insts.size(0)) , 'epoch': t, 'target_logprob': target_logprobs, 'train_pred_score': train_pred_scores}
            
            train_val_loss_dict['best_setting'] = save_best_setting(train_val_loss_dict['best_setting'], current_setting)
            del train_source_insts, train_pred_scores, current_setting, tinputs, xs, ys, slabels, tlabels, loss
            torch.cuda.empty_cache()
            mdan.train()
            logger.info("Iteration {}, loss = {}, clf_losses_val = {}".format(t, running_loss, train_val_loss_dict['clf_losses_val'][-1]))
        time_end = time.time()
       
        #Choose the best epoch in terms of classifier's validation loss
        #best_epoch = train_val_loss_dict['clf_losses_val'].index(min(train_val_loss_dict['clf_losses_val']))
        best_epoch = train_val_loss_dict['best_setting']['epoch']
        pred_acc = train_val_loss_dict['best_setting']['accuracy']
        preds_labels = torch.max(train_val_loss_dict['best_setting']['target_logprob'], 1)[1]
        pred_scores = torch.exp(train_val_loss_dict['best_setting']['target_logprob']).to(device)[:,1]
        error_dicts[data_name[i]] = preds_labels.cpu().numpy() != test_target_labels.cpu().numpy()
        logger.info("Prediction accuracy on {} = {}, time used = {} seconds.".format(data_name[i], pred_acc, time_end - time_start))
        results[data_name[i]] = pred_acc
        train_pred_scores = train_val_loss_dict['best_setting']['train_pred_score']
        # Delete some unnecessary data
        del train_val_loss_dict['best_setting']['train_pred_score']
        del train_val_loss_dict['best_setting']['target_logprob']
        #AUCROC
        fpr, tpr, _ = roc_curve(test_target_labels.detach().cpu().numpy(), pred_scores.detach().cpu().numpy())
        auc_roc = auc(fpr, tpr)

    logger.info("Prediction accuracy with multiple source domain adaptation using madnNet: ")
    logger.info(results)
    pickle.dump(error_dicts, open("./error_dicts/{}-{}-{}-{}.pkl".format(args.name, args.frac, args.model, args.mode), "wb"))
    pickle.dump(train_val_loss_dict, open("./train_val_loss_dict/train_val_loss_dict-{}-{}-{}-{}-mu_{}-l_{}_last.pkl".format(args.name, args.frac, args.model, args.mode, mu, args.hidden_layers), "wb"))
    with open("./pred_scores/pred_scores-{}-{}-{}-{}-mu_{}-l_{}_last.pkl".format(args.name, args.frac, args.model, args.mode, mu, args.hidden_layers), "wb") as file_pred_scores:
      pickle.dump(pred_scores.detach().cpu().numpy(), file_pred_scores)
      pickle.dump(test_target_labels.detach().cpu().numpy(), file_pred_scores)
      pickle.dump(train_pred_scores.detach().cpu().numpy(), file_pred_scores)
      pickle.dump(train_source_labels.detach().cpu().numpy(), file_pred_scores)
    logger.info("*" * 100)
     
    with open('../../Results/mu_accuracy.csv','a') as csv_f:
      fieldnames = ['mu','accuracy', 'auc_roc', 'stopped_epoch', 'epochs', 'hidden_layers', 'lr']
      writer = csv.DictWriter(csv_f, fieldnames=fieldnames)
      writer.writerow({"mu": args.mu, "accuracy": pred_acc, "auc_roc": auc_roc, "stopped_epoch": best_epoch+1,"epochs": args.epoch, "hidden_layers": configs["hidden_layers"], "lr": lr})
    #del mdan, optimizer, test_target_labels, test_target_insts, train_source_insts, train_source_labels, slabels, tlabels, xs, ys, tinputs, train_pred_scores, pred_scores
    #torch.cuda.empty_cache()
else:
    raise ValueError("No support for the following model: {}.".format(args.model))


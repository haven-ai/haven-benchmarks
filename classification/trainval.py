from haven import haven_chk as hc
from haven import haven_results as hr
from haven import haven_utils as hu
import torch
import torchvision
import tqdm
import pandas as pd
import pprint
import itertools
import os
import pylab as plt
import exp_configs
import time
import numpy as np
from torch.utils.data import RandomSampler, DataLoader

from src import models
from src import datasets

import argparse


def trainval(exp_dict, savedir_base, datadir, reset=False, num_workers=0, use_cuda=False):
    # bookkeeping
    pprint.pprint(exp_dict)  # print the experiment configuration
    exp_id = hu.hash_dict(exp_dict)  # generate a unique id for the experiment
    savedir = os.path.join(savedir_base, exp_id)  # generate a route with the experiment id
    if reset:
        hc.delete_and_backup_experiment(savedir)

    os.makedirs(savedir, exist_ok=True)  # create the route to keep the experiment result
    hu.save_json(os.path.join(savedir, "exp_dict.json"), exp_dict)  # save the experiment config as json
    print("Experiment saved in %s" % savedir)

    # set cuda
    if use_cuda:
        device = 'cuda'
        assert torch.cuda.is_available(), 'cuda is not available, please run with "-c 0"'  # check if cuda is available 
    else:
        device = 'cpu'

    # Dataset
    # ==================
    # train set 
    # load the dataset for training from the datasets
    train_set = datasets.get_dataset(dataset_name=exp_dict["dataset"]["name"],
                                     train_flag=True,
                                     datadir=datadir,
                                     exp_dict=exp_dict)
    # val set
    # load the dataset for validation from the datasets
    val_set = datasets.get_dataset(dataset_name=exp_dict["dataset"]["name"],
                                    train_flag=False,
                                    datadir=datadir,
                                    exp_dict=exp_dict)
    
    # Model
    # ==================
    model = models.get_model(exp_dict).to(device)
    model_path = os.path.join(savedir, "model.pth")  # generate the route to keep the model of the experiment
    score_list_path = os.path.join(savedir, "score_list.pkl")  # generate the route to keep the score list

    if os.path.exists(score_list_path):  
        # resume experiment from the last checkpoint, load the latest model
        # epoch starts from last completed epoch plus one
        model.set_state_dict(hu.torch_load(model_path))
        score_list = hu.load_pkl(score_list_path)
        s_epoch = score_list[-1]["epoch"] + 1
    else:
        # restart experiment
        # epoch starts from zero
        score_list = []
        s_epoch = 0

    # Train & Val
    # ==================
    print("Starting experiment at epoch %d" % (s_epoch))

    train_sampler = RandomSampler(data_source=train_set, replacement=True, num_samples=2*len(val_set))
    train_loader = DataLoader(train_set, sampler=train_sampler, batch_size=exp_dict["batch_size"],
                              drop_last=True, num_workers=num_workers)

    val_sampler = torch.utils.data.SequentialSampler(val_set)
    val_loader = DataLoader(val_set,
                            sampler=val_sampler,
                            batch_size=1,
                            num_workers=num_workers)

    for e in range(s_epoch, exp_dict["max_epoch"]):
        score_dict = {}

        # Train the model
        train_dict = model.train_on_loader(train_loader)
        score_dict.update(train_dict)  # update the training loss

        # Validate and Visualize the model
        val_dict = model.val_on_loader(val_loader,
                        savedir_images=os.path.join(savedir, "images"),
                        n_images=3)
        score_dict.update(val_dict)  # update the validation accuracy

        # Get new score_dict
        score_dict["epoch"] = len(score_list)  # keep track of the epoch as score_list increments
        score_list += [score_dict]

        # Report & Save
        score_df = pd.DataFrame(score_list)
        print("\n", score_df.tail(), "\n")  # print out the epoch, train_loss, and val_acc in the score_list as a table
        hu.torch_save(model_path, model.get_state_dict()) # save the model state (i.e. state_dic, including optimizer) to the model path
        hu.save_pkl(score_list_path, score_list)
        print("Checkpoint Saved: %s" % savedir)

        # Save Best Checkpoint
        if e == 0 or (score_dict.get("val_acc", 0) > score_df["val_acc"][:-1].fillna(0).max()):
            hu.save_pkl(os.path.join(
                savedir, "score_list_best.pkl"), score_list)
            hu.torch_save(os.path.join(savedir, "model_best.pth"),
                          model.get_state_dict())
            print("Saved Best: %s" % savedir)

    print('Experiment completed et epoch %d' % e)


if __name__ == "__main__":
    # create a parser that will hold all the information necessary to parse the command line into Python data type
    parser = argparse.ArgumentParser()

    # add required arguments and default arguments so that the parser know how to take strings on the command line
    parser.add_argument('-e', '--exp_group_list', nargs="+")
    parser.add_argument('-sb', '--savedir_base', required=True)
    parser.add_argument('-d', '--datadir', required=True)
    parser.add_argument("-r", "--reset", default=0, type=int)
    parser.add_argument("-ei", "--exp_id", default=None)
    parser.add_argument("-j", "--run_jobs", default=0, type=int)
    parser.add_argument("-nw", "--num_workers", type=int, default=0)
    parser.add_argument("-c", "--use_cuda", type=int, default=0)    # user can define whether to run with cuda or cpu, default is not using cuda

    # parse the arguments from the command line
    args = parser.parse_args()

    # Collect experiments
    # ===================
    if args.exp_id is not None:
        # select one experiment
        savedir = os.path.join(args.savedir_base, args.exp_id)
        exp_dict = hu.load_json(os.path.join(savedir, "exp_dict.json"))

        exp_list = [exp_dict]

    else:
        # select exp group
        exp_list = []
        for exp_group_name in args.exp_group_list:
            exp_list += exp_configs.EXP_GROUPS[exp_group_name]

    # Run experiments
    # ===============
    if args.run_jobs:
        # run with job scheduler
        from haven import haven_jobs as hj
        hj.run_exp_list_jobs(exp_list, 
                       savedir_base=args.savedir_base, 
                       workdir=os.path.dirname(os.path.realpath(__file__)))

    else:
        for exp_dict in exp_list:
            # do trainval
            trainval(exp_dict=exp_dict,
                     savedir_base=args.savedir_base,
                     datadir=args.datadir,
                     reset=args.reset,
                     num_workers=args.num_workers,
                     use_cuda=args.use_cuda)

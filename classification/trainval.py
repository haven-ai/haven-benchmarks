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


def trainval(exp_dict, savedir_base, datadir, reset=False, num_workers=0):
    # bookkeeping
    pprint.pprint(exp_dict)  # print the experiment configuration
    exp_id = hu.hash_dict(exp_dict)  # generate random id for the experiment
    savedir = os.path.join(savedir_base, exp_id)  # generate the route to keep the experiment result
    if reset:
        hc.delete_and_backup_experiment(savedir)

    os.makedirs(savedir, exist_ok=True)
    hu.save_json(os.path.join(savedir, "exp_dict.json"), exp_dict)  # save the experiment config as json
    print("Experiment saved in %s" % savedir)

    # dataset
    # train set
    train_set = datasets.get_dataset(dataset_name=exp_dict["dataset"]["name"],
                                     train_flag=True,
                                     datadir=datadir,
                                     exp_dict=exp_dict)

    test_set = datasets.get_dataset(dataset_name=exp_dict["dataset"]["name"],
                                    train_flag=False,
                                    datadir=datadir,
                                    exp_dict=exp_dict)

    model = models.get_model(exp_dict["model"]["name"]).cuda()
    model_path = os.path.join(savedir, "model.pth")
    score_list_path = os.path.join(savedir, "score_list_pkl")

    if os.path.exists(score_list_path):
        model.load_state_dict(hu.torch_load(model_path))
        score_list = hu.load_pkl(score_list_path)
        s_epoch = score_list[-1]["epoch"] - 1
    else:
        score_list = []
        s_epoch = 0

    # train & val
    print("Starting experiment at epoch %d" % (s_epoch))

    train_sampler = RandomSampler(data_source=train_set, replacement=True, num_samples=2*len(test_set))
    train_loader = DataLoader(train_set, sampler=train_sampler, batch_size=exp_dict["batch_size"],
                              drop_last=True, num_workers=num_workers)
    for e in range(s_epoch, exp_dict["max_epoch"]):
        score_dict = {}
        # todo: figure out how to use mlp and print the score then save the best checkpoint

    print('Experiment completed et epoch %d' % e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-e', '--exp_group_list', nargs="+")
    parser.add_argument('-sb', '--savedir_base', required=True)
    parser.add_argument('-d', '--datadir', required=True)
    parser.add_argument("-r", "--reset", default=0, type=int)
    parser.add_argument("-ei", "--exp_id", default=None)
    parser.add_argument("-j", "--run_jobs", default=0, type=int)
    parser.add_argument("-nw", "--num_workers", type=int, default=0)

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
        from haven import haven_jobs as hjb

        jm = hjb.JobManager(exp_list=exp_list, savedir_base=args.savedir_base)
        jm_summary_list = jm.get_summary()
        print(jm.get_summary()['status'])

        import usr_configs as uc

        uc.run_jobs(exp_list, args.savedir_base, args.datadir)

    else:
        for exp_dict in exp_list:
            # do trainval
            trainval(exp_dict=exp_dict,
                     savedir_base=args.savedir_base,
                     datadir=args.datadir,
                     reset=args.reset,
                     num_workers=args.num_workers)

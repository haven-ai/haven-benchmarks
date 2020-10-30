from haven import haven_utils as hu
RUNS = [0, 1, 2]
opt_list = ['sgd']
EXP_GROUPS = {}

# def get_benchmark(benchmark, opt_list=opt_list):
#     if benchmark == 'mnist':
#         EXP_GROUPS['mnist'] = {"dataset": {'name': 'mnist'},
#                 "model": {'name': 'mlp'},
#                 "runs": RUNS,
#                 "batch_size": [128],
#                 "max_epoch": [100],
#                 'dataset_size': [
#                     {'train': 'all', 'val': 'all'},
#                 ],
#                 "loss_func": ["softmax_loss"],
#                 "opt": opt_list,
#                 "acc_func": ["softmax_accuracy"],
#                 }
#     elif benchmark == 'syn':
#         EXP_GROUPS['syn'] = {"dataset": {'name': 'synthetic'},
#                 "model": {'name': 'logistic'},
#                 "runs": RUNS,
#                 "batch_size": [128],
#                 "max_epoch": [200],
#                 'dataset_size': [
#                     {'train': 'all', 'val': 'all'},
#                 ],
#                 "loss_func": ["softmax_loss"],
#                 "opt": opt_list,
#                 "acc_func": ["softmax_accuracy"],
#                 'margin': [0.05, 0.1, 0.5, 0.01],
#                 "n_samples": [1000],
#                 "d": 20
#                 }
        
#     elif benchmark == 'cifar10':
#         EXP_GROUPS['cifar10'] = {"dataset": {'name': 'synthetic'},
#                 "model": {'name': ["densenet121", "resnet34"]},
#                 "runs": RUNS,
#                 "batch_size": [128],
#                 "max_epoch": [200],
#                 'dataset_size': [
#                     {'train': 'all', 'val': 'all'},
#                 ],
#                 "loss_func": ["softmax_loss"],
#                 "opt": opt_list,
#                 "acc_func": ["softmax_accuracy"]
#                 }


EXP_GROUPS['mnist'] = {"dataset": {'name': 'mnist'},
        "model": {'name': 'mlp'},
        "runs": RUNS,
        "batch_size": [128],
        "max_epoch": [100],
        'dataset_size': [
            {'train': 'all', 'val': 'all'},
        ],
        "loss_func": ["softmax_loss"],
        "opt": opt_list,
        "acc_func": ["softmax_accuracy"],
        }

EXP_GROUPS['syn'] = {"dataset": {'name': 'synthetic'},
        "model": {'name': 'logistic'},
        "runs": RUNS,
        "batch_size": [128],
        "max_epoch": [200],
        'dataset_size': [
            {'train': 'all', 'val': 'all'},
        ],
        "loss_func": ["softmax_loss"],
        "opt": opt_list,
        "acc_func": ["softmax_accuracy"],
        'margin': [0.05, 0.1, 0.5, 0.01],
        "n_samples": [1000],
        "d": 20
        }

EXP_GROUPS['cifar10'] = {"dataset": {'name': 'synthetic'},
        "model": {'name': ["densenet121", "resnet34"]},
        "runs": RUNS,
        "batch_size": [128],
        "max_epoch": [200],
        'dataset_size': [
            {'train': 'all', 'val': 'all'},
        ],
        "loss_func": ["softmax_loss"],
        "opt": opt_list,
        "acc_func": ["softmax_accuracy"]
        }
# TODO: may need more info for each optimizer
EXP_GROUPS = {k: hu.cartesian_exp_group(v) for k, v in EXP_GROUPS.items()}
from haven import haven_utils as hu

EXP_GROUPS = {}
EXP_GROUPS['mnist_batch_size'] = {"dataset": {'name': 'mnist'},
                                  "model": {'name': 'mlp'},
                                  "batch_size": [64, 128, 256, 512, 1024],
                                  "max_epoch": [100],
                                  'dataset_size': [
                                      {'train': 'all', 'val': 'all'},
                                  ],
                                  'optimizer': ['adam'],
                                  'lr': [1e-5]
                                  }
EXP_GROUPS['mnist_learning_rate'] = {"dataset": {'name': 'mnist'},
                                  "model": {'name': 'mlp'},
                                  "batch_size": [256],
                                  "max_epoch": [100],
                                  'dataset_size': [
                                      {'train': 'all', 'val': 'all'},
                                  ],
                                  'optimizer': ['adam'],
                                  'lr': [5e-6, 1e-5, 5e-5, 1e-4, 5e-4]
                                  }
EXP_GROUPS = {k: hu.cartesian_exp_group(v) for k, v in EXP_GROUPS.items()}
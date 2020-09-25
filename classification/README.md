# MNIST Example

## ToDo

- Create `exp_configs.py` and define two experiment groups (see reference https://github.com/ElementAI/LCFCN/blob/master/exp_configs.py)
    - EXP_GROUPS[`mnist_batch_size`] for comparng across batch sizes
    - EXP_GROUPS[`mnist_learning_rate`] for comparing across learning rates

- Create the MNIST dataset as follows
    - create `src/datasets.py` 
    - create function `def get_dataset(dataset_name, exp_dict)` that returns the MNIST dataset as Pytorch object (see https://github.com/IssamLaradji/sls/blob/master/src/datasets.py)

- Create the LeNet as follows, 
    - create `src/models.py` 
    - create function `def get_model(model_name, exp_dict)` that returns the LeNet model as Pytorch object (see https://github.com/IssamLaradji/sls/blob/master/src/models.py where it is defined as `mlp`)

- Create a `trainval.py` that follows the structure in `trainval.py` in https://github.com/ElementAI/LCFCN
    - First create the `parse_args` arguments that allows us to speciify the exp_group, savedir_base and so on
    - Second create the `def trainval` function that loads the datasets and the model
    - Then create the main `trainval` loop that performs the train and validation
    - In this case implement the functions `train_on_loader`, and `val_on_loader` in the model 
    - Aggregate the results from `train_on_loader`, and `val_on_loader` into a score_dict, and save the score_list
    - Each score dict should record the epoch, validation accuracy and training loss

- Run `trainval.py` with experiment group . The command should be something like

```
python trainval.py -e mnist_batch_size -sb <savedir_base> -d <datadir> -r 1
```
- Visualize the results 
    - Create a jupyter file as  `results/mnist_benchmarks.ipyb`
    - Use the jupyer code defined in the README of `https://github.com/ElementAI/LCFCN` to visualize the results
    - Show train_loss line plots for the score_list across batch sizes
    - Show validation accuracy bar chart across batch sizes 

## Expected structure
```
classification/
├── src/
│   ├── __init__.py
│   ├── datasets.py
│   └── models.py
├── scripts/
│   ├── visualize_mnist.py
│   └── train_on_single_image.py
├── exp_configs.py
├── README.md
└── trainval.py
```

## Other libraries that use Haven
- LCFCN https://github.com/ElementAI/LCFCN
- Embedding Propagation https://github.com/ElementAI/embedding-propagation
- Bilevel Augment https://github.com/ElementAI/bilevel_augment
- SSN optimizer https://github.com/IssamLaradji/ssn
- SLS optimizer https://github.com/IssamLaradji/sls
- Ada SLS optimizer https://github.com/IssamLaradji/ada_sls
- SPS optimizer https://github.com/IssamLaradji/sps
- Fish Dataset https://github.com/alzayats/DeepFish
- Covid Weak Supervision https://github.com/IssamLaradji/covid19_weak_supervision

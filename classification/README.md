# Classification - MNIST Example

## 1. Install Haven
`pip install --upgrade git+https://github.com/haven-ai/haven-ai`

This command installs the [Haven library](https://github.com/haven-ai/haven-ai) which helps in managing the experiments.

## 2. Experiement
Run the experiment with the following command on terminal

`python trainval.py -e <exp_group_name> -sb <directory_to_save_results> -d <directory_to_save_datasets> -r 1`

-e: Indicate which experiemnt you want to run

option1: mnist_batch_size. This experiemnt group investigates the effect of training batch sizes on the model. 
    
option2: mnist_learning_rate. This experiment group investigates the effect of learning rates on the model.

-sb: Specify where the experiment results will be stored

-d: Specify where the datasets will be stored

-r: Indicate whether to reset the model after each experiment


## 3. Visualize Results
### 3.1 Launch Jupyter by running the following  on terminal
```
jupyter nbextension enable --py widgetsnbextension --sys-prefix
jupyter notebook
```

### 3.2 On a Jupyter cell, run the following script
```python
from haven import haven_jupyter as hj
from haven import haven_results as hr
from haven import haven_utils as hu
savedir_base = '<the directory of the result>'
fname = '<the directory of the experiemnt configuration file>'

exp_list = []
# indicate the experiment group you have run here
for exp_group in [
    'mnist_learning_rate', 
    'mnist_batch_size'
                 ]:
    exp_list += hu.load_py(fname).EXP_GROUPS[exp_group]

# get experiments
rm = hr.ResultManager(exp_list=exp_list, 
                      savedir_base=savedir_base, 
                      verbose=0
                     )
hj.get_dashboard(rm, vars(), wide_display=True)
```

## 4. Results

### 4.1 Table

<img width="1065" alt="Screen Shot 2020-10-18 at 6 12 01 PM" src="https://user-images.githubusercontent.com/46538726/96387114-dbb34b00-116d-11eb-824d-c532436a57d1.png">

### 4.2 Line Plots

<img width="1032" alt="Screen Shot 2020-10-26 at 2 31 40 AM" src="https://user-images.githubusercontent.com/46538726/97140951-2287de80-1734-11eb-838c-e5200b71a4ff.png">

<img width="1029" alt="Screen Shot 2020-10-26 at 2 32 44 AM" src="https://user-images.githubusercontent.com/46538726/97140984-359aae80-1734-11eb-8d40-8b3ba1b9aa81.png">

### 4.3 Bar plots

<img width="1039" alt="Screen Shot 2020-10-26 at 2 33 07 AM" src="https://user-images.githubusercontent.com/46538726/97140999-3f241680-1734-11eb-9ddf-a0bd9209298a.png">

<img width="1030" alt="Screen Shot 2020-10-26 at 2 34 45 AM" src="https://user-images.githubusercontent.com/46538726/97141014-4cd99c00-1734-11eb-8032-28f005fd05bd.png">

## 5. Code Structure
```
classification/
│
├── src/
│   ├── datasets.py
│   └── models.py
├── scripts/
│   ├── visualize_mnist.py
│   └── train_on_single_image.py
├── exp_configs.py
├── README.md
└── trainval.py
```

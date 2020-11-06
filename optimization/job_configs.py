import os
# need to set up the environment variable before use
# ACCOUNT_ID = os.environ['SLURM_ACCOUNT']
ACCOUNT_ID = "123123"

# change the output directory
JOB_CONFIG = {
    'time': '12:00:00',
    'cpu-per-task': '2',
    'mem-per-cpu': '20G',
    'gres': 'gpu:p100:1',
    'output': 'OUTPUT_DIR_FOR_SLURM'
}

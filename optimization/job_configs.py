import os
# need to set up the environment variable before use
ACCOUNT_ID = os.environ['SLURM_ACCOUNT']

# change the output directory
JOB_CONFIG = {
    'time': '12:00:00',
    'cpu_per_task': 2,
    'mem_per_cpu': '20G',
    'gres': 'gpu:p100:1',
    'output': 'OUTPUT_DIR_FOR_SLURM'
}

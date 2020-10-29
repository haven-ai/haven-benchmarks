#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --account=def-dnowrouz-ab
#SBATCH --mem-per-cpu=20G
#SBATCH --output=/home/xhdeng/shared/results/classification/%x-%j.out
python ./trainval.py -e mnist_batch_size -sb /home/xhdeng/shared/results/classification/mnist_batch_size -d /home/xhdeng/shared/datasets/mnist -r 1
python ./trainval.py -e mnist_learning_rate -sb /home/xhdeng/shared/results/classification/mnist_learning_rate -d /home/xhdeng/shared/datasets/mnist -r 1
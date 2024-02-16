#!/bin/bash

echo "Starting Image Data Augmentation - Low Rank Approximation with the SVD"

# Descriptors
#SBATCH --job-name=DL_CV_SVD
#SBATCH --output=/hpc-logs/DL_CV_SVD.out

# Resources
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus 1
#SBATCH --mem=10G

# Notifications
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=<cfgauss@domain.com>

# Execution
echo "Purging modules"
module purge

echo "Loading Python 3.10.12 module"
module load python/3.10.12

echo "Creating virtual environment svd-env"
virtualenv svd-env
source svd-env/bin/activate

echo "Installing requirements"
pip install -r requirements.txt

echo "Starting Python script: do-study.py"
python setup.py
python /study/do-study.py 1

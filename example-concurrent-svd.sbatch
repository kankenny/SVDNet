#!/bin/bash

echo "Starting Image Data Augmentation - Low Rank Approximation with the SVD"

###############################################################
# Image Data Augmentation - Low Rank Approximation with the SVD
###############################################################



#SBATCH --job-name=DL_CV_SVD
#SBATCH --output=/hpc-logs/DL_CV_SVD.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=3
#SBATCH --gpus 1
#SBATCH --mem=10G
#SBATCH --output=output_%j.out
#SBATCH --error=error_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=<cfgauss@domain.com>


echo "Purging modules"
module purge
echo "Loading Python 3.10.12 module"
module load python/3.10.12

echo "Creating virtual environment svd-env"
virtualenv svd-env
source svd-env/bin/activate

echo "Installing requirements"
pip install -r requirements.txt


echo "Starting Python scripts: do-study.py"
python setup.py

srun -n 1 python /study/do-study.py 1 &
srun -n 1 python /study/do-study.py 2 & 
srun -n 1 python /study/do-study.py 3 &

wait
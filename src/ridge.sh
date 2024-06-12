#!/bin/bash
#SBATCH --account=rrg-pbellec
#SBATCH --mem-per-cpu=1.5G      # increase as needed
#SBATCH --cpus-per-task=6  # Cores proportional to GPUs: 6 on Cedar, 10 on Béluga, 16 on Graham.
#SBATCH --time=140:00:00

module load python/3.10
# Prepare virtualenv
VENV_PATH=/scratch/isilb/encoder_venv

virtualenv --no-download $VENV_PATH
source $VENV_PATH/bin/activate
pip install --no-index --upgrade pip
pip install --no-index -r ../requirements.txt


SOURCEDIR=~/src   #I copied the files in this directory inside my home directory

# Start ridge regression
python main.py
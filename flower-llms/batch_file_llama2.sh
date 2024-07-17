#!/bin/bash

#SBATCH -c 4  # Number of Cores per Task
#SBATCH --mem=100000  # Requested Memory
#SBATCH -p gypsum-rtx8000  # Partition
#SBATCH -G 1  # Number of GPUs
#SBATCH -t 40:00:00  # Job time limit
#SBATCH -o Llama2_centralized.out  # %j = job ID

source ~/../../work/tmehboob_umass_edu/Federated_learning/Flower_fm/fed_learning/LLMs-Llama/pnnl/bin/activate

# Execute your application
python3 /work/tmehboob_umass_edu/Federated_learning/Flower_fm/fed_learning/LLMs-Llama/llm_llama2.py

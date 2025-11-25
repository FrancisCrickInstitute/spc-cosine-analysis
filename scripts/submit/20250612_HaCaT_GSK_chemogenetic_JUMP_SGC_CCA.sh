#!/bin/bash

#SBATCH --job-name=cosine_distance
#SBATCH --mem=1024G
#SBATCH --time=4:00:00
#SBATCH --partition=ncpu
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1

# Specify the path to your Python environment
PYTHON="/camp/home/tuerslj/.conda/envs/cosine_distance/bin/python"

# Path to config file
CONFIG_PATH="/nemo/stp/hts/working/Joe_Tuersley/code/spc-cosine-analysis/scripts/config/20250612_HaCaT_GSK_chemogenetic_JUMP_SGC_CCA.yml"

# Path to the Python script
SCRIPT_PATH="/nemo/stp/hts/working/Joe_Tuersley/code/spc-cosine-analysis/scripts/spc_analysis/main.py"

# Add the scripts directory to PYTHONPATH
export PYTHONPATH="/nemo/stp/hts/working/Joe_Tuersley/code/spc-cosine-analysis/scripts/:$PYTHONPATH"

# Execute the command with the config file as an argument
$PYTHON $SCRIPT_PATH --config $CONFIG_PATH
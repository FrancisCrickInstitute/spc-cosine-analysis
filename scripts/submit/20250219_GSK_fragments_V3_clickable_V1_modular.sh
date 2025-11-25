#!/bin/bash

#SBATCH --job-name=cosine_distance
#SBATCH --mem=1024G
#SBATCH --time=1:00:00
#SBATCH --partition=ncpu
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1

# Specify the path to your Python environment
PYTHON="/camp/home/tuerslj/.conda/envs/cosine_distance/bin/python"

# Path to config file
CONFIG_PATH="/nemo/stp/hts/working/Joe_Tuersley/code/spherical-phenotype-clustering-2/cosine_distance/scripts/config/20250219_GSK_fragments_V3_clickable_V1_config.yml"

# Path to the Python script
SCRIPT_PATH="/nemo/stp/hts/working/Joe_Tuersley/code/spherical-phenotype-clustering-2/cosine_distance/scripts/spc_analysis/main.py"

# Add the scripts directory to PYTHONPATH
export PYTHONPATH="/nemo/stp/hts/working/Joe_Tuersley/code/spherical-phenotype-clustering-2/cosine_distance/scripts:$PYTHONPATH"

# Execute the command with the config file as an argument
$PYTHON $SCRIPT_PATH --config $CONFIG_PATH
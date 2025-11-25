#!/bin/bash

#SBATCH --job-name=ResNet152_no_CRISPR_cosine_analysis
#SBATCH --mem=1024G
#SBATCH --time=8:00:00
#SBATCH --partition=ncpu
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1

# Specify the path to your Python environment
PYTHON="/camp/home/tuerslj/.conda/envs/cosine_distance/bin/python"

# Path to config file
CONFIG_PATH="/nemo/stp/hts/working/Joe_Tuersley/code/spc-cosine-analysis/scripts/config/20250715_HaCaT_all_datasets_no_CRISPR_ResNet_152_pcr_lr3e-4_t0p10_redo_2.yml"

# Path to the Python script
SCRIPT_PATH="/nemo/stp/hts/working/Joe_Tuersley/code/spc-cosine-analysis/scripts/spc_analysis/main.py"

# Add the scripts directory to PYTHONPATH
export PYTHONPATH="/nemo/stp/hts/working/Joe_Tuersley/code/spc-cosine-analysis/scripts/:$PYTHONPATH"

# Fix Numba caching issue - disable disk cache but keep JIT compilation
export NUMBA_CACHE_DIR=/tmp/numba_cache_$$
export NUMBA_DISABLE_PERFORMANCE_WARNINGS=1

# Execute the command
$PYTHON $SCRIPT_PATH --config $CONFIG_PATH
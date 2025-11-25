#!/bin/bash

#SBATCH --job-name=20251122_HaCaT_gsk_all_no_CRISPR_no_HTC_V1_no_CCA_V1_ResNet18
#SBATCH --mem=1024G
#SBATCH --time=8:00:00
#SBATCH --partition=ncpu
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1

# Specify the path to your Python environment
PYTHON="/camp/home/tuerslj/.conda/envs/cosine_distance/bin/python"

# Path to config file
CONFIG_PATH="/nemo/stp/hts/working/Joe_Tuersley/code/spc-cosine-analysis/scripts/config/20251122_HaCaT_gsk_all_no_CRISPR_no_HTC_V1_no_CCA_V1_ResNet18.yml"

# Path to the Python script
SCRIPT_PATH="/nemo/stp/hts/working/Joe_Tuersley/code/spc-cosine-analysis/scripts/spc_analysis/main.py"

# Add the scripts directory to PYTHONPATH
export PYTHONPATH="/nemo/stp/hts/working/Joe_Tuersley/code/spc-cosine-analysis/scripts/:$PYTHONPATH"

# Fix Numba caching issue - disable disk cache but keep JIT compilation
export NUMBA_CACHE_DIR=/tmp/numba_cache_$$
export NUMBA_DISABLE_PERFORMANCE_WARNINGS=1

# Execute the command
$PYTHON $SCRIPT_PATH --config $CONFIG_PATH
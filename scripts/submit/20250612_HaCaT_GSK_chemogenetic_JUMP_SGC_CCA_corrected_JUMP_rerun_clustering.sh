#!/bin/bash

#SBATCH --job-name=hierarchical_clustering_rerun
#SBATCH --mem=512G
#SBATCH --time=2:00:00
#SBATCH --partition=ncpu
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1

# Specify the path to your Python environment
PYTHON="/camp/home/tuerslj/.conda/envs/cosine_distance/bin/python"

# MODIFY THESE PATHS FOR YOUR SPECIFIC RUN:
DATA_FILE="/nemo/stp/hts/working/Joe_Tuersley/code/spc-cosine-analysis/analysis/20250612_HaCaT_GSK_chemogenetic_JUMP_SGC_CCA/spc_analysis_20250723_135914/data/visualization_data_treatment_agg.csv"
OUTPUT_DIR="/nemo/stp/hts/working/Joe_Tuersley/code/spc-cosine-analysis/analysis/20250612_HaCaT_GSK_chemogenetic_JUMP_SGC_CCA/spc_analysis_20250723_135914/"

# Path to the rerun script
SCRIPT_PATH="/nemo/stp/hts/working/Joe_Tuersley/code/spc-cosine-analysis/scripts/spc_analysis/rerun_clustering.py"

# Add the scripts directory to PYTHONPATH
export PYTHONPATH="/nemo/stp/hts/working/Joe_Tuersley/code/spc-cosine-analysis/scripts/:$PYTHONPATH"

# Execute the rerun command
$PYTHON $SCRIPT_PATH --data_file $DATA_FILE --output_dir $OUTPUT_DIR

echo "Hierarchical clustering rerun completed!"
echo "Results saved to: $OUTPUT_DIR/hierarchical_clustering/hierarchical_cluster_map_rerun/"
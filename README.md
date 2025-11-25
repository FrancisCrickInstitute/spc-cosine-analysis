# SPC Cosine Analysis Pipeline

Python pipeline for analysing Spherical Phenotype Clustering (SPC) embeddings with comprehensive distance metrics, landmark identification, compound scoring, and hierarchical clustering visualizations.

> **Note:** This pipeline processes data downstream of the [spc-distributed](https://github.com/FrancisCrickInstitute/spc-distributed) repository, which generates the SPC embeddings used as input here.

## Table of Contents
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Input Files](#input-files)
- [Configuration File](#configuration-file)
- [Processing Pipeline](#processing-pipeline)
- [Metrics Calculated](#metrics-calculated)
- [Landmark Analysis](#landmark-analysis)
- [Compound Scoring](#compound-scoring)
- [Hierarchical Clustering](#hierarchical-clustering)
- [Output Structure](#output-structure)
- [Usage Examples](#usage-examples)
- [Troubleshooting](#troubleshooting)

## Requirements

**Python >= 3.9**

See `env.yml` for complete package requirements.

**Key dependencies:**
- `pandas`, `numpy`, `scikit-learn`, `scipy`, `pyyaml`, `tqdm`
- `plotly`, `matplotlib`, `seaborn`, `umap-learn`

## Installation

```bash
conda env create -f scripts/env.yml
conda activate cosine_distance
```

## Quick Start

```bash
# Submit to SLURM
sbatch submit/20251111_HaCaT_gsk_all_no_CRISPR_no_HTC_V1_no_CCA_V1_ResNet18.sh

# Or run directly
python spc_analysis/main.py --config config/your_config.yml
```

### Example Files

The latest example configuration and submission scripts:

| File | Description |
|------|-------------|
| `scripts/config/20251122_HaCaT_gsk_all_no_CRISPR_no_HTC_V1_no_CCA_V1_ResNet18.yml` | **Latest config** - Full example with all options |
| `scripts/submit/20251111_HaCaT_gsk_all_no_CRISPR_no_HTC_V1_no_CCA_V1_ResNet18.sh` | **Latest submit script** - SLURM submission example |

Use these as templates for your own analyses.

---

## Input Files

### 1. Embeddings File (required)
NumPy array (`.npy`) containing SPC embeddings from the [spc-distributed](https://github.com/FrancisCrickInstitute/spc-distributed) pipeline:
- Shape: `(n_samples, embedding_dim)` - typically 384 dimensions
- One row per well

### 2. Metadata File (required)
CSV file with well-level metadata containing:
- `plate` - Plate identifier
- `well` - Well position (e.g., A01, B12)
- `treatment` - Compound name with concentration (e.g., `CompoundA@10.0`)
- `compound_name` - Compound identifier
- `compound_uM` - Concentration in micromolar

**Optional but useful columns:**
- `moa` - Mechanism of action
- `library` - Compound library source
- `PP_ID`, `SMILES` - Compound identifiers

### 3. Harmony File (optional)
CellProfiler/Harmony output for cell count information:
- Used for cell count correlation analysis
- Mapped via `harmony_column_mapping` in config

### 4. Config File (required)
YAML configuration controlling all pipeline behaviour. See [Configuration File](#configuration-file) section.

---

## Configuration File

The config YAML controls all aspects of the pipeline. Key sections:

```yaml
# ============================================================================
# INPUT/OUTPUT PATHS
# ============================================================================
embeddings_file: "/path/to/embeddings.npy"
metadata_file: "/path/to/metadata.csv"
harmony_file: "/path/to/harmony_output.csv"  # Optional
output_dir: "/path/to/output"

# ============================================================================
# PLATE DEFINITIONS
# ============================================================================
plate_definitions:
  "32084":
    type: "reference"    # Used to identify landmarks
    library: "JUMP"
  "32085":
    type: "test"         # Compared against landmarks
    library: "GSK_Fragments"

# ============================================================================
# LIBRARY DEFINITIONS
# ============================================================================
library_definitions:
  reference_libraries:
    - "JUMP"
    - "SGC"
  test_libraries:
    - "GSK_Fragments"

# ============================================================================
# ANALYSIS PARAMETERS
# ============================================================================
mad_threshold: 0.05              # MAD threshold for landmark selection
dmso_threshold_percentile: "99"  # Percentile for DMSO distance threshold
similarity_threshold: 0.2        # Cosine similarity threshold

# ============================================================================
# VISUALIZATIONS
# ============================================================================
create_hierarchical_chunks: true
hierarchical_chunk_size: 200
```

---

## Processing Pipeline

The pipeline executes the following steps:

1. **Data Loading & Merging** - Load embeddings, metadata, and optional Harmony data
2. **Dataset Splitting** - Separate reference and test plates based on `plate_definitions`
3. **Dispersion Metrics** - Calculate MAD, variance, std of cosine distances per treatment
4. **DMSO Distance** - Compute distance from DMSO centroid for all treatments
5. **Landmark Identification** - Select reference compounds with low dispersion and high DMSO distance
6. **Landmark Distance** - Calculate distances from all compounds to identified landmarks
7. **Compound Scoring** - Generate harmonic mean scores combining multiple metrics
8. **Visualizations** - Create correlation plots, histograms, UMAP/t-SNE, hierarchical clustering

---

## Metrics Calculated

### Dispersion Metrics
Calculated per treatment across replicate wells:

| Metric | Description |
|--------|-------------|
| `mad_cosine` | Median Absolute Deviation of pairwise cosine distances |
| `var_cosine` | Variance of pairwise cosine distances |
| `std_cosine` | Standard deviation of pairwise cosine distances |

### Distance Metrics
| Metric | Description |
|--------|-------------|
| `cosine_distance_from_dmso` | Distance from treatment centroid to DMSO centroid |
| `closest_landmark_distance` | Cosine distance to nearest landmark compound |

---

## Landmark Analysis

### What Are Landmarks?

**Landmarks** are reference compounds with:
1. High reproducibility (low MAD across replicates)
2. Strong phenotypic signal (high distance from DMSO)
3. Serve as "anchors" in phenotypic space for comparing test compounds

### Selection Criteria

Landmarks are identified from reference plates meeting:
- MAD below `mad_threshold` (configurable)
- DMSO distance above percentile threshold (configurable)

### Outputs

- `cellprofiler_landmarks.csv` - Identified landmark compounds
- `reference_to_landmark_distances.csv` - Reference compound distances to landmarks
- `test_to_landmark_distances.csv` - Test compound distances to landmarks

---

## Compound Scoring

Compounds are scored using weighted harmonic means combining multiple metrics:

### 2-Term Harmonic Means
- `harmonic_mean_2term` - DMSO distance + landmark similarity
- `harmonic_mean_2term_mad_cosine` - DMSO distance + MAD (inverted)

### 3-Term Harmonic Means
- `harmonic_mean_3term` - DMSO distance + landmark similarity + MAD
- `harmonic_mean_3term_var_cosine` - Using variance instead of MAD
- `harmonic_mean_3term_std_cosine` - Using standard deviation

Higher scores indicate compounds with strong, reproducible phenotypes similar to known landmarks.

---

## Hierarchical Clustering

Generates chunked hierarchical clustering heatmaps as multi-page PDFs.

### Split Types

| Split | Description |
|-------|-------------|
| `test_and_reference` | All treatments combined |
| `test_only` | Test compounds only |
| `reference_only` | Reference compounds only |
| `reference_landmark` | Reference landmarks only |
| `test_and_all_reference_landmarks` | Test + all reference landmarks |
| `test_valid_and_relevant_landmarks` | Valid test + relevant landmarks |

### Outputs

```
hierarchical_clustering/
├── test_and_reference_all_chunks.pdf
├── test_only_all_chunks.pdf
├── reference_only_all_chunks.pdf
├── reference_landmark_all_chunks.pdf
├── test_and_all_reference_landmarks_all_chunks.pdf
└── test_valid_and_relevant_landmarks_all_chunks.pdf
```

---

## Output Structure

```
output_dir/spc_analysis_YYYYMMDD_HHMMSS/
├── data/
│   ├── config.json                    # Saved configuration
│   ├── merged_data.csv                # Combined embeddings + metadata
│   └── spc_for_viz_app.csv            # Final viz export (no embeddings)
│
├── analysis/
│   ├── mad_analysis/
│   │   ├── reference_metrics.csv
│   │   └── test_metrics.csv
│   ├── dmso_distances/
│   │   ├── reference_dmso_distances.csv
│   │   └── test_dmso_distances.csv
│   ├── landmark_distances/
│   │   ├── reference_to_landmark_distances.csv
│   │   └── test_to_landmark_distances.csv
│   ├── compound_reference_scores.csv
│   └── compound_test_scores.csv
│
└── visualizations/
    ├── correlation/                   # Cell count correlation plots
    ├── histograms/                    # Distribution histograms
    │   ├── dispersion_metrics/
    │   ├── dmso_distances/
    │   ├── landmark_distances/
    │   └── scores/
    ├── dmso_distributions/            # DMSO distance distributions
    ├── landmark_plots/                # Landmark vs metric plots
    ├── dmso_vs_metrics/               # DMSO distance vs dispersion
    ├── dimensionality_reduction/      # UMAP and t-SNE plots
    │   ├── umap/
    │   └── tsne/
    └── hierarchical_clustering/       # Clustering PDFs
```

---

## Usage Examples

### 1. Basic SLURM Submission
```bash
sbatch submit/20251111_HaCaT_gsk_all_no_CRISPR_no_HTC_V1_no_CCA_V1_ResNet18.sh
```

### 2. Direct Python Execution
```bash
cd scripts
python spc_analysis/main.py --config config/your_config.yml
```

### 3. Custom Config Path
```bash
python spc_analysis/main.py --config /path/to/custom_config.yml
```

---

## Troubleshooting

### "No embedding columns found"
Ensure your embeddings file is correctly formatted as a NumPy array with shape `(n_samples, n_dimensions)`.

### "No DMSO samples found"
Check that your metadata contains treatments starting with "DMSO" in the `treatment` column.

### Landmark analysis produces no landmarks
- Lower the `mad_threshold` value
- Lower the `dmso_threshold_percentile`
- Verify reference plates are correctly labelled in `plate_definitions`

### Memory error on HPC
Increase SLURM memory:
```bash
#SBATCH --mem=1024G
```

### Hierarchical clustering not running
Ensure `create_hierarchical_chunks: true` in your config file.

### Empty visualizations
Check that `spc_for_viz_app.csv` was generated successfully in the data directory.

---

## Typical Processing Statistics

| Metric | Value |
|--------|-------|
| Input | ~50,000 wells across 10+ plates |
| Embedding dimensions | 384 |
| Reference treatments | ~2,000 |
| Test treatments | ~10,000 |
| Identified landmarks | ~100-500 |
| Output | Interactive HTML plots + PDF clustering |
| Time | ~1-2 hours on HPC (1TB RAM) |

---

## Questions or Issues?

1. Check SLURM output logs in `scripts/slurm_out/`
2. Review config file structure against example configs
3. Ensure plate barcodes in `plate_definitions` match metadata exactly
4. Verify `library_definitions` lists match your library names

---

## Related Repositories

- [spc-distributed](https://github.com/FrancisCrickInstitute/spc-distributed) - Upstream SPC embedding generation

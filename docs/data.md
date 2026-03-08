# Data

## Datasets (all free, <1GB total)

| Data Type | Dataset | Location | Use |
|-----------|---------|----------|-----|
| Genomics | DNA Classification (Diabetes) | `data/dna_classification/` | CNN-based diabetes DNA classification |
| Genomics | ClinVar `variant_summary.txt` (~50MB) | `data/clinvar/` | Variant → disease lookup |
| Genomics + Transcriptomics | METABRIC from Kaggle (~50MB) | `data/metabric/` | Mutations + mRNA z-scores + clinical |
| Pharmacology | PharmGKB clinical annotations | `data/pharmgkb/` | Drug-gene associations |
| Drug Safety | Kaggle Drug Side Effects (<50MB) | `data/drug_side_effects/` | Adverse reactions |

### DNA Classification Dataset (`data/dna_classification/`)

Source: [DNA Classification Project](https://github.com/mobilttterbang/DNA_Classification_Project)

| File | Description |
|------|-------------|
| `raw/Complete_DM_DNA_Sequence.csv` | Labeled DNA sequences (DMT1, DMT2, NONDM) |
| `raw/DMT2_1296.fasta` | Type 2 Diabetes sequences in FASTA format |
| `raw/NONDM.fasta` | Non-diabetic sequences in FASTA format |
| `models/CNN_2Layers_3mers.h5` | Pre-trained 2-layer CNN weights |

## Patient Cases (from METABRIC)

- **Case 1**: PIK3CA + TP53 mutations, high ESR1, on tamoxifen → treatment optimization
- **Case 2**: BRCA1 variant, triple-negative, young → risk assessment + clinical trials
- **Case 3**: Multiple low-significance variants, conflicting signals → diagnostic dilemma

## Format

- All data as pandas DataFrames, cached as Parquet in `data/`
- Files organized per data source (e.g., ClinVar, METABRIC, PharmGKB)
- `data/` is gitignored — only `.gitkeep` is tracked
- Each tool function: query in (gene, variant, drug) → dict out

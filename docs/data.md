# Data

## Datasets (all free, <1GB total)

| Data Type | Dataset | Use |
|-----------|---------|-----|
| Genomics | ClinVar `variant_summary.txt` (~50MB) | Variant → disease lookup |
| Genomics + Transcriptomics | METABRIC from Kaggle (~50MB) | Mutations + mRNA z-scores + clinical |
| Pharmacology | PharmGKB clinical annotations | Drug-gene associations |
| Drug Safety | Kaggle Drug Side Effects (<50MB) | Adverse reactions |

## Patient Cases (from METABRIC)

- **Case 1**: PIK3CA + TP53 mutations, high ESR1, on tamoxifen → treatment optimization
- **Case 2**: BRCA1 variant, triple-negative, young → risk assessment + clinical trials
- **Case 3**: Multiple low-significance variants, conflicting signals → diagnostic dilemma

## Format

- All data as pandas DataFrames, cached as Parquet in `data/`
- Files organized per data source (e.g., ClinVar, METABRIC, PharmGKB)
- `data/` is gitignored — only `.gitkeep` is tracked
- Each tool function: query in (gene, variant, drug) → dict out

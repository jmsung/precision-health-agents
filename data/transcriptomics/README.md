# Diabetes Transcriptomics Dataset

## Source

**GSE26168** — "Type 2 Diabetes mellitus: mRNA and miRNA profiling"
- Platform: GPL6883 (Illumina HumanRef-8 v3.0 expression bead chip)
- Tissue: peripheral blood
- Published: Dec 2010 (GEO)
- PubMed: 21829658

## Samples

| Group | Count | Description |
|-------|-------|-------------|
| Control | 8 | Healthy controls |
| IFG | 7 | Impaired fasting glucose (pre-diabetic) |
| T2DM | 9 | Type 2 diabetes mellitus |

## Processed Dataset

`raw/diabetes_transcriptomics.csv` — 24 samples × 117 features

### Features
- `condition`: control / IFG / T2DM
- `condition_numeric`: 0 / 1 / 2
- 5 pathway scores (mean z-score of pathway genes):
  - `pathway_beta_cell_stress`
  - `pathway_inflammation_immune`
  - `pathway_insulin_resistance`
  - `pathway_fibrosis_ecm`
  - `pathway_oxidative_mitochondrial`
- 110 individual gene expression values

### Gene Panels (110 genes across 5 pathways)

| Pathway | Genes | Examples |
|---------|-------|----------|
| Beta cell stress | 20 | INS, PDX1, GCK, TCF7L2, ABCC8 |
| Inflammation/immune | 25 | TNF, IL6, IL1B, TLR4, NLRP3 |
| Insulin resistance | 25 | INSR, IRS1, AKT1, PPARG, FOXO1 |
| Fibrosis/ECM | 21 | COL1A1, TGFB1, MMP9, FN1, VIM |
| Oxidative/mitochondrial | 22 | SOD2, GPX1, SIRT1, UCP2, NFE2L2 |

## Processing

Run `scripts/process_transcriptomics.py` to regenerate from raw series matrix.

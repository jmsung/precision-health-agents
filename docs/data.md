# Data

## Datasets (all free, <1GB total)

| Data Type | Dataset | Location | Use |
|-----------|---------|----------|-----|
| Genomics | DNA Classification (Diabetes) | `data/dna_classification/` | CNN-based diabetes DNA classification (DMT1/DMT2/NONDM) |
| Clinical | Pima Indians Diabetes (Kaggle) | `data/diabetes/` | MLP clinical diabetes prediction (8 features) |
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

### Pima Indians Diabetes Dataset (`data/diabetes/`)

Source: [Kaggle — mathchi/diabetes-data-set](https://www.kaggle.com/datasets/mathchi/diabetes-data-set) (CC0 1.0)
Original source: National Institute of Diabetes and Digestive and Kidney Diseases (NIDDK)

| File | Description |
|------|-------------|
| `raw/diabetes.csv` | 768 patients, 8 clinical features, binary outcome |
| `models/mlp_diabetes.keras` | Pre-trained 2-layer MLP (75% test accuracy) |
| `models/scaler.npy` | StandardScaler mean + scale for inference |

Features: Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age

## Patient Cases

### Diabetes Precision Medicine Cases

- **Case 1 — Confirmed Diabetic**: Clinical positive + DMT2 DNA → hospital, Type 2 drugs (metformin)
- **Case 2 — DNA Override (early intervention)**: Clinical negative + DMT2 DNA → hospital despite clean labs
- **Case 3 — Clinical Override (avoid unnecessary treatment)**: Clinical positive + NONDM DNA → reconsider drugs, health trainer first
- **Case 4 — Type 1 vs Type 2 drug differentiation**: Two similar clinical profiles, DMT1 vs DMT2 DNA → completely different drug plans

## Format

- All data as pandas DataFrames, cached as Parquet in `data/`
- Files organized per data source (e.g., ClinVar, METABRIC, PharmGKB)
- `data/` is gitignored — only `.gitkeep` is tracked
- Each tool function: query in (gene, variant, drug) → dict out

# Data

## Datasets (all free, <1GB total)

| Data Type | Dataset | Location | Use |
|-----------|---------|----------|-----|
| Genomics | DNA Classification (Diabetes) | `data/dna_classification/` | CNN-based diabetes DNA classification (DMT1/DMT2/NONDM) |
| Clinical | Pima Indians Diabetes (Kaggle) | `data/diabetes/` | MLP clinical diabetes prediction (8 features) |
| Exercise | Best 50 Exercises (Kaggle) | `data/exercises/` | Exercise recommendation lookup |
| Fitness | Gym Members Exercise Dataset (Kaggle) | `data/gym_members/` | Health trainer evaluation + future ML |
| Transcriptomics | GSE26168 Blood Transcriptome (GEO) | `data/transcriptomics/` | Gene expression pathway analysis (110 genes, 24 samples) |
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

### Best 50 Exercises Dataset (`data/exercises/`)

Source: [Kaggle — prajwaldongre/best-50-exercise-for-your-body](https://www.kaggle.com/datasets/prajwaldongre/best-50-exercise-for-your-body)

| File | Description |
|------|-------------|
| `raw/exercises.csv` | 50 exercises with Name, Type, BodyPart, Equipment, Level, Description, Benefits, CaloriesPerMinute |

Types: Strength, Cardio, Flexibility, Plyometric
Body parts: Chest, Back, Shoulders, Arms, Core, Legs, Full Body
Equipment: Bodyweight, Dumbbell, Barbell, Machine
Levels: Beginner, Intermediate

### Gym Members Exercise Dataset (`data/gym_members/`)

Source: [Kaggle — valakhorasani/gym-members-exercise-dataset](https://www.kaggle.com/datasets/valakhorasani/gym-members-exercise-dataset) (Apache 2.0)

| File | Description |
|------|-------------|
| `raw/gym_members.csv` | 973 gym members with demographics, exercise habits, and workout type |

Key columns: Age, Gender, Weight (kg), Height (m), BMI, Workout_Type (Cardio/Strength/Yoga/HIIT), Experience_Level (1/2/3), Workout_Frequency (days/week), Session_Duration (hours)

Used for: 3-layer health trainer evaluation (`scripts/evaluate_health_trainer.py`) — experience level accuracy, workout type baseline, clinical constraint verification. Future: demographic ML model to improve workout type scoring.

### Diabetes Transcriptomics Dataset (`data/transcriptomics/`)

Source: [GEO GSE26168](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE26168) — "Type 2 Diabetes mellitus: mRNA and miRNA profiling"
Platform: GPL6883 (Illumina HumanRef-8 v3.0 expression bead chip)

| File | Description |
|------|-------------|
| `raw/GSE26168_series_matrix.txt` | Full expression matrix (24,526 probes × 24 samples) |
| `raw/GPL6883_probe_gene_map.tsv` | Probe-to-gene symbol annotation |
| `raw/diabetes_transcriptomics.csv` | Processed: 24 samples × 117 features (110 genes + 5 pathway scores + 2 metadata) |

Samples: 8 control, 7 IFG (impaired fasting glucose), 9 T2DM
Tissue: peripheral blood

110 genes curated across 5 diabetes-relevant pathway panels:
- Beta cell stress (20 genes): INS, PDX1, GCK, TCF7L2, ABCC8...
- Inflammation/immune (25 genes): TNF, IL6, IL1B, TLR4, NLRP3...
- Insulin resistance (25 genes): INSR, IRS1, AKT1, PPARG, FOXO1...
- Fibrosis/ECM (21 genes): COL1A1, TGFB1, MMP9, FN1, VIM...
- Oxidative/mitochondrial (22 genes): SOD2, GPX1, SIRT1, UCP2, NFE2L2...

Processing script: `scripts/process_transcriptomics.py`

## Patient Cases

### Diabetes Precision Medicine Cases

- **Case 1 — Confirmed Diabetic**: Clinical positive + DMT2 DNA → hospital, Type 2 drugs (metformin)
- **Case 2 — DNA Override (early intervention)**: Clinical negative + DMT2 DNA → hospital despite clean labs
- **Case 3 — Clinical Override (avoid unnecessary treatment)**: Clinical positive + NONDM DNA → reconsider drugs, health trainer first
- **Case 4 — Type 1 vs Type 2 drug differentiation**: Two similar clinical profiles, DMT1 vs DMT2 DNA → completely different drug plans
- **Case 5 — Health trainer referral**: DMT2 DNA (72% confidence) + Non-Diabetic clinical (44% probability, moderate risk) → HEALTH_TRAINER → classify_workout_type (Strength, Beginner) → personalised exercise plan with clinical reasoning

## Format

- All data as pandas DataFrames, cached as Parquet in `data/`
- Files organized per data source (e.g., ClinVar, METABRIC, PharmGKB)
- `data/` is gitignored by default, with exceptions for tracked datasets: `data/diabetes/`, `data/exercises/` (raw + README tracked; models gitignored for exercises)
- `data/gym_members/` is NOT tracked — download via `kaggle datasets download -d valakhorasani/gym-members-exercise-dataset`
- Each tool function: query in (gene, variant, drug) → dict out

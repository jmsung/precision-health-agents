# Diabetes Medications Database

## Overview
Curated medication database for the PharmacologyAgent, based on ADA Standards of Care 2024 and clinical pharmacology guidelines.

## File
- `raw/diabetes_medications.csv` — 16 medications across 8 drug classes

## Schema
| Column | Description |
|--------|-------------|
| name | Generic drug name |
| class | Pharmacological class (e.g., SGLT2 Inhibitor, GLP-1 RA) |
| mechanism | Mechanism of action |
| primary_subtype | Best-matched diabetes molecular subtype from transcriptomics |
| contraindicated_complications | Complications where this drug should be avoided |
| recommended_complications | Complications where this drug has proven benefit |
| route | Administration route (oral/injection) |
| monitoring | Required lab/clinical monitoring |
| common_side_effects | Frequent adverse effects |
| ada_first_line | Whether ADA recommends as first-line (true/false) |
| notes | Clinical pearls and usage guidance |

## Molecular Subtypes (from Transcriptomics)
- `inflammation_dominant` — GLP-1 RAs preferred (anti-inflammatory)
- `beta_cell_failure` — Insulin therapy essential
- `metabolic_insulin_resistant` — Metformin + SGLT2i/TZD
- `fibrotic_complication` — SGLT2i + ACEi for organ protection
- `mixed` — Combination approach, DPP-4i as bridge

## Sources
- ADA Standards of Care 2024
- EASD/ADA Consensus Report on T2DM Management
- KDIGO Guidelines (renal)
- ACC/AHA Guidelines (cardiovascular)

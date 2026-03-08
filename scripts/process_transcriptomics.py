"""Process GSE26168 series matrix into a curated diabetes transcriptomics dataset.

Downloads probe-to-gene mapping from Illumina GPL6883 platform,
filters to diabetes-relevant pathway genes, and outputs a clean CSV.

Pathway gene panels based on transcriptomic themes from diabetes literature:
1. Beta cell stress / insulin secretion
2. Inflammation & immune activation
3. Insulin resistance / signaling
4. Fibrosis & extracellular matrix remodeling
5. Oxidative & mitochondrial stress
"""

from __future__ import annotations

import csv
import gzip
import io
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data" / "transcriptomics"
RAW_DIR = DATA_DIR / "raw"
MATRIX_PATH = RAW_DIR / "GSE26168_series_matrix.txt"
PLATFORM_PATH = RAW_DIR / "GPL6883_probe_gene_map.tsv"
OUTPUT_PATH = DATA_DIR / "raw" / "diabetes_transcriptomics.csv"

# ---------------------------------------------------------------------------
# Diabetes-relevant gene panels (curated from literature)
# ---------------------------------------------------------------------------
PATHWAY_GENES: dict[str, list[str]] = {
    "beta_cell_stress": [
        "INS", "GCG", "PDX1", "NKX6-1", "MAFA", "SLC2A2", "GCK",
        "PCSK1", "PCSK2", "ABCC8", "KCNJ11", "SLC30A8", "TCF7L2",
        "NEUROD1", "PAX6", "IAPP", "CHGA", "CHGB", "UCN3", "ERO1B",
    ],
    "inflammation_immune": [
        "TNF", "IL6", "IL1B", "IL8", "CXCL8", "CCL2", "CCL5",
        "NFKB1", "RELA", "IKBKB", "TLR4", "TLR2", "NLRP3",
        "CASP1", "IL18", "IFNG", "IL10", "TGFB1", "CD68", "ITGAX",
        "CD14", "CRP", "SAA1", "SOCS3", "JAK2", "STAT3",
    ],
    "insulin_resistance": [
        "INSR", "IRS1", "IRS2", "PIK3CA", "PIK3R1", "AKT1", "AKT2",
        "SLC2A4", "PPARG", "PPARGC1A", "ADIPOQ", "LEP", "LEPR",
        "PTPN1", "GRB2", "SOS1", "MAPK1", "MAPK3", "GSK3B",
        "FOXO1", "SREBF1", "PCK1", "G6PC", "PRKAA1", "PRKAA2",
    ],
    "fibrosis_ecm": [
        "COL1A1", "COL3A1", "COL4A1", "FN1", "TGFB1", "TGFBR1",
        "TGFBR2", "SMAD2", "SMAD3", "SMAD4", "CTGF", "CCN2",
        "MMP2", "MMP9", "TIMP1", "TIMP2", "ACTA2", "VIM",
        "LOX", "SERPINE1", "THBS1", "SPARC",
    ],
    "oxidative_mitochondrial": [
        "SOD1", "SOD2", "CAT", "GPX1", "GPX4", "NFE2L2", "KEAP1",
        "HMOX1", "NQO1", "TXNRD1", "TXN", "PRDX1", "PRDX3",
        "MT-ND1", "MT-CO1", "MT-ATP6", "NDUFS1", "SDHA", "UQCRC1",
        "COX5A", "ATP5F1A", "UCP2", "PPARGC1A", "SIRT1", "SIRT3",
    ],
}

ALL_GENES = set()
for genes in PATHWAY_GENES.values():
    ALL_GENES.update(genes)

# Gene -> pathway mapping (a gene can appear in multiple pathways)
GENE_TO_PATHWAYS: dict[str, list[str]] = {}
for pathway, genes in PATHWAY_GENES.items():
    for gene in genes:
        GENE_TO_PATHWAYS.setdefault(gene, []).append(pathway)


def download_platform_annotation() -> pd.DataFrame:
    """Download GPL6883 probe-to-gene annotation from GEO."""
    if PLATFORM_PATH.exists():
        return pd.read_csv(PLATFORM_PATH, sep="\t")

    url = "https://ftp.ncbi.nlm.nih.gov/geo/platforms/GPL6nnn/GPL6883/annot/GPL6883.annot.gz"
    print(f"Downloading GPL6883 annotation from {url}...")
    response = urllib.request.urlopen(url)
    data = gzip.decompress(response.read()).decode("utf-8", errors="replace")

    # Parse the annotation file (skip comment lines starting with #)
    lines = data.split("\n")
    header_idx = None
    for i, line in enumerate(lines):
        if line.startswith("ID\t") or line.startswith('"ID"'):
            header_idx = i
            break

    if header_idx is None:
        raise ValueError("Could not find header in annotation file")

    # Read from header onwards
    content = "\n".join(lines[header_idx:])
    df = pd.read_csv(io.StringIO(content), sep="\t", low_memory=False)

    # Extract probe ID and gene symbol columns
    id_col = "ID"
    gene_col = None
    for col in df.columns:
        if "gene_symbol" in col.lower() or "symbol" in col.lower():
            gene_col = col
            break
        if "gene" in col.lower() and "assignment" in col.lower():
            gene_col = col
            break

    if gene_col is None:
        print("Available columns:", list(df.columns))
        raise ValueError("Could not find gene symbol column")

    print(f"Using columns: {id_col}, {gene_col}")
    result = df[[id_col, gene_col]].rename(columns={id_col: "probe_id", gene_col: "gene_symbol"})
    result.to_csv(PLATFORM_PATH, sep="\t", index=False)
    return result


def parse_series_matrix() -> tuple[pd.DataFrame, dict[str, str]]:
    """Parse GSE26168 series matrix into expression DataFrame and sample labels."""
    with open(MATRIX_PATH) as f:
        lines = f.readlines()

    # Extract sample titles for labels
    sample_labels = {}
    for line in lines:
        if line.startswith("!Sample_title"):
            parts = line.strip().split("\t")[1:]
            titles = [p.strip('"') for p in parts]
        if line.startswith("!Sample_geo_accession"):
            parts = line.strip().split("\t")[1:]
            accessions = [p.strip('"') for p in parts]

    for acc, title in zip(accessions, titles):
        if "control" in title:
            sample_labels[acc] = "control"
        elif "impaired" in title:
            sample_labels[acc] = "IFG"
        else:
            sample_labels[acc] = "T2DM"

    # Find expression data
    data_start = None
    data_end = None
    for i, line in enumerate(lines):
        if line.startswith("!series_matrix_table_begin"):
            data_start = i + 1
        if line.startswith("!series_matrix_table_end"):
            data_end = i

    # Parse expression matrix
    header = lines[data_start].strip().split("\t")
    header = [h.strip('"') for h in header]

    rows = []
    for line in lines[data_start + 1 : data_end]:
        parts = line.strip().split("\t")
        probe_id = parts[0].strip('"')
        values = [float(v) for v in parts[1:]]
        rows.append([probe_id] + values)

    df = pd.DataFrame(rows, columns=header)
    df = df.rename(columns={"ID_REF": "probe_id"})
    return df, sample_labels


def build_curated_dataset():
    """Build the final curated transcriptomics CSV."""
    print("Step 1: Parsing series matrix...")
    expr_df, sample_labels = parse_series_matrix()
    print(f"  Expression matrix: {expr_df.shape[0]} probes × {expr_df.shape[1] - 1} samples")
    print(f"  Groups: {dict(pd.Series(sample_labels).value_counts())}")

    print("\nStep 2: Loading probe-to-gene annotation...")
    annot_df = download_platform_annotation()
    print(f"  Annotation rows: {len(annot_df)}")

    print("\nStep 3: Mapping probes to genes...")
    # Merge expression with annotation
    merged = expr_df.merge(annot_df, on="probe_id", how="inner")
    print(f"  Merged rows: {len(merged)}")

    # Filter to our pathway genes
    # Gene symbol column may have multiple genes separated by " /// "
    def extract_matching_gene(symbol_str):
        if pd.isna(symbol_str):
            return None
        for gene in str(symbol_str).split(" /// "):
            gene = gene.strip()
            if gene in ALL_GENES:
                return gene
        return None

    merged["gene"] = merged["gene_symbol"].apply(extract_matching_gene)
    filtered = merged[merged["gene"].notna()].copy()
    print(f"  Pathway-matched probes: {len(filtered)}")
    print(f"  Unique genes found: {filtered['gene'].nunique()}")
    print(f"  Genes: {sorted(filtered['gene'].unique())}")

    # If multiple probes per gene, take the one with highest mean expression
    sample_cols = [c for c in filtered.columns if c.startswith("GSM")]
    filtered["mean_expr"] = filtered[sample_cols].mean(axis=1)
    filtered = filtered.sort_values("mean_expr", ascending=False).drop_duplicates(subset="gene", keep="first")
    filtered = filtered.drop(columns=["mean_expr"])
    print(f"  After dedup (best probe per gene): {len(filtered)}")

    print("\nStep 4: Building output CSV...")
    # Transpose: samples as rows, genes as columns
    gene_expr = filtered.set_index("gene")[sample_cols].T
    gene_expr.index.name = "sample_id"

    # Add metadata columns
    gene_expr.insert(0, "condition", gene_expr.index.map(sample_labels))
    gene_expr.insert(1, "condition_numeric", gene_expr["condition"].map(
        {"control": 0, "IFG": 1, "T2DM": 2}
    ))

    # Add pathway scores (mean z-score of genes in each pathway)
    gene_cols = [c for c in gene_expr.columns if c not in ("condition", "condition_numeric")]

    # Z-score normalize gene expression across samples
    for col in gene_cols:
        values = gene_expr[col].astype(float)
        mean, std = values.mean(), values.std()
        if std > 0:
            gene_expr[f"{col}_zscore"] = (values - mean) / std

    # Compute pathway scores
    for pathway, genes in PATHWAY_GENES.items():
        zscore_cols = [f"{g}_zscore" for g in genes if f"{g}_zscore" in gene_expr.columns]
        if zscore_cols:
            gene_expr[f"pathway_{pathway}"] = gene_expr[zscore_cols].mean(axis=1)

    # Drop individual z-score columns (keep raw + pathway scores)
    zscore_cols = [c for c in gene_expr.columns if c.endswith("_zscore")]
    gene_expr = gene_expr.drop(columns=zscore_cols)

    # Sort columns: metadata, pathway scores, then gene expression
    meta_cols = ["condition", "condition_numeric"]
    pathway_cols = sorted([c for c in gene_expr.columns if c.startswith("pathway_")])
    gene_cols = sorted([c for c in gene_expr.columns if c not in meta_cols + pathway_cols])
    gene_expr = gene_expr[meta_cols + pathway_cols + gene_cols]

    gene_expr.to_csv(OUTPUT_PATH)
    print(f"\nOutput saved to {OUTPUT_PATH}")
    print(f"Shape: {gene_expr.shape} (samples × features)")
    print(f"\nPathway columns: {pathway_cols}")
    print(f"\nSample summary:")
    print(gene_expr.groupby("condition")[pathway_cols].mean().round(3).to_string())


if __name__ == "__main__":
    build_curated_dataset()

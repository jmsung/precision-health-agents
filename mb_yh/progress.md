# Progress — YH (Bottom-Up: Data Pipeline, Tools, Agents, Testing)

## Active

## Hold

## Todo

### Phase 1: Data Pipeline
- [ ] Create `scripts/download_data.py` — download datasets (ClinVar, METABRIC, PharmGKB, Kaggle)
- [ ] Build 3 patient test cases from METABRIC/Kaggle data
- [ ] Cache datasets as Parquet in `data/`
- [ ] Validate data formats match `models.py` schemas

### Phase 2: Tools (all 6 domains)
- [ ] `tools/genomics_tools.py` — myvariant.info, ClinVar Entrez
- [ ] `tools/transcriptomics_tools.py` — GSEApy pathway enrichment
- [ ] `tools/proteomics_tools.py` — UniProt REST API
- [ ] `tools/pharma_tools.py` — DGIpy, OpenFDA, ChEMBL
- [ ] `tools/clinical_tools.py` — JSON knowledge base
- [ ] `tools/literature_tools.py` — PubMed Entrez, Semantic Scholar
- [ ] Unit tests for each tool module

### Phase 3: Agent Implementations
- [ ] **Genomics** agent — wire tools + system prompt
- [ ] **Transcriptomics** agent — wire tools + system prompt
- [ ] **Proteomics** agent — wire tools + system prompt
- [ ] **Pharmacology** agent — wire tools + system prompt
- [ ] **Clinical Guidelines** agent — wire tools + system prompt
- [ ] **Literature Review** agent — wire tools + system prompt
- [ ] System prompts: `prompts/{genomics,transcriptomics,proteomics,pharmacology,clinical,literature,synthesis}.txt`

### Phase 4: Dashboard & Demo
- [ ] `app/dashboard.py` — Tab 1: MDT Meeting
- [ ] Tab 2: Evaluation (score heatmap, aggregate metrics, latency/cost)
- [ ] Tab 3: Ralph Loop Timeline
- [ ] `scripts/demo.py` — pre-cached demo script
- [ ] Pre-cache API responses for demo patient
- [ ] Test full end-to-end flow on all 3 cases
- [ ] Prepare 2-min demo + write submission

## Completed

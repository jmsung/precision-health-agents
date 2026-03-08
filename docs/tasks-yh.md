# Tasks — YH

## Track: Clinical Agents + Demo

### Phase 1: Data Prep (10:00-10:30)
- [ ] Create `scripts/download_data.py` — download ClinVar, METABRIC, PharmGKB
- [ ] Build 3 patient test cases from METABRIC data
- [ ] Cache datasets as Parquet in `data/`

### Phase 2: Clinical Agents (10:30-2:00)
- [ ] **Pharmacology** agent + `tools/pharma_tools.py` (DGIpy, OpenFDA, ChEMBL)
- [ ] **Clinical Guidelines** agent + `tools/clinical_tools.py` (JSON knowledge base)
- [ ] **Literature Review** agent + `tools/literature_tools.py` (PubMed Entrez, Semantic Scholar)
- [ ] System prompts: `prompts/pharmacology.txt`, `clinical.txt`, `literature.txt`
- [ ] Write `prompts/synthesis.txt` for final report generation

### Phase 3: Dashboard (2:30-4:30)
- [ ] `app/dashboard.py` — Tab 1: MDT Meeting (patient select → run → agent cards → synthesis)
- [ ] Tab 2: Evaluation (score heatmap, aggregate metrics, latency/cost)
- [ ] Tab 3: Ralph Loop Timeline (iteration timeline, before/after, prompt viewer)

### Phase 4: Demo (4:30-6:15)
- [ ] `scripts/demo.py` — pre-cached demo script
- [ ] Pre-cache API responses for demo patient (fallback for network issues)
- [ ] Test full end-to-end flow on all 3 cases
- [ ] Prepare 2-min demo: problem → live run → eval dashboard → Ralph Loop improvement
- [ ] Write submission

### Sync Points
- **10:30** — Get BaseAgent interface + AgentResult schema from JS
- **12:30** — First 2-agent test (pharmacology + genomics)
- **2:30** — All 6 agents compile, first full pipeline
- **4:30** — Full demo with eval

### Agent Interface (match JS's base.py)
Each agent must implement:
- `get_system_prompt() -> str` — load from `prompts/` dir
- `get_tools() -> list[dict]` — Claude API tool schemas
- `analyze(query, blackboard) -> AgentResult` — inherited agentic loop

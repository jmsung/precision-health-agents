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
- [x] `tools/dna_classifier.py` — pre-trained 2-layer CNN (3-mer tokenization) for diabetes DNA classification (DMT1/DMT2/NONDM)
- [ ] `tools/genomics_tools.py` — myvariant.info, ClinVar Entrez (+ unit tests)
- [ ] `tools/transcriptomics_tools.py` — GSEApy pathway enrichment
- [ ] `tools/proteomics_tools.py` — UniProt REST API
- [ ] `tools/pharma_tools.py` — DGIpy, OpenFDA, ChEMBL
- [ ] `tools/clinical_tools.py` — JSON knowledge base
- [ ] `tools/literature_tools.py` — PubMed Entrez, Semantic Scholar
- [ ] Unit tests for each tool module

### Phase 3: Agent Implementations
- [x] **Doctor** agent — conversational intake → classify_diabetes tool → hospital/health_trainer recommendation
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

### Docs
- [x] `docs/vision.md` — rewritten around DNA-precision diabetes medicine; two key scenarios (false positive/negative); drug differentiation by DNA type
- [x] `docs/architecture.md` — added Doctor Agent sequential pipeline diagram; decision table (genomics × clinical → recommendation)
- [x] `docs/data.md` — added Pima Indians diabetes dataset; updated patient cases to match diabetes precision medicine story
- [x] `docs/demo.md` — rewritten around 4 concrete diabetes cases; updated priority tiers
- [x] `docs/doctor_agent.md` — new doc: agent purpose, API, output models, genomics integration, model details

### Data
- [x] Organized `dna_classification` dataset under `data/dna_classification/{raw,models}/` (CSV + FASTA + model weights)
- [x] Downloaded Pima Indians Diabetes dataset (Kaggle: mathchi/diabetes-data-set) → `data/diabetes/raw/diabetes.csv` (768 patients, 8 features); `data/diabetes/README.md` added

### Models
- [x] `src/bioai/models.py` — shared Pydantic data contract between agents and orchestrator
  - `GenomicsFindings` — predicted_class, confidence, probabilities, risk_level, interpretation
  - `AgentResult` — agent, status, findings, summary, error
  - `HealthAssessment` — orchestrator output aggregating all agent results
  - `RiskLevel` / `AgentStatus` enums

### Tools
- [x] `src/bioai/tools/dna_classifier.py` — DNA classifier tool wrapping pre-trained 2-layer CNN (3-mer tokenization, DMT1/DMT2/NONDM)
- [x] `src/bioai/tools/diabetes_classifier.py` — clinical diabetes risk tool wrapping pre-trained MLP; returns prediction, probability, risk_level (low/moderate/high)

### Agents
- [x] `src/bioai/agents/genomics.py` — GenomicsAgent wired with classify_dna tool; returns typed `AgentResult`
- [x] `src/bioai/agents/doctor.py` — DoctorAgent: multi-turn chat gathers 8 clinical features, calls classify_diabetes, returns Recommendation (hospital | health_trainer)

### Tests
- [x] `tests/test_dna_classifier.py` — 5 tests, all passing
  - `test_load_model` — model loads, output shape `(None, 3)`
  - `test_load_tokenizer` — tokenizer has valid 3-mer vocabulary
  - `test_classify_dna_output_structure` — keys, valid class, probs sum to 1
  - `test_classify_dna_confidence_matches_predicted_class` — confidence == prob of predicted class
  - `test_classify_dna_short_sequence` — short sequences handled correctly
- [x] `tests/test_genomics_agent.py` — 3 tests, all passing
  - `test_agent_calls_dna_classifier_tool` — agent invokes tool, returns typed GenomicsFindings
  - `test_agent_returns_summary` — agent includes Claude narrative in result
  - `test_agent_returns_error_on_failure` — API failure returns error status gracefully
- [x] `tests/test_doctor_agent.py` — 5 tests, all passing (mocked API)
  - `test_chat_returns_text_reply` — agent replies with string
  - `test_tool_called_and_findings_set` — tool invoked, findings populated
  - `test_low_risk_recommends_health_trainer` — low risk → health_trainer
  - `test_result_returns_agent_result` — result() returns valid AgentResult
  - `test_multi_turn_conversation` — multiple exchanges tracked correctly
- [x] `tests/test_diabetes_classifier.py` — 5 tests, all passing
  - `test_non_diabetic_low_risk` — healthy patient classifies as Non-Diabetic
  - `test_diabetic_high_risk` — high-risk profile classifies as Diabetic
  - `test_return_shape` — output has correct keys
  - `test_probability_range` — probability in [0, 1]
  - `test_risk_level_thresholds` — low/high risk boundaries correct

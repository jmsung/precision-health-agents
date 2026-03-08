# Demo Plan

## The Story (2 minutes)

The demo tells a concrete story that most people immediately understand:

> *"Your doctor says you have diabetes. Should you trust that? What if your DNA says otherwise?"*

We show two patients with the same clinical reading — and completely different genetic realities — getting two different, correct recommendations.

---

## Presentation Flow

1. **(15s) Problem** — Doctors diagnose diabetes from blood tests alone. But two patients with identical glucose and BMI can have completely different futures. DNA changes everything.

2. **(60s) Live Demo** — Two patients, same clinical numbers, different DNA:
   - Patient A: Clinical positive + DMT2 DNA + active pathways → confirmed → pharmacology → Type 2 drugs
   - Patient B: Clinical positive + NONDM DNA → genetic override → health trainer, avoid unnecessary drugs
   - Patient C: Clinical positive + DMT2 DNA + NO pathway activation → false positive → health trainer (3-layer validation catches it)

3. **(30s) The Drug Decision** — Patient A's DNA says DMT2, transcriptomics confirms with beta cell stress pathway. Pharmacology agent recommends metformin/GLP-1, not insulin.

4. **(15s) Architecture** — Three validation layers. Genomics + Doctor + Transcriptomics working together.

---

## Dashboard (Streamlit)

| Tab | Content |
|-----|---------|
| **Patient Intake** | Chat interface → Doctor Agent gathers clinical info conversationally |
| **DNA Analysis** | Genomics Agent result card — DMT1/DMT2/NONDM with confidence |
| **Decision** | Combined recommendation — hospital vs health trainer, with reasoning |
| **Drug Plan** | Pharmacology Agent — DNA-matched drug recommendations |
| **Evaluation** | Test case × agent score heatmap, latency/cost |

---

## Demo Patient Cases

### Case 1 — Confirmed Diabetic (Clinical + DNA agree)
- Clinical: glucose=160, BMI=31, age=42, pregnant×2, mother has diabetes
- DNA: DMT2 (high confidence)
- Decision: → **Hospital** (confirmed Type 2) → Metformin / GLP-1 agonists

### Case 2 — DNA Override: Early Intervention
- Clinical: glucose=95, BMI=24 (looks healthy)
- DNA: DMT2 (high confidence)
- Decision: → **Hospital** (genetic risk overrides clean labs — catch it early)

### Case 3 — Clinical Override: Avoid Unnecessary Treatment
- Clinical: glucose=148, BMI=33 (looks diabetic)
- DNA: NONDM (no genetic predisposition)
- Decision: → **Health Trainer** (clinical positive, but not genetically predisposed — lifestyle first)

### Case 4 — Transcriptomics False Positive Filter
- Clinical: glucose=155, BMI=30 (looks diabetic)
- DNA: DMT2 (high confidence)
- Genomics + Doctor → **Hospital** (confirmed)
- Transcriptomics: Gene expression shows **no diabetes pathway activation** (all z-scores near 0)
- Decision: → **Health Trainer** (false positive — molecular evidence doesn't support diabetes, avoid unnecessary drugs)

### Case 5 — Type 1 vs Type 2 Drug Differentiation
- Two patients, same clinical profile
- Patient X: DMT1 DNA → **Insulin therapy**
- Patient Y: DMT2 DNA → **Metformin + GLP-1**

---

## Priority Tiers

### P0 — MVP
- [x] `models.py` — shared Pydantic contracts
- [x] Genomics Agent — DNA classification (DMT1/DMT2/NONDM)
- [x] Doctor Agent — conversational intake → classify_diabetes → recommendation
- [x] Health Trainer Agent — exercise prescription with clinical rules (ADA 2023)
- [ ] `scripts/run.py` — CLI to run the full pipeline on a case
- [ ] Streamlit: Patient Intake + DNA Analysis + Decision tabs

### P1 — Should Have
- [ ] Pharmacology Agent — DNA-matched drug recommendations
- [x] Evaluation framework with LLM-as-judge (13/13 pass, 3 agents evaluated)
- [x] Ralph Loop (3 iterations — improved health_trainer + doctor prompts)
- [ ] Case 3 (clinical override) fully demonstrated in dashboard

### P2 — Nice to Have
- [x] Transcriptomics Agent — 3rd validation layer with false positive filter
- [ ] Remaining agents (Proteomics, Literature, Clinical Guidelines)
- [ ] Streamlit Ralph Loop tab
- [x] All 4 demo cases pre-cached (mock outputs saved)

### Explicitly Skip
- GPU-requiring models (ESM-2, DNABERT-2, AlphaFold)
- FHIR data formatting
- Synthea patient generation
- Database/persistent storage
- Auth/user management

---

## Verification Commands

```bash
uv run python scripts/run.py --case 1              # full two-agent pipeline
uv run python scripts/evaluate.py                   # eval suite
uv run python scripts/evaluate.py --ralph --iter 3  # Ralph Loop
uv run streamlit run app/dashboard.py               # dashboard
```

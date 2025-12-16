# Milestone 3 (ACL) — Ultra‑Detailed Evaluation Master Guide (Graph‑RAG)
**Goal of this document:** Make your team *evaluation‑proof*. It explains **every part** of the evaluation requirements and **every component** in your implemented system (FantasyTrivia) with:
- what it does,
- how it works in *your code*,
- what to present in slides,
- what to show in the live demo,
- what they can ask in Q&A,
- failure modes + recovery steps.

This is written to match the “don’t explain lab concepts” rule: we focus only on what you built and how it satisfies the rubric.

**Evaluation slot:** 45 minutes (team)
- **Team Presentation (15%)**: **18–22 minutes** (strict)
- **Individual Q&A (15%)**: remaining time; **only the asked member answers**

**Project in this workspace:** *FantasyTrivia* — Graph‑RAG for Fantasy Premier League (FPL)

---

## 0) Non‑Negotiables (Rules + How to Avoid Losing Points)

### 0.1 What they are grading (translate guideline → checklist)
You must demonstrate **all** of the following **live**:

1) **Fully integrated pipeline**
   - Input → preprocessing → retrieval (baseline + embeddings) → context → LLM → answer → UI.

2) **Input preprocessing** includes (must be shown):
   - System overview of preprocessing outputs
   - Intent classification
   - Entity extraction
   - Input embedding (if used in your retrieval)
   - Error analysis + improvement attempts

3) **Graph retrieval layer**
   - **Baseline:** Cypher queries
     - at least **10 query templates** that answer **10 questions**
     - extracted entities must be used to query the KG
   - **Embeddings:** choose **one approach** and compare **≥2 embedding models**
     - You chose **feature vector embeddings** and implemented **two models**:
       - numerical feature vectors
       - text embeddings

4) **LLM layer**
   - combines baseline + embedding results into one context
   - uses structured prompt: **context + persona + task/instructions**
   - compares **≥3 models**
   - comparison includes **quantitative + qualitative** impressions

5) **UI**
   - user can type a question (and/or select a question)
   - user can view KG‑retrieved context
   - user can view final LLM answer
   - full backend integration
   - UI stays functional after answering (multiple questions in a row)

### 0.2 What NOT to do (to avoid auto‑deductions)
- Don’t teach lab material (what is Neo4j, what is RAG). They already know.
- Don’t do intro, motivation, related work.
- Don’t show only a diagram or only text.
- Don’t demo isolated pieces.

### 0.3 Live demo stability rules (how to prevent “random failure”)
Before you walk into evaluation:
- Run the app on a safe port (Windows sometimes blocks 8501):
  - `streamlit run app.py --server.port 9090`
- Confirm the app opens in browser.
- Run at least two questions end‑to‑end.
- Confirm `config.txt` credentials are correct.

---

## 1) System Architecture (Exactly What You Built)

### 1.1 One‑slide pipeline (say this in 20 seconds)
“The user asks a natural language FPL question in Streamlit. We preprocess the input (intent, entities, embedding), then retrieve information from the Neo4j knowledge graph using **two retrieval strategies**: (1) baseline Cypher queries and (2) embedding similarity search. We combine both into a structured context and pass it into an LLM prompt with persona + instructions. The UI shows both the retrieved context and the final answer, and remains interactive after each query.”

### 1.2 Code map (always know where things are)
- UI: `app.py`
- Preprocessing: `src/preprocessing.py` (`InputPreprocessor`)
- Retrieval orchestration: `src/retrieval.py` (`GraphRetriever`)
- Baseline Cypher templates: `src/cypher_queries.py` (`CypherQueryLibrary`)
- Embedding creation + storage: `src/create_embeddings.py` (`PlayerEmbedder`)
- LLM layer: `src/llm_layer.py` (`LLMLayer`)

### 1.3 Ground truth: what your pipeline returns
For each user query, your backend returns a single structured dict:
- `intent`: one label
- `cypher_results`: list of dicts
- `similar_players`: list of dicts (player + similarity)
- `embedding_type`: `text` or `numerical`
- `query`: original user string

This is the “integration evidence” you show in demo (because it proves baseline + embeddings both executed).

---

## 2) Component Ownership (4 Team Members)
Fill this table and keep it in your slides (one small slide is enough):

| Component | Owner | Must explain in Q&A |
|---|---|---|
| 1) Input Preprocessing | ________ | intent + entity extraction + embedding + errors |
| 2) Graph Retrieval Layer | ________ | baseline Cypher + embeddings + hybrid retrieval |
| 3) LLM Layer | ________ | prompt structure + 3-model comparison |
| 4) UI + Integration | ________ | Streamlit modes + backend calls + stability |

**Rule:** In Q&A, only the asked person answers.

---

## 3) Component 1 — Input Preprocessing (Everything You Must Know)

### 3.1 Purpose (what preprocessing must achieve)
Preprocessing must convert **raw language** into **structured signals** usable by retrieval:
- intent: which type of question is it?
- entities: which player/team/position/season/stat is being referenced?
- embedding: a semantic representation of the query text

In your code this is: `InputPreprocessor.preprocess(query)`.

### 3.2 Intent classification (implementation detail)
**File:** `src/preprocessing.py`  
**Method:** `classify_intent(query: str) -> str`

**Algorithm**
1) Lowercase the query.
2) For each intent category, count how many intent keywords appear as substrings.
3) Pick the intent with the maximum score.
4) If none match, return `general_question`.

**Intent labels you support (9)**
- `player_stats`
- `team_query`
- `top_players`
- `fixture_query`
- `position_query`
- `season_query`
- `comparison`
- `recommendation`
- `general_question`

**What to say in evaluation (why this is valid)**
- Deterministic + fast + no external dependency.
- Debuggable under evaluation conditions.
- Handles the required scope of question types for the chosen task.

**Typical exam question + best answer**
Q: “Why didn’t you use an LLM for intent classification?”
A: “We chose a rule‑based classifier for deterministic behavior and low latency. It reduces failure modes (rate limits / prompt drift). We validated coverage by testing example queries across all intent types and iteratively expanding keywords based on observed misclassifications.”

### 3.3 Entity extraction (implementation detail)
**File:** `src/preprocessing.py`  
**Method:** `extract_entities(query: str) -> Dict[str, List[str]]`

Your extracted fields:
- `players`: list of detected player name strings
- `teams`: currently present but not fully populated by heuristics (your retrieval relies mostly on players/positions/seasons/stats)
- `positions`: normalized to Neo4j position labels (`GK`, `DEF`, `MID`, `FWD`)
- `seasons`: from known seasons list
- `stats`: goals/assists/points/clean sheets/saves/minutes
- `numbers`: digits extracted from query

**Player name extraction logic (key improvement)**
- Split query into words.
- Scan left‑to‑right.
- When a word starts with a capital letter, collect it and all consecutive capitalized words as a “name candidate”.
- If candidate length ≥ 2, add to players list.
- If no multi‑word players found, fallback: single capitalized words > 3 chars excluding “Tell/Show/Who/Which/What/The”.

**What to show in slides (fast proof)**
- Example:
  - Input: “Tell me about Erling Haaland’s performance”
  - Output entities: `players=["Erling Haaland"]`

**Known limitations (say them confidently)**
- If user types everything lowercase (e.g., “haaland”), the multi‑word name heuristic may fail.
- Team extraction isn’t as strong as player extraction (team queries depend on how team names are referenced).

**Mitigations you can mention (without claiming you implemented them)**
- Fuzzy matching in Cypher (`CONTAINS`) helps when partial names are extracted.
- Text embedding retrieval still returns semantically relevant players even when entity extraction is weak.

### 3.4 Input embedding (implementation detail)
**Model:** SentenceTransformer `all-MiniLM-L6-v2`  
**Dim:** 384  
**Method:** `embed_query(query)` → list[float]

**Critical stability detail:** `device='cpu'` is used in model initialization to avoid device/meta tensor issues and keep evaluation reproducible.

### 3.5 Error analysis & improvement attempts (mandatory)
You must have 1 slide with “issue → evidence → fix”. Use these (they are real to your project):

1) **Entity extraction bug**
- Symptom: “Tell me about Erling Haaland” failed to extract player.
- Root cause: earlier logic missed names at sentence start.
- Fix: consecutive capitalized word scan across the whole sentence.

2) **Least/most handling**
- Symptom: “fewest/least/lowest” was misrouted.
- Fix: added `least/fewest/lowest/worst` keywords to `top_players` intent.

---

## 4) Component 2 — Graph Retrieval Layer (Baseline + Embeddings)

### 4.1 Retrieval responsibilities (what you must defend in Q&A)
Your retrieval layer must:
1) interpret intent/entities from preprocessing,
2) execute the correct Cypher template (baseline),
3) execute embedding similarity search (semantic),
4) combine both into one output for the UI + LLM.

### 4.2 Baseline retrieval (Cypher)
**File:** `src/retrieval.py`  
**Method:** `baseline_retrieval(user_query) -> (intent, results)`

**Execution sequence (say it like this)**
1) `preprocess()` returns `(intent, entities, embedding)`.
2) `get_query_for_intent(intent, entities)` selects a Cypher template.
3) `_run_cypher_query(cypher_query)` executes Neo4j query.
4) Return intent + list of rows.

**Evidence to show in demo**
- Print/expand “intent” and “cypher_results count”.
- Open “Show Source Data” to show actual rows.

### 4.3 Cypher query templates (≥10)
**File:** `src/cypher_queries.py`

You must show a table in slides: “question type → query function → what it returns”.

Minimum set you can safely present (10):
1) `get_top_scorers()` → `player, total_goals`
2) `get_top_assisters()` → `player, total_assists`
3) `get_top_points()` → `player, total_points`
4) `get_player_stats(name)` → aggregated matches/goals/assists/points/minutes/cards
5) `get_team_players(team)` → list of players
6) `get_top_clean_sheets(position)` → clean sheets leaders
7) `get_player_fixtures(name)` → gameweek + opponent + performance
8) `compare_players(p1,p2)` → side-by-side aggregated stats
9) `get_position_distribution(team)` → counts by position
10) `get_most_valuable_players(position)` → points efficiency

**Robustness improvements you must highlight**
- Player matching is fuzzy:
  - `get_player_stats()` uses `p.player_name CONTAINS '<name>'` + `LIMIT 1`
  - `compare_players()` uses `CONTAINS` and `LIMIT 2`

**Common failure cases (and what to do live)**
- If a query returns 0 rows:
  1) re-ask using a clearer name (“Mohamed Salah” not “Salah”), OR
  2) use embedding toggle to show semantic results still work, OR
  3) show the “Search by name” query (if you demo it).

### 4.4 Embedding-based retrieval (your selected approach + 2 models)

#### 4.4.1 Chosen approach (state it exactly)
You chose **feature vector embeddings** (not node embeddings).

This choice is defensible because:
- Player performance is naturally represented as a numerical feature vector.
- Similarity queries (cosine) become meaningful and explainable.

#### 4.4.2 Embedding Model A — Numerical feature vectors
**File:** `src/create_embeddings.py`  
**Method:** `create_numerical_embeddings(player_stats)`

Features used (13):
- total_points, total_goals, total_assists, total_minutes
- clean_sheets, saves, bonus, bps
- matches_played, influence, creativity, threat, ict_index

Normalization (explain with one line):
For each feature dimension, you apply min–max scaling:

$$x' = \frac{x - \min(x)}{\max(x) - \min(x)}$$

Why normalization matters:
- prevents “minutes” from dominating similarity purely due to scale.

#### 4.4.3 Embedding Model B — Text embeddings
**File:** `src/create_embeddings.py`  
**Method:** `create_text_embeddings(player_stats)`

You generate a natural language description per player (matches, points, goals, assists, etc.), embed it using SentenceTransformer, and store `text_embedding`.

What to say:
- Numerical embeddings capture *performance profiles*.
- Text embeddings capture *semantic query matching* (“good midfielder for points”).

#### 4.4.4 Where embeddings are stored (critical to mention)
`store_embeddings_in_neo4j()` sets node properties:
- `p.numerical_embedding` (vector)
- `p.text_embedding` (vector)
and creates vector indexes:
- `player_numerical_index`
- `player_text_index`

#### 4.4.5 How embedding retrieval runs at query time
**File:** `src/retrieval.py`
- `embedding_retrieval_text(user_query)`:
  - embeds the query
  - cosine similarity between query embedding and each player’s `text_embedding`
- `embedding_retrieval_numerical(user_query)`:
  - extracts a player name
  - compares that player’s numerical embedding to others

### 4.5 Hybrid retrieval (baseline + embeddings combined)
**File:** `src/retrieval.py`  
**Method:** `hybrid_retrieval(user_query, use_text_embeddings=True)`

This is your “integration proof”. In demo you show:
- baseline results (structured facts)
- embedding results (similar players)
- both are visible in UI.

---

## 5) Component 3 — LLM Layer (Prompting + 3-model comparison)

### 5.1 What your LLM layer is responsible for
1) build a **grounded prompt** using retrieved context,
2) run multiple models,
3) compare outputs quantitatively + qualitatively.

### 5.2 Structured context construction
**File:** `src/retrieval.py`  
**Method:** `format_context_for_llm(retrieval_results)`

It includes:
- query
- intent
- top 10 Cypher rows (formatted)
- similar players with similarity scores

This is exactly what you show in the UI under “Show Retrieved Context”.

### 5.3 Prompt structure (persona + context + task)
**File:** `src/llm_layer.py`  
**Method:** `create_prompt(context, question)`

Your prompt is evaluation-compliant:
- **Persona:** FPL expert assistant
- **Context:** KG results + embeddings
- **Task/Instructions:** use only context, cite stats, say when info is missing

### 5.4 3-model comparison (what to present)
**File:** `src/llm_layer.py`  
**Models:**
- FLAN‑T5 Large
- Falcon 7B Instruct
- Phi‑2

#### Quantitative (required)
Use the metrics you already produce:
- `time_seconds` (latency)
- `success` / `error` (availability)

#### Qualitative (required)
Use a scoring rubric (1–5). Put this table in slides:

| Model | Faithfulness | Specificity | Readability | Conciseness | Notes |
|---|---:|---:|---:|---:|---|
| FLAN‑T5 |  |  |  |  |  |
| Falcon |  |  |  |  |  |
| Phi‑2 |  |  |  |  |  |

**How to fill it live (fast)**
- Faithfulness: does it invent stats not present in context?
- Specificity: does it cite numbers from cypher results?
- Readability: clean structure, bullet points.
- Conciseness: avoids long irrelevant paragraphs.

#### Known LLM limitations (say them early)
- HF inference can rate-limit or fail without a token.
- Mitigation: token input + ability to re-run + focus on integration.

---

## 6) Component 4 — UI + Integration (Streamlit)

### 6.1 Required UI items (map to your UI)
Guideline requirement → where it appears:
- “User can write question” → Chat mode text box
- “View KG context” → Show Retrieved Context toggle
- “View final answer” → answer output panel
- “Select a question” → Example Questions expander (and recent query history)
- “Integration with backend” → retriever/LLM called from UI
- “Functional after answer” → ask multiple queries, history persists

### 6.2 Modes (what each proves)
1) **Chat** proves full pipeline integration.
2) **Compare Embeddings** proves you have two embedding approaches and can show differences.
3) **Graph Stats** proves live Neo4j connectivity and graph content.

### 6.3 Session state + logging (what to say)
- Uses session state to keep retriever and LLM objects initialized.
- Logs queries to `query_log.txt` (evidence of usage + evaluation trace).

### 6.4 Demo plan (the safe sequence)
Always demo in this order:
1) Chat: player stats query
2) Chat: top scorers query
3) Toggle embeddings (text vs numerical) and show “similar players” changes
4) Compare Embeddings mode
5) Graph Stats mode

---

## 7) Error Analysis & Improvements (Mandatory Slide)
Use “Issue → Root Cause → Fix → Outcome”. Keep it short but specific.

Recommended 4 items:
1) Name extraction failure → improved algorithm → now extracts full names.
2) Exact match no results → `CONTAINS` fuzzy matching → higher hit rate.
3) Missing dependencies (plotly) → requirements install → app runs.
4) WinError 10013 port blocked → run on port 9090 → stable live demo.

---

## 8) Presentation Script (18–22 minutes, no overruns)
Use this timeline exactly.

0:00–2:00 Architecture
2:00–4:00 Preprocessing
4:00–7:00 Baseline retrieval (10+ Cypher templates)
7:00–10:00 Embeddings (two models + why)
10:00–14:00 LLM layer (prompt + 3 models + comparison)
14:00–16:00 Error analysis
16:00–21:00 Live demo
21:00–22:00 Buffer + transition to Q&A

---

## 9) Individual Q&A (Model Answers + What to Open)

### Component 1 (Preprocessing)
Open: `src/preprocessing.py`
- Explain intent scoring and why it’s stable.
- Explain entity extraction with one example and one limitation.
- Explain why CPU embeddings.

### Component 2 (Retrieval)
Open: `src/retrieval.py`, `src/cypher_queries.py`, `src/create_embeddings.py`
- Walk through baseline flow.
- Show two Cypher templates and what they return.
- Explain numerical vs text embeddings, min-max scaling, cosine similarity.

### Component 3 (LLM)
Open: `src/llm_layer.py`
- Show prompt and justify “use only context”.
- Explain the 3-model comparison method and metrics.

### Component 4 (UI)
Open: `app.py`
- Explain modes and why each exists.
- Show session state and query logging.

---

## 10) Commands (Copy/Paste)

Activate venv:
- `.venv\Scripts\Activate.ps1`

Install deps:
- `python -m pip install -r requirements.txt`

Run app (safe port):
- `streamlit run app.py --server.port 9090`

---

## 11) PDF Export

### Option A (always works)
Open `M3_Evaluation_Master_Guide.html` → Ctrl+P → Save as PDF.

### Option B (Opera GX/Edge/Chrome headless)
Run:
- `PowerShell -NoProfile -ExecutionPolicy Bypass -File .\tools\export_master_guide_pdf.ps1`

---

**End of ultra‑detailed guide.**
- Conciseness

**Important limitation to state**
- HuggingFace serverless calls may rate‑limit or fail without a token. Your UI supports a demo mode and token input.

---

## 6) Component 4 — UI + Integration (Streamlit)

### 6.1 What the UI must show (per guideline)
- Use case is reflected (FPL Q&A).
- User can write a question.
- UI shows KG‑retrieved context.
- UI shows final LLM answer.
- UI remains functional after answering.
- UI integrates the pipeline backend.

### 6.2 Your UI structure
File: `app.py`
- Modes:
  1) **💬 Chat**: full pipeline
  2) **🔬 Compare Embeddings**: side‑by‑side similarity results
  3) **📊 Graph Stats**: counts + plots from Neo4j
- Sidebar controls:
  - Toggle text vs numerical embeddings
  - Show context
  - Show source data
  - Model selector (includes mock)
  - Token input
- Query logging: appends to `query_log.txt`

### 6.3 “Still functional after answer” proof
- Ask 3–4 different questions sequentially.
- Show sidebar “Recent Queries” updates.

---

## 7) Error Analysis & Improvements (Mandatory Slide)
You must present errors + fixes as *engineering* proof, not as drama.

### 7.1 Must‑mention issues you already hit
- **No results due to exact name matching** → switched to `CONTAINS`.
- **Entity extraction missed names** → consecutive‑capitalized extraction.
- **Runtime dependency missing** (e.g., plotly) → install via requirements.
- **Port binding blocked (WinError 10013)** → run Streamlit on a higher port.

### 7.2 “Fault‑proof” operating checklist (prevents live demo disasters)
Before evaluation:
- Confirm dependencies installed: `pip install -r requirements.txt`
- Confirm Streamlit launches on a safe port:
  - If port 8501 blocked: `streamlit run app.py --server.port 9090`
- Confirm Neo4j credentials in `config.txt`
- Confirm internet access (for HF models); keep mock fallback ready
- Run one test question end‑to‑end

---

## 8) Presentation Script (18–22 Minutes) — Exact Timing
Use this exactly; do not improvise on timing.

### 0:00–2:00 — High‑Level Architecture
- Show pipeline diagram.
- State: hybrid retrieval (Cypher + embeddings), structured prompt, UI integration.

### 2:00–4:00 — Input Preprocessing
- Intent classifier: keyword scoring.
- Entity extraction: consecutive capitalized names + FPL positions.
- Embedding: SentenceTransformer.
- Mention 1 improvement.

### 4:00–7:00 — Baseline Retrieval (Cypher)
- Show table of 10+ query templates.
- Show 1–2 query snippets.
- Mention fuzzy matching improvement.

### 7:00–10:00 — Embedding Retrieval
- State: feature vector embeddings.
- Compare numerical vs text embeddings.
- Show one similarity result example.

### 10:00–14:00 — LLM Layer
- Context construction.
- Prompt structure (persona/context/task).
- 3‑model comparison: speed + qualitative rubric.

### 14:00–16:00 — Error Analysis & Improvements
- 4 bullets: issue → fix.

### 16:00–21:00 — Live Demo (must be live)
**Demo order (repeatable):**
1) Start Chat mode.
2) Ask: “Tell me about Erling Haaland’s performance” → show intent/entities, cypher results, similar players, answer.
3) Ask: “Who scored the most goals in 2022‑23?” → show top list.
4) Switch embeddings toggle and re‑ask a query → show changed similar players.
5) Compare Embeddings mode: type a known star, show charts.
6) Graph Stats mode: show counts + one chart.

Stop by 21–22 minutes.

---

## 9) Individual Q&A — What Each Member Must Know

### Component 1 (Preprocessing) — likely questions
- Walk through `classify_intent()`; why rule‑based.
- Show entity extraction; why it handles names at sentence start.
- What happens with typos / lowercase names.
- Why this embedding model; why CPU.

### Component 2 (Retrieval) — likely questions
- How do you choose the Cypher query from intent/entities.
- Show at least 2 templates.
- How do you combine baseline + embeddings.
- Why feature vectors (not node embeddings).
- Explain normalization and cosine similarity.

### Component 3 (LLM) — likely questions
- Explain the prompt structure.
- How do you ensure faithfulness (answer only from context).
- Show how you measure time and compare models.
- What are failure modes (rate limit, model errors) and your fallback.

### Component 4 (UI) — likely questions
- How the UI calls retriever and LLM.
- Why session state is used.
- How you keep UI functional after an answer.
- What happens if Neo4j fails / internet fails.

---

## 10) “No Room for Error” — Final Day Checklist

### 10.1 Technical checklist (do this before you leave home)
- Start venv.
- Run Streamlit on a known‑working port:
  - `streamlit run app.py --server.port 9090`
- Open `http://localhost:9090`.
- Run 2 questions end‑to‑end.
- Confirm Neo4j connection works.
- Keep `config.txt` correct.

### 10.2 Evaluation behavior checklist
- Everyone knows their component boundaries.
- Only the asked person answers.
- If something breaks: state the issue, apply the prepared fix, continue.

---

## 11) Commands (Copy/Paste)

### Activate venv (PowerShell)
- `.venv\Scripts\Activate.ps1`

### Install dependencies
- `python -m pip install -r requirements.txt`

### Run app (safe port)
- `streamlit run app.py --server.port 9090`

---

## 12) Export to PDF (Reliable Method)
You have two options:

### Option A (always works): Browser print
1) Open the HTML file (we will generate it): `M3_Evaluation_Master_Guide.html`
2) Press **Ctrl+P** → choose **Save as PDF**

### Option B (one command): Edge headless export
Run the script we will generate: `tools\export_master_guide_pdf.ps1`

---

**End of document.**

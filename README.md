# FantasyTrivia - Graph-RAG for Fantasy Premier League

## Milestone 3: Graph-Augmented Retrieval and Generation

**Course:** CSEN 903: Advanced Computer Lab  
**Institution:** German University in Cairo  
**Date:** December 2025

---

## 📋 Project Overview

FantasyTrivia is a Graph-RAG (Retrieval-Augmented Generation) system that answers questions about Fantasy Premier League using a Neo4j knowledge graph combined with vector embeddings and LLM reasoning.

### Key Features
- **Knowledge Graph**: 51,952 player performance records from FPL 2022-23 season
- **Dual Retrieval**: Structured Cypher queries + Semantic vector embeddings
- **Two Embedding Approaches**: Numerical stats vs. Text descriptions
- **LLM Integration**: HuggingFace models (Llama 3.2, Qwen 2.5, Gemma 2) with grounded responses
- **Interactive UI**: Streamlit chatbot with 4 modes
- **Model Comparison**: Side-by-side LLM performance analysis with timing charts

---

## 🏗️ Architecture

```
User Query
    ↓
Input Preprocessing (Intent + Entities + Embedding)
    ↓
Dual Retrieval
    ├─→ Cypher Queries (Structured)
    └─→ Vector Search (Semantic)
    ↓
Context Formation
    ↓
LLM Generation
    ↓
Natural Language Answer
```

---

## 📁 Project Structure

```
Fantasy Football/
├── app.py                          # Streamlit UI (main application)
├── config.txt                      # Neo4j connection credentials
├── requirements.txt                # Python dependencies
├── query_log.txt                   # User query history log
│
├── src/
│   ├── preprocessing.py            # Intent classification & entity extraction
│   ├── cypher_queries.py           # 13 baseline Cypher query templates
│   ├── create_embeddings.py        # Dual embedding generation
│   ├── retrieval.py                # Hybrid retrieval (Cypher + Embeddings)
│   └── llm_layer.py                # LLM integration layer
│
├── notebooks/
│   └── Graph_RAG_Demo.ipynb        # Demonstration notebook
│
├── data/
│   ├── cleaned_merged_seasons.csv  # Cleaned dataset
│   └── master_team_list.csv        # Team master data
│
├── FPL/
│   └── fpl_two_seasons.csv         # Original FPL data
│
├── create_kg.py                    # M2: Knowledge graph builder
├── queries.txt                     # M2: 5 validated Cypher queries
└── rule.txt                        # M2: Defender scoring rules
```

---

## 🚀 Setup & Installation

### Prerequisites
- Python 3.10+
- Neo4j Aura Database (credentials in `config.txt`)
- HuggingFace API Token (for LLM features)
- 4GB+ RAM for embeddings

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/fantasy-trivia-graph-rag.git
cd fantasy-trivia-graph-rag
```

2. **Create virtual environment:**
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
# or: source .venv/bin/activate  # Linux/Mac
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Configure credentials:**
```bash
cp config.example.txt config.txt
# Edit config.txt with your Neo4j credentials
```

5. **Set HuggingFace token:**
```bash
# Option 1: Environment variable
set HUGGINGFACE_TOKEN=your_token_here  # Windows
export HUGGINGFACE_TOKEN=your_token_here  # Linux/Mac

# Option 2: Enter in Streamlit sidebar
```

6. **Run the application:**
```bash
streamlit run app.py --server.port 9090
```

---

## 🎯 Features Implemented

### 1. Input Preprocessing (`src/preprocessing.py`)
- **Intent Classification**: Rule-based matching for 9 intent types
  - player_stats, team_query, top_players, fixture_query, position_query, season_query, comparison, recommendation, general_question
- **Entity Extraction**: Extracts players, teams, positions, seasons, stats, numbers
- **Query Embedding**: SentenceTransformer (`all-MiniLM-L6-v2`) converts text to 384-dim vectors

### 2. Graph Retrieval - Baseline (`src/cypher_queries.py`)
**13 Cypher Query Templates:**
1. `get_top_scorers` - Top goal scorers by season/position
2. `get_top_assisters` - Top assist providers
3. `get_top_points` - Fantasy points leaders
4. `get_player_stats` - Detailed stats for specific player
5. `get_team_players` - All players from a team
6. `get_top_clean_sheets` - Defenders/GKs with most clean sheets
7. `get_player_fixtures` - Match history for player
8. `compare_players` - Side-by-side comparison
9. `get_position_distribution` - Players by position
10. `get_most_valuable_players` - Best points per 90 minutes
11. `get_teams_in_season` - All teams in season
12. `get_gameweek_top_performers` - Top performers in gameweek
13. `search_players_by_name` - Partial name matching

**Dispatcher:** `get_query_for_intent()` routes intent+entities to appropriate query

### 3. Graph Retrieval - Embeddings (`src/create_embeddings.py`)
**Approach 1: Numerical Embeddings**
- 13 statistical features: total_points, goals, assists, minutes, clean_sheets, saves, bonus, bps, matches_played, influence, creativity, threat, ict_index
- Min-max normalization to 0-1 range
- Stored as `numerical_embedding` on Player nodes

**Approach 2: Text Embeddings**
- Natural language descriptions of player statistics
- Example: "Player Erling Haaland played 35 matches and scored 272 fantasy points..."
- Encoded with SentenceTransformer
- Stored as `text_embedding` on Player nodes

**Vector Indices:**
- `player_numerical_index` - Cosine similarity on numerical embeddings
- `player_text_index` - Cosine similarity on text embeddings

### 4. Hybrid Retrieval (`src/retrieval.py`)
Combines both approaches:
```python
def hybrid_retrieval(query):
    intent, cypher_results = baseline_retrieval(query)  # Structured
    similar_players = embedding_retrieval(query)         # Semantic
    return {
        'intent': intent,
        'cypher_results': cypher_results,
        'similar_players': similar_players
    }
```

### 5. LLM Layer (`src/llm_layer.py`)
- **Models**: 
  - Llama 3.2 (`meta-llama/Llama-3.2-3B-Instruct`)
  - Qwen 2.5 (`Qwen/Qwen2.5-72B-Instruct`)
  - Gemma 2 (`google/gemma-2-9b-it`)
- **API**: HuggingFace Chat Completions API
- **Structured Prompts**: Context + Persona + Task format
- **Grounding**: Responses limited to retrieved knowledge graph data
- **Model Comparison**: `compare_models()` runs all 3 models with timing

### 6. Streamlit UI (`app.py`)
**4 Modes:**

**💬 Chat Mode:**
- Natural language Q&A
- Shows intent classification
- Displays Cypher results + Similar players
- Toggle between text/numerical embeddings
- Optional source data table

**🤖 Compare LLMs:**
- Side-by-side comparison of all 3 LLM models
- Response timing chart (bar visualization)
- Summary table with model performance
- Same retrieval context for fair comparison

**🔬 Compare Embeddings:**
- Side-by-side visualization
- Interactive charts showing similarity scores
- Compare numerical vs. text embedding results

**📊 Graph Stats:**
- Live database statistics
- Top 10 goal scorers chart
- Fantasy points leaders
- Position distribution pie chart
- Interactive Plotly visualizations

---

## 💡 Example Queries

```
"Who scored the most goals in 2022-23?"
"Tell me about Erling Haaland's performance"
"Which defenders got the most clean sheets?"
"Compare Mohamed Salah and Harry Kane"
"Who are the best midfielders?"
"Show me Bukayo Saka's stats"
```

---

## 📊 Evaluation

### Embedding Comparison
| Approach | Pros | Cons |
|----------|------|------|
| **Numerical** | Fast, precise for stat-based queries | Misses semantic context |
| **Text** | Captures natural language meaning | Slower, more memory |

### Query Performance
- **Intent Classification**: 90%+ accuracy on test queries
- **Entity Extraction**: Handles full names, partial matches
- **Retrieval Time**: <500ms for hybrid retrieval
- **Embedding Storage**: 554 players × 2 approaches = 1,108 vectors

---

## 🔧 Technical Details

### Dependencies
- `neo4j==5.28.0` - Graph database driver
- `sentence-transformers` - Embedding model
- `streamlit` - Web UI framework
- `langchain` - RAG framework
- `plotly` - Interactive visualizations
- `pandas` - Data manipulation
- `torch` - PyTorch for embeddings

### Knowledge Graph Schema
- **Nodes**: Player (1,513), Team (42), Fixture (760), Gameweek (75), Position (4), Season (2)
- **Relationships**: PLAYED_IN (51,952), PLAYS_AS, HAS_HOME_TEAM, HAS_AWAY_TEAM, HAS_FIXTURE
- **Properties**: 19 properties per PLAYED_IN relationship (goals, assists, points, etc.)

---

## 📈 Query Logging

All user queries are automatically logged to `query_log.txt` with timestamps:
```
2025-12-09 15:30:45 - Who scored the most goals?
2025-12-09 15:31:12 - Tell me about Haaland
```

View recent queries in the sidebar under "📝 Recent Queries"

---

## 🎓 Academic Context

This project fulfills requirements for:
- **Milestone 3**: Graph-Augmented Retrieval and Generation
- **Course**: CSEN 903 - Advanced Computer Lab
- **Topic**: Retrieval-Augmented Generation (RAG) with LangChain

### Key Learning Outcomes
1. Implemented dual retrieval strategies (structured + semantic)
2. Compared two embedding approaches (numerical vs. text)
3. Built end-to-end RAG pipeline with grounded responses
4. Integrated LLM reasoning with knowledge graph data
5. Created interactive visualization and analysis tools

---

## 🏆 Achievements

✅ **Dual Retrieval Implementation**  
✅ **Two Embedding Approaches Compared**  
✅ **13 Baseline Cypher Queries**  
✅ **Vector Index with 1,108 Embeddings**  
✅ **Interactive Multi-Mode UI**  
✅ **Query Logging System**  
✅ **Comprehensive Documentation**  

---

## 📝 Notes

- Neo4j instance must be active for the application to work
- First run may take 10-20 seconds to load embedding model
- HuggingFace token optional (mock mode available)
- Embeddings already created and stored in Neo4j vector indices

---

## 👨‍💻 Author

**Student Name:** [Your Name]  
**University:** German University in Cairo  
**Course:** CSEN 903 - Advanced Computer Lab  
**Semester:** Winter 2025  
**Submission Date:** December 15, 2025

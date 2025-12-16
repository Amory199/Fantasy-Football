"""
FantasyTrivia: Graph-RAG powered FPL Question Answering System
Streamlit UI combining Neo4j knowledge graph with LLM reasoning
"""

import streamlit as st
import sys
import os
sys.path.append('src')

from retrieval import GraphRetriever, load_config
from llm_layer import LLMLayer
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# page config
st.set_page_config(
    page_title="FantasyTrivia - FPL Q&A",
    page_icon="⚽",
    layout="wide"
)

# title
st.title("⚽ FantasyTrivia")
st.subheader("Graph-RAG powered Fantasy Premier League Q&A System")

# sidebar for settings
st.sidebar.header("Settings")

# Mode selection
app_mode = st.sidebar.radio(
    "Mode",
    ["💬 Chat", "🤖 Compare LLMs", "🔬 Compare Embeddings", "📊 Graph Stats"],
    help="Choose between chatbot, LLM comparison, embedding comparison, or graph statistics"
)

use_text_embeddings = st.sidebar.checkbox("Use Text Embeddings", value=True, 
    help="Use text-based embeddings instead of numerical stats embeddings")

show_context = st.sidebar.checkbox("Show Retrieved Context", value=True,
    help="Display the knowledge graph context used for answering")

show_sources = st.sidebar.checkbox("Show Source Data", value=False,
    help="Display the raw data retrieved from Neo4j")

st.sidebar.markdown("---")
st.sidebar.markdown("**FantasyTrivia** - Graph-RAG for FPL")
st.sidebar.caption("Neo4j + Embeddings + LLM")

# Query history
if st.session_state.get('query_history'):
    st.sidebar.markdown("---")
    with st.sidebar.expander("📝 Recent Queries"):
        for i, q in enumerate(reversed(st.session_state.query_history[-10:]), 1):
            st.caption(f"{i}. {q}")

# HuggingFace token input (optional)
hf_token = st.sidebar.text_input(
    "HuggingFace Token (Optional)",
    type="password",
    help=(
        "Provide your HF token for higher rate limits. "
        "If left empty, the app will use HUGGINGFACE_TOKEN/HF_TOKEN from your environment if set; "
        "otherwise it will use mock responses."
    ),
)

env_hf_token = (os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN") or "").strip()
sidebar_hf_token = (hf_token or "").strip()
effective_hf_token = sidebar_hf_token if sidebar_hf_token else env_hf_token
has_hf_token = bool(effective_hf_token)

if (not sidebar_hf_token) and env_hf_token:
    st.sidebar.success("Using HuggingFace token from environment (HUGGINGFACE_TOKEN/HF_TOKEN).")
elif not has_hf_token:
    st.sidebar.info("No HuggingFace token set — using mock responses.")

# model selection (for demonstration - needs HF token to work)
model_option = st.sidebar.selectbox(
    "LLM Model",
    ["llama (Llama 3.2)", "qwen (Qwen 2.5)", "gemma (Gemma 2)", "mock (Demo)"],
    index=3  # default to mock
)

# initialize session state
if 'retriever' not in st.session_state:
    config = load_config()
    st.session_state.retriever = GraphRetriever(config)

if ('llm' not in st.session_state) or (st.session_state.get('hf_token_last') != effective_hf_token):
    st.session_state.llm = LLMLayer(effective_hf_token if has_hf_token else None)
    st.session_state.hf_token_last = effective_hf_token

if 'query_history' not in st.session_state:
    st.session_state.query_history = []

# ============================================================================
# MODE 1: CHAT INTERFACE
# ============================================================================
if app_mode == "💬 Chat":
    # main query interface
    st.write("---")
    st.write("### Ask a question about Fantasy Premier League 2022-23 season:")

    # example questions
    with st.expander("📝 Example Questions"):
        st.markdown("""
        - Who scored the most goals in 2022-23?
        - Tell me about Erling Haaland's performance
        - Which defenders got the most clean sheets?
        - Compare Mohamed Salah and Harry Kane
        - Who are the best midfielders for fantasy points?
        - Which players from Arsenal had the most assists?
        - Show me Bukayo Saka's stats
        """)

    # query input
    user_query = st.text_input("Your question:", placeholder="e.g., Who scored the most goals in 2022-23?", key="chat_query_input")

    if user_query:
        # Record the query
        if user_query not in st.session_state.query_history:
            st.session_state.query_history.append(user_query)
            # Save to file
            with open('query_log.txt', 'a', encoding='utf-8') as f:
                from datetime import datetime
                f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {user_query}\n")
        
        with st.spinner("🔍 Searching knowledge graph..."):
            # get hybrid retrieval results
            retrieval_results = st.session_state.retriever.hybrid_retrieval(
                user_query,
                use_text_embeddings=use_text_embeddings
            )
            
            # format context
            context = st.session_state.retriever.format_context_for_llm(retrieval_results)
        
        # display retrieved context if enabled
        if show_context:
            st.write("---")
            st.write("### 📊 Retrieved Knowledge Graph Context")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Intent:** `{retrieval_results['intent']}`")
                st.write(f"**Cypher Results:** {len(retrieval_results['cypher_results'])} records")
                
                if retrieval_results['cypher_results']:
                    st.write("**Structured Data from Graph:**")
                    # show first 5 results in a nice format
                    for i, record in enumerate(retrieval_results['cypher_results'][:5], 1):
                        st.write(f"{i}. " + ", ".join(f"{k}: {v}" for k, v in record.items()))
                    
                    if len(retrieval_results['cypher_results']) > 5:
                        st.write(f"... and {len(retrieval_results['cypher_results'])-5} more")
            
            with col2:
                st.write(f"**Embedding Type:** `{retrieval_results['embedding_type']}`")
                st.write(f"**Similar Players:** {len(retrieval_results['similar_players'])} found")
                
                if retrieval_results['similar_players']:
                    st.write("**Semantically Similar Players:**")
                    for i, player in enumerate(retrieval_results['similar_players'], 1):
                        st.write(f"{i}. {player['player']} (similarity: {player['similarity']:.4f})")
        
        # show source data table
        if show_sources and retrieval_results['cypher_results']:
            st.write("---")
            st.write("### 📋 Source Data Table")
            df = pd.DataFrame(retrieval_results['cypher_results'][:20])
            st.dataframe(df, use_container_width=True)
        
        # generate answer
        st.write("---")
        st.write("### 🤖 LLM Answer")
        
        # check if using mock or real LLM
        if "mock" in model_option.lower() or not has_hf_token:
            # mock response for demonstration
            st.info("💡 Using mock LLM response for demonstration. Provide a HuggingFace token in the sidebar to use real models.")
            
            # create a simple rule-based mock answer
            if retrieval_results['cypher_results']:
                first_result = retrieval_results['cypher_results'][0]
                
                # generate a natural-sounding answer based on the query intent
                if retrieval_results['intent'] == 'top_players':
                    if 'player' in first_result and 'total_goals' in first_result:
                        answer = f"Based on the 2022-23 season data, **{first_result['player']}** scored the most goals with **{first_result['total_goals']} goals**."
                    elif 'player' in first_result and 'total_points' in first_result:
                        answer = f"Based on the data, **{first_result['player']}** led with **{first_result['total_points']} fantasy points**."
                    else:
                        answer = f"The top performer was **{first_result.get('player', 'Unknown')}**."
                
                elif retrieval_results['intent'] == 'player_stats':
                    player_name = first_result.get('player', 'the player')
                    stats = ', '.join(f"{k}: {v}" for k, v in first_result.items() if k != 'player')
                    answer = f"Here are the statistics for **{player_name}**: {stats}"
                
                else:
                    # generic answer
                    top_3 = retrieval_results['cypher_results'][:3]
                    answer = "Based on the knowledge graph, the top results are:\n\n"
                    for i, record in enumerate(top_3, 1):
                        answer += f"{i}. " + ", ".join(f"{k}: {v}" for k, v in record.items()) + "\n"
            else:
                answer = "I couldn't find specific data to answer your question in the knowledge graph. The semantic search found some related players, but no exact match for your query."
            
            st.success(answer)
            
            # show similar players from embedding search
            if retrieval_results['similar_players'] and len(retrieval_results['cypher_results']) == 0:
                st.write("\n**Related players found through semantic search:**")
                for player in retrieval_results['similar_players'][:3]:
                    st.write(f"- {player['player']}")
        
        else:
            # use real LLM
            with st.spinner(f"🧠 Generating answer with {model_option}..."):
                model_key = model_option.split()[0]  # extract model key
                result = st.session_state.llm.query_model(model_key, 
                    st.session_state.llm.create_prompt(context, user_query))
                
                if result['success']:
                    st.success(result['answer'])
                    st.caption(f"Response time: {result['time_seconds']}s")
                else:
                    st.error(f"Model error: {result['error']}")
                    st.info("Try using the mock model or provide a valid HuggingFace token.")

# ============================================================================
# MODE 2: COMPARE LLMs
# ============================================================================
elif app_mode == "🤖 Compare LLMs":
    st.write("---")
    st.write("### 🤖 Compare All LLM Models")
    st.write("Run a query through all 3 LLM models and compare their answers, response times, and quality.")
    
    if not has_hf_token:
        st.error("⚠️ HuggingFace token required for LLM comparison. Please enter your token in the sidebar.")
        st.info("You can get a free token at https://huggingface.co/settings/tokens")
    else:
        st.success(f"✅ HuggingFace token detected. Ready to compare models.")
        
        # Example questions
        with st.expander("📝 Example Questions"):
            st.markdown("""
            - Who scored the most goals in 2022-23?
            - Tell me about Erling Haaland's performance
            - Which defenders got the most clean sheets?
            - Compare Mohamed Salah and Harry Kane
            """)
        
        compare_query = st.text_input(
            "Enter your question:",
            placeholder="e.g., Who scored the most goals in 2022-23?",
            key="compare_llm_query"
        )
        
        if st.button("🚀 Compare All Models", type="primary") and compare_query:
            # First get retrieval results
            with st.spinner("🔍 Retrieving context from knowledge graph..."):
                retrieval_results = st.session_state.retriever.hybrid_retrieval(
                    compare_query,
                    use_text_embeddings=use_text_embeddings
                )
                context = st.session_state.retriever.format_context_for_llm(retrieval_results)
            
            # Show retrieved context
            with st.expander("📊 Retrieved Context", expanded=False):
                st.write(f"**Intent:** `{retrieval_results['intent']}`")
                st.write(f"**Cypher Results:** {len(retrieval_results['cypher_results'])} records")
                if retrieval_results['cypher_results']:
                    for i, record in enumerate(retrieval_results['cypher_results'][:5], 1):
                        st.write(f"{i}. " + ", ".join(f"{k}: {v}" for k, v in record.items()))
                st.write(f"**Similar Players:** {len(retrieval_results['similar_players'])} found")
            
            st.write("---")
            st.write("### Model Comparison Results")
            
            # Run comparison
            with st.spinner("🧠 Querying all 3 models (this may take 30-60 seconds)..."):
                results = st.session_state.llm.compare_models(context, compare_query)
            
            # Display results in columns
            cols = st.columns(3)
            
            for i, result in enumerate(results):
                with cols[i]:
                    model_name = result['model'].upper()
                    st.write(f"#### {model_name}")
                    st.caption(result['model_name'])
                    
                    if result['success']:
                        st.success(f"⏱️ {result['time_seconds']}s")
                        st.write("**Answer:**")
                        st.info(result['answer'])
                    else:
                        st.error(f"❌ Failed ({result['time_seconds']}s)")
                        st.warning(f"Error: {result['error']}")
            
            # Summary table
            st.write("---")
            st.write("### 📊 Comparison Summary")
            
            summary_data = []
            for r in results:
                summary_data.append({
                    "Model": r['model'].upper(),
                    "Full Name": r['model_name'],
                    "Status": "✅ Success" if r['success'] else "❌ Failed",
                    "Response Time (s)": r['time_seconds'],
                    "Answer Length": len(r['answer']) if r['answer'] else 0
                })
            
            df = pd.DataFrame(summary_data)
            st.dataframe(df, use_container_width=True)
            
            # Timing chart
            if any(r['success'] for r in results):
                fig = go.Figure(data=[
                    go.Bar(
                        x=[r['model'].upper() for r in results],
                        y=[r['time_seconds'] for r in results],
                        marker_color=['green' if r['success'] else 'red' for r in results]
                    )
                ])
                fig.update_layout(
                    title="Response Time Comparison",
                    xaxis_title="Model",
                    yaxis_title="Time (seconds)",
                    height=300
                )
                st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# MODE 3: COMPARE EMBEDDINGS
# ============================================================================
elif app_mode == "🔬 Compare Embeddings":
    st.write("---")
    st.write("### Compare Text vs Numerical Embeddings")
    st.write("See how different embedding approaches retrieve different similar players")
    
    player_name = st.text_input("Enter a player name:", placeholder="e.g., Erling Haaland")
    
    if player_name:
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("#### 📊 Numerical Embeddings")
            st.caption("Based on 13 statistical features")
            
            with st.spinner("Searching..."):
                num_results = st.session_state.retriever.embedding_retrieval_numerical(
                    f"players similar to {player_name}", limit=10
                )
            
            if num_results:
                # create visualization
                players = [r['player'] for r in num_results]
                similarities = [r['similarity'] for r in num_results]
                
                fig = go.Figure(data=[
                    go.Bar(x=similarities, y=players, orientation='h',
                           marker=dict(color=similarities, colorscale='Blues'))
                ])
                fig.update_layout(
                    title="Similar Players (Stats-Based)",
                    xaxis_title="Similarity Score",
                    yaxis_title="Player",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No results found. Try another player name.")
        
        with col2:
            st.write("#### 📝 Text Embeddings")
            st.caption("Based on natural language descriptions")
            
            with st.spinner("Searching..."):
                text_results = st.session_state.retriever.embedding_retrieval_text(
                    f"players similar to {player_name}", limit=10
                )
            
            if text_results:
                # create visualization
                players = [r['player'] for r in text_results]
                similarities = [r['similarity'] for r in text_results]
                
                fig = go.Figure(data=[
                    go.Bar(x=similarities, y=players, orientation='h',
                           marker=dict(color=similarities, colorscale='Greens'))
                ])
                fig.update_layout(
                    title="Similar Players (Text-Based)",
                    xaxis_title="Similarity Score",
                    yaxis_title="Player",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No results found. Try another player name.")

# ============================================================================
# MODE 3: GRAPH STATISTICS
# ============================================================================
elif app_mode == "📊 Graph Stats":
    st.write("---")
    st.write("### Knowledge Graph Statistics")
    
    with st.spinner("Loading graph statistics..."):
        # get basic stats
        stats_query = """
        MATCH (p:Player)
        WITH count(p) as player_count
        MATCH (t:Team)
        WITH player_count, count(t) as team_count
        MATCH (f:Fixture)
        WITH player_count, team_count, count(f) as fixture_count
        MATCH ()-[r:PLAYED_IN]->()
        RETURN player_count, team_count, fixture_count, count(r) as performance_count
        """
        
        from neo4j import GraphDatabase
        config = load_config()
        driver = GraphDatabase.driver(config['URI'], auth=(config['USERNAME'], config['PASSWORD']))
        
        with driver.session() as session:
            result = session.run(stats_query).single()
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Players", f"{result['player_count']:,}")
            with col2:
                st.metric("Teams", f"{result['team_count']:,}")
            with col3:
                st.metric("Fixtures", f"{result['fixture_count']:,}")
            with col4:
                st.metric("Performances", f"{result['performance_count']:,}")
            
            # top scorers chart
            st.write("---")
            st.write("### 🎯 Top 10 Goal Scorers")
            
            top_scorers_query = """
            MATCH (p:Player)-[r:PLAYED_IN]->(f:Fixture {season: '2022-23'})
            WITH p.player_name as player, sum(r.goals_scored) as total_goals
            WHERE total_goals > 0
            RETURN player, total_goals
            ORDER BY total_goals DESC
            LIMIT 10
            """
            
            scorers = session.run(top_scorers_query)
            scorers_data = [dict(r) for r in scorers]
            
            if scorers_data:
                df = pd.DataFrame(scorers_data)
                fig = px.bar(df, x='total_goals', y='player', orientation='h',
                            title='Top Goal Scorers 2022-23',
                            labels={'total_goals': 'Goals', 'player': 'Player'},
                            color='total_goals', color_continuous_scale='Reds')
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            # position distribution
            st.write("---")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("### 👥 Players by Position")
                pos_query = """
                MATCH (p:Player)-[:PLAYS_POSITION]->(pos:Position)
                RETURN pos.name as position, count(p) as player_count
                ORDER BY player_count DESC
                """
                pos_data = [dict(r) for r in session.run(pos_query)]
                
                if pos_data:
                    df_pos = pd.DataFrame(pos_data)
                    fig = px.pie(df_pos, values='player_count', names='position',
                                title='Position Distribution',
                                color_discrete_sequence=px.colors.qualitative.Set3)
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.write("### ⭐ Top Fantasy Points")
                points_query = """
                MATCH (p:Player)-[r:PLAYED_IN]->(f:Fixture {season: '2022-23'})
                WITH p.player_name as player, sum(r.total_points) as total_points
                WHERE total_points > 0
                RETURN player, total_points
                ORDER BY total_points DESC
                LIMIT 10
                """
                points_data = [dict(r) for r in session.run(points_query)]
                
                if points_data:
                    df_points = pd.DataFrame(points_data)
                    fig = px.bar(df_points, x='total_points', y='player', orientation='h',
                                labels={'total_points': 'Fantasy Points', 'player': 'Player'},
                                color='total_points', color_continuous_scale='Viridis')
                    fig.update_layout(height=350, showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
        
        driver.close()

# footer
st.write("---")
st.caption("""
**FantasyTrivia** | Graph-RAG System for FPL | Milestone 3  
Data: FPL 2022-23 Season | Knowledge Graph: Neo4j | Embeddings: SentenceTransformer | LLM: HuggingFace  
Dual Retrieval: Cypher Queries + Vector Embeddings
""")

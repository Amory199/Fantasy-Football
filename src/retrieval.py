"""
Graph-RAG Retrieval Layer
Combines Cypher queries with embedding-based similarity search
"""

import os
import certifi

# Fix SSL certificate verification on Windows
os.environ.setdefault('SSL_CERT_FILE', certifi.where())

from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
from preprocessing import InputPreprocessor
from cypher_queries import CypherQueryLibrary
from typing import List, Dict, Optional, Tuple
import numpy as np

def load_config():
    """Load Neo4j config"""
    config = {}
    with open('config.txt', 'r') as f:
        for line in f:
            line = line.strip()
            if '=' in line:
                key, value = line.split('=', 1)
                config[key] = value
    return config


class GraphRetriever:
    """
    Handles both structured (Cypher) and semantic (embedding) retrieval
    """
    
    def __init__(self, config: Dict):
        self.driver = GraphDatabase.driver(
            config['URI'],
            auth=(config['USERNAME'], config['PASSWORD'])
        )
        self.preprocessor = InputPreprocessor()
        self.query_lib = CypherQueryLibrary()
        
    def close(self):
        self.driver.close()
    
    def _run_cypher_query(self, query: str, params: Dict = None) -> List[Dict]:
        """Execute Cypher query and return results"""
        with self.driver.session() as session:
            result = session.run(query, params or {})
            return [dict(record) for record in result]
    
    def baseline_retrieval(self, user_query: str) -> Tuple[str, List[Dict]]:
        """
        Baseline approach: use intent + entities to select and execute Cypher query
        Returns (intent, results)
        """
        # preprocess to get intent and entities
        intent, entities, _ = self.preprocessor.preprocess(user_query)
        
        # get appropriate Cypher query based on intent and entities
        cypher_query = self.query_lib.get_query_for_intent(intent, entities)
        
        if not cypher_query:
            return intent, []
        
        # execute the query (no params needed, queries are self-contained)
        results = self._run_cypher_query(cypher_query)
        return intent, results
    
    def embedding_retrieval_numerical(self, user_query: str, limit: int = 5) -> List[Dict]:
        """
        Approach 1: Find similar players using numerical stat embeddings
        """
        # get query embedding using same features as player embeddings
        # for simplicity, just search by player name from entities
        _, entities, _ = self.preprocessor.preprocess(user_query)
        
        players = entities.get('players', [])
        if not players:
            return []
        
        # get the first player mentioned
        query_player = players[0]
        
        # find similar players using numerical embeddings
        query = """
            MATCH (p1:Player {player_name: $player_name})
            WHERE p1.numerical_embedding IS NOT NULL
            MATCH (p2:Player)
            WHERE p2.numerical_embedding IS NOT NULL
              AND p2.player_name <> $player_name
            WITH p2,
                 gds.similarity.cosine(p1.numerical_embedding, p2.numerical_embedding) as similarity
            RETURN p2.player_name as player, similarity
            ORDER BY similarity DESC
            LIMIT $limit
        """
        
        return self._run_cypher_query(query, {
            'player_name': query_player,
            'limit': limit
        })
    
    def embedding_retrieval_text(self, user_query: str, limit: int = 5) -> List[Dict]:
        """
        Approach 2: Find similar players using text description embeddings
        Uses the actual query text to find semantically similar player descriptions
        """
        # embed the user's query
        query_embedding = self.preprocessor.embed_query(user_query)
        
        # search for similar player descriptions
        query = """
            MATCH (p:Player)
            WHERE p.text_embedding IS NOT NULL
            WITH p,
                 gds.similarity.cosine(p.text_embedding, $query_emb) as similarity
            RETURN p.player_name as player, similarity
            ORDER BY similarity DESC
            LIMIT $limit
        """
        
        return self._run_cypher_query(query, {
            'query_emb': query_embedding,
            'limit': limit
        })
    
    def hybrid_retrieval(self, user_query: str, use_text_embeddings: bool = True) -> Dict:
        """
        Combines baseline Cypher query with embedding-based similarity search
        
        Returns a dict with:
        - intent: classified intent
        - cypher_results: structured query results
        - similar_players: semantically similar players
        - embedding_type: which embedding approach was used
        """
        # get baseline results
        intent, cypher_results = self.baseline_retrieval(user_query)
        
        # get embedding-based results
        if use_text_embeddings:
            similar_players = self.embedding_retrieval_text(user_query, limit=5)
            embedding_type = "text"
        else:
            similar_players = self.embedding_retrieval_numerical(user_query, limit=5)
            embedding_type = "numerical"
        
        return {
            'intent': intent,
            'cypher_results': cypher_results,
            'similar_players': similar_players,
            'embedding_type': embedding_type,
            'query': user_query
        }
    
    def format_context_for_llm(self, retrieval_results: Dict) -> str:
        """
        Format the retrieved information into a readable context for the LLM
        """
        context_parts = []
        
        # add the user's query
        context_parts.append(f"User Question: {retrieval_results['query']}\n")
        
        # add intent
        context_parts.append(f"Query Intent: {retrieval_results['intent']}\n")
        
        # add Cypher results
        if retrieval_results['cypher_results']:
            context_parts.append("Structured Data from Knowledge Graph:")
            for i, result in enumerate(retrieval_results['cypher_results'][:10], 1):
                # format each result nicely
                result_str = ", ".join(f"{k}: {v}" for k, v in result.items())
                context_parts.append(f"  {i}. {result_str}")
            context_parts.append("")
        
        # add similar players
        if retrieval_results['similar_players']:
            context_parts.append(f"Semantically Similar Players ({retrieval_results['embedding_type']} embeddings):")
            for i, player in enumerate(retrieval_results['similar_players'], 1):
                context_parts.append(f"  {i}. {player['player']} (similarity: {player['similarity']:.4f})")
            context_parts.append("")
        
        return "\n".join(context_parts)


def main():
    """Test the retrieval system"""
    print("="*60)
    print("TESTING GRAPH-RAG RETRIEVAL")
    print("="*60)
    
    config = load_config()
    retriever = GraphRetriever(config)
    
    # test queries
    test_queries = [
        "Who scored the most goals in 2022-23?",
        "Tell me about Erling Haaland's performance",
        "Which defenders got the most clean sheets?",
        "Compare Mohamed Salah and Harry Kane",
        "Who are the best midfielders?"
    ]
    
    try:
        for query in test_queries:
            print(f"\n{'='*60}")
            print(f"Query: {query}")
            print(f"{'='*60}")
            
            # test baseline retrieval
            intent, results = retriever.baseline_retrieval(query)
            print(f"\nIntent: {intent}")
            print(f"Cypher Results: {len(results)} records")
            if results:
                print(f"Sample: {results[0]}")
            
            # test hybrid retrieval with text embeddings
            hybrid_results = retriever.hybrid_retrieval(query, use_text_embeddings=True)
            print(f"\nHybrid Results:")
            print(f"  - Cypher: {len(hybrid_results['cypher_results'])} records")
            print(f"  - Similar players: {len(hybrid_results['similar_players'])} players")
            
            # show formatted context
            print(f"\n{'-'*60}")
            print("Formatted Context for LLM:")
            print(f"{'-'*60}")
            context = retriever.format_context_for_llm(hybrid_results)
            print(context)
    
    finally:
        retriever.close()
    
    print("\n" + "="*60)
    print("RETRIEVAL TESTING COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()

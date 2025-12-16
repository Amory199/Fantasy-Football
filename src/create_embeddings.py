"""
Create and store player embeddings in Neo4j
Uses numerical features to create vector representations of players
"""

import pandas as pd
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict

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

class PlayerEmbedder:
    def __init__(self, config: Dict[str, str], embedding_model: str = "all-MiniLM-L6-v2"):
        """
        Initialize with Neo4j connection and embedding model
        We'll try TWO approaches:
        1. Numerical features directly (stats-based)
        2. Text descriptions of players embedded with sentence transformer
        """
        self.driver = GraphDatabase.driver(
            config['URI'], 
            auth=(config['USERNAME'], config['PASSWORD'])
        )
        self.text_embedder = SentenceTransformer(embedding_model, device='cpu')
        
    def close(self):
        self.driver.close()
    
    def get_player_stats(self, season: str = '2022-23'):
        """
        Fetch aggregated stats for all players from Neo4j
        Returns list of player dicts with numerical features
        """
        query = f"""
        MATCH (p:Player)-[r:PLAYED_IN]->(f:Fixture {{season: '{season}'}})
        WITH p,
             sum(r.total_points) as total_points,
             sum(r.goals_scored) as total_goals,
             sum(r.assists) as total_assists,
             sum(r.minutes) as total_minutes,
             sum(r.clean_sheets) as clean_sheets,
             sum(r.saves) as saves,
             sum(r.bonus) as bonus,
             sum(r.bps) as bps,
             count(r) as matches_played,
             sum(r.influence) as influence,
             sum(r.creativity) as creativity,
             sum(r.threat) as threat,
             sum(r.ict_index) as ict_index
        WHERE total_minutes > 0
        RETURN p.player_name as name,
               p.player_element as element,
               total_points,
               total_goals,
               total_assists,
               total_minutes,
               clean_sheets,
               saves,
               bonus,
               bps,
               matches_played,
               influence,
               creativity,
               threat,
               ict_index
        """
        
        with self.driver.session() as session:
            result = session.run(query)
            return [dict(record) for record in result]
    
    def create_numerical_embeddings(self, player_stats: List[Dict]):
        """
        Approach 1: Use numerical stats directly as embeddings
        Normalize features to 0-1 range
        """
        feature_names = [
            'total_points', 'total_goals', 'total_assists', 
            'total_minutes', 'clean_sheets', 'saves', 'bonus', 
            'bps', 'matches_played', 'influence', 'creativity', 
            'threat', 'ict_index'
        ]
        
        # extract feature matrix
        features = []
        for player in player_stats:
            feature_vector = [float(player.get(feat, 0)) for feat in feature_names]
            features.append(feature_vector)
        
        features = np.array(features)
        
        # normalize each feature to 0-1
        mins = features.min(axis=0)
        maxs = features.max(axis=0)
        ranges = maxs - mins
        ranges[ranges == 0] = 1  # avoid division by zero
        normalized = (features - mins) / ranges
        
        # add embeddings to player dicts
        for i, player in enumerate(player_stats):
            player['numerical_embedding'] = normalized[i].tolist()
        
        return player_stats
    
    def create_text_embeddings(self, player_stats: List[Dict]):
        """
        Approach 2: Create text descriptions and embed them
        Convert numerical stats to natural language, then use sentence transformer
        """
        descriptions = []
        for player in player_stats:
            desc = f"""
            Player {player['name']} played {player['matches_played']} matches 
            and scored {player['total_points']} fantasy points. 
            They scored {player['total_goals']} goals and provided {player['total_assists']} assists.
            Total minutes played: {player['total_minutes']}.
            Clean sheets: {player['clean_sheets']}, Bonus points: {player['bonus']}.
            """
            descriptions.append(desc.strip())
        
        # embed all descriptions
        embeddings = self.text_embedder.encode(descriptions)
        
        # add to player dicts
        for i, player in enumerate(player_stats):
            player['text_embedding'] = embeddings[i].tolist()
        
        return player_stats
    
    def store_embeddings_in_neo4j(self, player_stats: List[Dict], embedding_type: str = 'numerical'):
        """
        Store embeddings as node properties in Neo4j
        Also create vector index for similarity search
        """
        with self.driver.session() as session:
            # store embeddings on player nodes
            for player in player_stats:
                embedding_key = f'{embedding_type}_embedding'
                if embedding_key in player:
                    session.run(f"""
                        MATCH (p:Player {{player_name: $name, player_element: $element}})
                        SET p.{embedding_type}_embedding = $embedding
                    """, name=player['name'], element=player['element'], 
                         embedding=player[embedding_key])
            
            print(f"✓ Stored {len(player_stats)} {embedding_type} embeddings")
            
            # create vector index if it doesn't exist
            index_name = f"player_{embedding_type}_index"
            embedding_dim = len(player_stats[0][f'{embedding_type}_embedding'])
            
            try:
                session.run(f"""
                    CREATE VECTOR INDEX {index_name} IF NOT EXISTS
                    FOR (p:Player)
                    ON p.{embedding_type}_embedding
                    OPTIONS {{
                        indexConfig: {{
                            `vector.dimensions`: {embedding_dim},
                            `vector.similarity_function`: 'cosine'
                        }}
                    }}
                """)
                print(f"✓ Created vector index: {index_name}")
            except Exception as e:
                print(f"Note: Vector index creation: {e}")
    
    def test_similarity_search(self, query_player: str, embedding_type: str = 'numerical', limit: int = 5):
        """
        Test similarity search - find similar players
        """
        with self.driver.session() as session:
            # get query player's embedding
            result = session.run(f"""
                MATCH (p:Player {{player_name: $name}})
                RETURN p.{embedding_type}_embedding as embedding
            """, name=query_player)
            
            record = result.single()
            if not record or not record['embedding']:
                print(f"Player {query_player} has no {embedding_type} embedding")
                return []
            
            query_embedding = record['embedding']
            
            # find similar players using vector index
            result = session.run(f"""
                MATCH (p:Player)
                WHERE p.{embedding_type}_embedding IS NOT NULL
                  AND p.player_name <> $query_name
                WITH p, 
                     gds.similarity.cosine(p.{embedding_type}_embedding, $query_emb) as similarity
                RETURN p.player_name as player, similarity
                ORDER BY similarity DESC
                LIMIT {limit}
            """, query_name=query_player, query_emb=query_embedding)
            
            similar_players = [dict(record) for record in result]
            return similar_players


def main():
    print("="*60)
    print("CREATING PLAYER EMBEDDINGS FOR NEO4J")
    print("="*60)
    
    config = load_config()
    embedder = PlayerEmbedder(config)
    
    try:
        # get player stats
        print("\n1. Fetching player statistics from Neo4j...")
        player_stats = embedder.get_player_stats(season='2022-23')
        print(f"✓ Retrieved stats for {len(player_stats)} players")
        
        # create numerical embeddings (Approach 1)
        print("\n2. Creating numerical embeddings (Approach 1)...")
        player_stats = embedder.create_numerical_embeddings(player_stats)
        embedder.store_embeddings_in_neo4j(player_stats, 'numerical')
        
        # create text embeddings (Approach 2)
        print("\n3. Creating text-based embeddings (Approach 2)...")
        player_stats = embedder.create_text_embeddings(player_stats)
        embedder.store_embeddings_in_neo4j(player_stats, 'text')
        
        # test similarity search
        print("\n4. Testing similarity search...")
        print("\nNumerical embedding - Players similar to Erling Haaland:")
        similar = embedder.test_similarity_search('Erling Haaland', 'numerical', limit=5)
        for p in similar:
            print(f"  {p['player']}: {p['similarity']:.4f}")
        
        print("\nText embedding - Players similar to Mohamed Salah:")
        similar = embedder.test_similarity_search('Mohamed Salah', 'text', limit=5)
        for p in similar:
            print(f"  {p['player']}: {p['similarity']:.4f}")
        
        print("\n" + "="*60)
        print("EMBEDDINGS CREATED SUCCESSFULLY!")
        print("="*60)
        
    finally:
        embedder.close()

if __name__ == "__main__":
    main()

"""
Input preprocessing module for FPL Graph-RAG
Handles intent classification, entity extraction, and input embedding
"""

import re
import os
from typing import Dict, List, Tuple
from sentence_transformers import SentenceTransformer

class InputPreprocessor:
    def __init__(self, embedding_model_name: str = "all-MiniLM-L6-v2"):
        """Initialize preprocessor with embedding model"""
        # Fix for meta tensor issue: don't specify device in constructor
        # The model will load to CPU by default anyway
        self.embedding_model = SentenceTransformer(embedding_model_name)
        
        # Define intent keywords (simple rule-based approach)
        self.intent_keywords = {
            'player_stats': ['stats', 'performance', 'about', 'tell me'],
            'team_query': ['team', 'club', 'squad', 'which team'],
            'top_players': ['best', 'top', 'highest', 'most', 'least', 'fewest', 'lowest', 'worst', 'leading', 'goals', 'assists', 'points', 'scored'],
            'fixture_query': ['fixture', 'match', 'game', 'played against', 'vs'],
            'position_query': ['defender', 'midfielder', 'forward', 'goalkeeper', 'position'],
            'season_query': ['season', '2021-22', '2022-23'],
            'comparison': ['compare', 'vs', 'versus', 'better', 'difference between'],
            'recommendation': ['recommend', 'suggest', 'should i pick', 'who to choose'],
            'general_question': ['who', 'what', 'when', 'where', 'how']
        }
        
        # FPL-specific entities to extract
        self.positions = ['GK', 'DEF', 'MID', 'FWD', 'goalkeeper', 'defender', 'midfielder', 'forward']
        self.seasons = ['2021-22', '2022-23']
        
    def classify_intent(self, query: str) -> str:
        """
        Classify user intent using keyword matching
        Returns the most likely intent category
        """
        query_lower = query.lower()
        intent_scores = {}
        
        for intent, keywords in self.intent_keywords.items():
            score = sum(1 for keyword in keywords if keyword in query_lower)
            if score > 0:
                intent_scores[intent] = score
        
        if intent_scores:
            # return the intent with highest score
            return max(intent_scores, key=intent_scores.get)
        
        return 'general_question'
    
    def extract_entities(self, query: str) -> Dict[str, List[str]]:
        """
        Extract relevant entities from the query
        Returns dict with entity types and their values
        """
        entities = {
            'players': [],
            'teams': [],
            'positions': [],
            'seasons': [],
            'stats': [],
            'numbers': []
        }
        
        query_lower = query.lower()
        
        # Extract positions
        for pos in self.positions:
            if pos.lower() in query_lower:
                # normalize to Neo4j format
                if 'goalkeeper' in pos.lower():
                    entities['positions'].append('GK')
                elif 'defender' in pos.lower():
                    entities['positions'].append('DEF')
                elif 'midfielder' in pos.lower():
                    entities['positions'].append('MID')
                elif 'forward' in pos.lower():
                    entities['positions'].append('FWD')
                else:
                    entities['positions'].append(pos.upper())
        
        # Extract seasons
        for season in self.seasons:
            if season in query:
                entities['seasons'].append(season)
        
        # Extract player names - improved logic
        # Look for consecutive capitalized words (proper nouns)
        words = query.split()
        i = 0
        while i < len(words):
            # Clean the word - remove trailing possessive 's or apostrophe
            clean_word = words[i].rstrip("'s").rstrip("'")
            if clean_word and clean_word[0].isupper():
                # Start collecting capitalized words
                name_parts = [clean_word]
                j = i + 1
                while j < len(words):
                    next_clean = words[j].rstrip("'s").rstrip("'")
                    if next_clean and next_clean[0].isupper():
                        name_parts.append(next_clean)
                        j += 1
                    else:
                        break
                
                # If we found at least 2 capitalized words, it's likely a name
                if len(name_parts) >= 2:
                    full_name = ' '.join(name_parts)
                    entities['players'].append(full_name)
                    i = j
                else:
                    i += 1
            else:
                i += 1
        
        # Also check for common single-name players or partial matches
        # If no player found, try extracting any capitalized word
        if not entities['players']:
            for word in words:
                if word and len(word) > 3 and word[0].isupper() and word not in ['Tell', 'Show', 'Who', 'Which', 'What', 'The']:
                    entities['players'].append(word)
        
        # Extract stat types
        stat_keywords = ['goals', 'assists', 'points', 'clean sheets', 'saves', 'minutes']
        for stat in stat_keywords:
            if stat in query_lower:
                entities['stats'].append(stat)
        
        # Extract numbers
        numbers = re.findall(r'\b\d+\b', query)
        entities['numbers'] = [int(n) for n in numbers]
        
        return entities
    
    def embed_query(self, query: str) -> List[float]:
        """
        Convert query to embedding vector for semantic search
        Returns embedding as list of floats
        """
        embedding = self.embedding_model.encode(query)
        return embedding.tolist()
    
    def preprocess(self, query: str) -> Tuple[str, Dict, List[float]]:
        """
        Full preprocessing pipeline
        Returns: (intent, entities, embedding)
        """
        intent = self.classify_intent(query)
        entities = self.extract_entities(query)
        embedding = self.embed_query(query)
        
        return intent, entities, embedding


if __name__ == "__main__":
    # quick test
    preprocessor = InputPreprocessor()
    
    test_queries = [
        "Who are the top scorers in 2022-23?",
        "How many goals did Erling Haaland score?",
        "Compare Mohamed Salah and Kevin De Bruyne",
        "Which defenders had the most clean sheets?",
        "Recommend me some good midfielders for my team"
    ]
    
    for query in test_queries:
        intent, entities, embedding = preprocessor.preprocess(query)
        print(f"\nQuery: {query}")
        print(f"Intent: {intent}")
        print(f"Entities: {entities}")
        print(f"Embedding shape: {len(embedding)}")

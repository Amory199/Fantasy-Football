"""
Cypher query templates for FPL Graph-RAG baseline retrieval
10+ query templates covering different question types
"""

class CypherQueryLibrary:
    """
    Library of parameterized Cypher queries for FPL knowledge graph
    """
    
    @staticmethod
    def get_top_scorers(season: str = None, position: str = None, limit: int = 10):
        """
        Query 1: Get top goal scorers
        Filters by season and/or position if provided
        """
        query = """
        MATCH (p:Player)-[r:PLAYED_IN]->(f:Fixture)
        WHERE 1=1
        """
        if season:
            query += f" AND f.season = '{season}'"
        if position:
            query += f"""
            AND EXISTS {{
                MATCH (p)-[:PLAYS_AS]->(pos:Position {{name: '{position}'}})
            }}
            """
        query += """
        WITH p, sum(r.goals_scored) as total_goals
        WHERE total_goals > 0
        RETURN p.player_name as player, total_goals
        ORDER BY total_goals DESC
        """
        query += f" LIMIT {limit}"
        return query
    
    @staticmethod
    def get_top_assisters(season: str = None, position: str = None, limit: int = 10):
        """
        Query 2: Get top assist providers
        """
        query = """
        MATCH (p:Player)-[r:PLAYED_IN]->(f:Fixture)
        WHERE 1=1
        """
        if season:
            query += f" AND f.season = '{season}'"
        if position:
            query += f"""
            AND EXISTS {{
                MATCH (p)-[:PLAYS_AS]->(pos:Position {{name: '{position}'}})
            }}
            """
        query += """
        WITH p, sum(r.assists) as total_assists
        WHERE total_assists > 0
        RETURN p.player_name as player, total_assists
        ORDER BY total_assists DESC
        """
        query += f" LIMIT {limit}"
        return query
    
    @staticmethod
    def get_top_points(season: str = None, position: str = None, limit: int = 10):
        """
        Query 3: Get top point scorers (fantasy points)
        """
        query = """
        MATCH (p:Player)-[r:PLAYED_IN]->(f:Fixture)
        WHERE 1=1
        """
        if season:
            query += f" AND f.season = '{season}'"
        if position:
            query += f"""
            AND EXISTS {{
                MATCH (p)-[:PLAYS_AS]->(pos:Position {{name: '{position}'}})
            }}
            """
        query += """
        WITH p, sum(r.total_points) as total_points
        WHERE total_points > 0
        RETURN p.player_name as player, total_points
        ORDER BY total_points DESC
        """
        query += f" LIMIT {limit}"
        return query
    
    @staticmethod
    def get_player_stats(player_name: str, season: str = None):
        """
        Query 4: Get detailed stats for a specific player
        Uses CONTAINS for partial matching
        """
        # escape quotes in player name
        safe_name = player_name.replace("'", "\\'")
        query = f"""
        MATCH (p:Player)-[r:PLAYED_IN]->(f:Fixture)
        WHERE p.player_name CONTAINS '{safe_name}'
        """
        if season:
            query += f" AND f.season = '{season}'"
        query += """
        WITH p, 
             count(r) as matches_played,
             sum(r.goals_scored) as total_goals,
             sum(r.assists) as total_assists,
             sum(r.total_points) as total_points,
             sum(r.minutes) as total_minutes,
             sum(r.clean_sheets) as clean_sheets,
             sum(r.yellow_cards) as yellow_cards,
             sum(r.red_cards) as red_cards
        RETURN p.player_name as player,
               matches_played,
               total_goals,
               total_assists,
               total_points,
               total_minutes,
               clean_sheets,
               yellow_cards,
               red_cards
        LIMIT 1
        """
        return query
    
    @staticmethod
    def get_team_players(team_name: str, season: str = None):
        """
        Query 5: Get all players from a specific team
        """
        query = f"""
        MATCH (p:Player)-[r:PLAYED_IN]->(f:Fixture)
        MATCH (f)-[:HAS_HOME_TEAM|HAS_AWAY_TEAM]->(t:Team {{name: '{team_name}'}})
        """
        if season:
            query += f" WHERE f.season = '{season}'"
        query += """
        RETURN DISTINCT p.player_name as player, p.player_element as element
        ORDER BY p.player_name
        """
        return query
    
    @staticmethod
    def get_top_clean_sheets(position: str = 'DEF', season: str = None, limit: int = 10):
        """
        Query 6: Get players with most clean sheets (usually defenders/goalkeepers)
        """
        query = f"""
        MATCH (p:Player)-[:PLAYS_AS]->(pos:Position {{name: '{position}'}})
        MATCH (p)-[r:PLAYED_IN]->(f:Fixture)
        """
        if season:
            query += f" WHERE f.season = '{season}'"
        query += """
        WITH p, sum(r.clean_sheets) as total_clean_sheets
        WHERE total_clean_sheets > 0
        RETURN p.player_name as player, total_clean_sheets
        ORDER BY total_clean_sheets DESC
        """
        query += f" LIMIT {limit}"
        return query
    
    @staticmethod
    def get_player_fixtures(player_name: str, season: str = None, limit: int = 20):
        """
        Query 7: Get fixture history for a player
        """
        query = f"""
        MATCH (p:Player {{player_name: '{player_name}'}})-[r:PLAYED_IN]->(f:Fixture)
        MATCH (f)-[:HAS_HOME_TEAM]->(home:Team)
        MATCH (f)-[:HAS_AWAY_TEAM]->(away:Team)
        MATCH (gw:Gameweek)-[:HAS_FIXTURE]->(f)
        """
        if season:
            query += f" WHERE f.season = '{season}'"
        query += """
        RETURN gw.GW_number as gameweek,
               f.season as season,
               home.name as home_team,
               away.name as away_team,
               r.minutes as minutes,
               r.goals_scored as goals,
               r.assists as assists,
               r.total_points as points
        ORDER BY f.season, gw.GW_number
        """
        query += f" LIMIT {limit}"
        return query
    
    @staticmethod
    def compare_players(player1: str, player2: str, season: str = None):
        """
        Query 8: Compare two players' statistics
        Uses CONTAINS for partial matching
        """
        # escape quotes in player names
        safe_p1 = player1.replace("'", "\\'")
        safe_p2 = player2.replace("'", "\\'")
        query = f"""
        MATCH (p:Player)-[r:PLAYED_IN]->(f:Fixture)
        WHERE (p.player_name CONTAINS '{safe_p1}' OR p.player_name CONTAINS '{safe_p2}')
        """
        if season:
            query += f" AND f.season = '{season}'"
        query += """
        WITH p.player_name as player,
             count(r) as matches,
             sum(r.goals_scored) as goals,
             sum(r.assists) as assists,
             sum(r.total_points) as points,
             sum(r.minutes) as minutes
        RETURN player, matches, goals, assists, points, minutes
        ORDER BY player
        LIMIT 2
        """
        return query
    
    @staticmethod
    def get_position_distribution(team_name: str = None, season: str = None):
        """
        Query 9: Get count of players by position
        """
        query = """
        MATCH (p:Player)-[:PLAYS_AS]->(pos:Position)
        """
        if team_name or season:
            query += " MATCH (p)-[r:PLAYED_IN]->(f:Fixture)"
            if team_name:
                query += f"""
                MATCH (f)-[:HAS_HOME_TEAM|HAS_AWAY_TEAM]->(t:Team {{name: '{team_name}'}})
                """
            if season:
                query += f" WHERE f.season = '{season}'"
        query += """
        RETURN pos.name as position, count(DISTINCT p) as player_count
        ORDER BY player_count DESC
        """
        return query
    
    @staticmethod
    def get_most_valuable_players(position: str = None, season: str = None, limit: int = 10):
        """
        Query 10: Get players with best points per minute ratio
        """
        query = """
        MATCH (p:Player)-[r:PLAYED_IN]->(f:Fixture)
        WHERE r.minutes > 0
        """
        if season:
            query += f" AND f.season = '{season}'"
        if position:
            query += f"""
            AND EXISTS {{
                MATCH (p)-[:PLAYS_AS]->(pos:Position {{name: '{position}'}})
            }}
            """
        query += """
        WITH p, 
             sum(r.total_points) as total_points,
             sum(r.minutes) as total_minutes
        WHERE total_minutes >= 900
        WITH p, total_points, total_minutes, 
             toFloat(total_points) / (toFloat(total_minutes) / 90.0) as points_per_90
        RETURN p.player_name as player,
               total_points,
               total_minutes,
               round(points_per_90, 2) as points_per_90min
        ORDER BY points_per_90 DESC
        """
        query += f" LIMIT {limit}"
        return query
    
    @staticmethod
    def get_teams_in_season(season: str):
        """
        Query 11: Get all teams that played in a specific season
        """
        query = f"""
        MATCH (s:Season {{season_name: '{season}'}})-[:HAS_GW]->(gw:Gameweek)-[:HAS_FIXTURE]->(f:Fixture)
        MATCH (f)-[:HAS_HOME_TEAM|HAS_AWAY_TEAM]->(t:Team)
        RETURN DISTINCT t.name as team
        ORDER BY team
        """
        return query
    
    @staticmethod
    def get_gameweek_top_performers(season: str, gameweek: int, limit: int = 5):
        """
        Query 12: Get top performers in a specific gameweek
        """
        query = f"""
        MATCH (gw:Gameweek {{season: '{season}', GW_number: {gameweek}}})-[:HAS_FIXTURE]->(f:Fixture)
        MATCH (p:Player)-[r:PLAYED_IN]->(f)
        WHERE r.total_points > 0
        RETURN p.player_name as player,
               r.total_points as points,
               r.goals_scored as goals,
               r.assists as assists,
               r.minutes as minutes
        ORDER BY r.total_points DESC
        LIMIT {limit}
        """
        return query
    
    @staticmethod
    def search_players_by_name(partial_name: str, limit: int = 10):
        """
        Query 13: Search for players by partial name match
        """
        query = f"""
        MATCH (p:Player)
        WHERE toLower(p.player_name) CONTAINS toLower('{partial_name}')
        RETURN DISTINCT p.player_name as player, p.player_element as element
        ORDER BY p.player_name
        LIMIT {limit}
        """
        return query
    
    @staticmethod
    def get_query_for_intent(intent: str, entities: dict):
        """
        Main dispatcher: select appropriate query based on intent and entities
        Returns the right Cypher query with parameters filled in
        """
        season = entities.get('seasons', [None])[0] if entities.get('seasons') else None
        position = entities.get('positions', [None])[0] if entities.get('positions') else None
        players = entities.get('players', [])
        
        if intent == 'player_stats' and players:
            return CypherQueryLibrary.get_player_stats(players[0], season)
        
        elif intent == 'comparison' and len(players) >= 2:
            return CypherQueryLibrary.compare_players(players[0], players[1], season)
        
        elif intent == 'top_players':
            if 'goals' in entities.get('stats', []):
                return CypherQueryLibrary.get_top_scorers(season, position)
            elif 'assists' in entities.get('stats', []):
                return CypherQueryLibrary.get_top_assisters(season, position)
            elif 'clean sheets' in entities.get('stats', []):
                return CypherQueryLibrary.get_top_clean_sheets(position or 'DEF', season)
            else:
                return CypherQueryLibrary.get_top_points(season, position)
        
        elif intent == 'fixture_query' and players:
            return CypherQueryLibrary.get_player_fixtures(players[0], season)
        
        elif intent == 'position_query':
            return CypherQueryLibrary.get_position_distribution(season=season)
        
        elif intent == 'recommendation':
            # for recommendations, get most valuable players
            return CypherQueryLibrary.get_most_valuable_players(position, season)
        
        # default: top point scorers
        return CypherQueryLibrary.get_top_points(season, position)


if __name__ == "__main__":
    # test queries
    lib = CypherQueryLibrary()
    
    print("Query 1 - Top scorers in 2022-23:")
    print(lib.get_top_scorers(season='2022-23', limit=5))
    print("\n" + "="*60 + "\n")
    
    print("Query 4 - Haaland stats:")
    print(lib.get_player_stats('Erling Haaland', '2022-23'))
    print("\n" + "="*60 + "\n")
    
    print("Query 8 - Compare Salah vs De Bruyne:")
    print(lib.compare_players('Mohamed Salah', 'Kevin De Bruyne', '2022-23'))

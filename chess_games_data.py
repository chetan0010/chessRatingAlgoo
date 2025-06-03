import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def create_chess_games_csv(filename="chess_games.csv", n_games=2000, n_players=100):
    """
    Create a realistic chess games CSV file with various scenarios
    """
    
    # Generate player names with different skill levels
    regular_players = [f"Player_{i:03d}" for i in range(1, n_players-10)]
    strong_players = [f"Master_{i}" for i in range(1, 6)]  # Strong players
    suspicious_players = [f"Suspect_{i}" for i in range(1, 4)]  # Potential cheaters
    beginner_players = [f"Beginner_{i}" for i in range(1, 4)]  # Weak players
    
    all_players = regular_players + strong_players + suspicious_players + beginner_players
    
    # Assign skill levels (approximate ELO)
    player_skills = {}
    
    # Regular players: 1000-1600
    for player in regular_players:
        player_skills[player] = np.random.normal(1300, 150)
    
    # Strong players: 1800-2200
    for player in strong_players:
        player_skills[player] = np.random.normal(2000, 100)
    
    # Suspicious players: Start low but will have unusual patterns
    for player in suspicious_players:
        player_skills[player] = np.random.normal(1200, 50)
    
    # Beginner players: 800-1200
    for player in beginner_players:
        player_skills[player] = np.random.normal(1000, 100)
    
    games = []
    current_date = datetime.now() - timedelta(days=365)
    
    # Track games for suspicious behavior patterns
    games_played = {player: 0 for player in all_players}
    
    for game_id in range(n_games):
        # Select players
        white = random.choice(all_players)
        black = random.choice(all_players)
        while black == white:
            black = random.choice(all_players)
        
        # Game date progression
        current_date += timedelta(hours=random.randint(1, 24))
        
        # Time controls with realistic distribution
        time_control = np.random.choice(
            ['blitz', 'rapid', 'classical', 'correspondence'], 
            p=[0.4, 0.3, 0.25, 0.05]
        )
        
        # Calculate win probability based on skill difference
        white_skill = player_skills[white]
        black_skill = player_skills[black]
        
        # Add suspicious player behavior
        if white in suspicious_players and games_played[white] > 50:
            white_skill += 300  # Sudden improvement
        if black in suspicious_players and games_played[black] > 50:
            black_skill += 300
        
        # Calculate expected outcome
        skill_diff = white_skill - black_skill
        white_win_prob = 1 / (1 + 10**(-skill_diff / 400))
        
        # Determine result with some randomness
        rand = random.random()
        if rand < white_win_prob * 0.85:  # White wins
            result = "1-0"
            winner = white
        elif rand < white_win_prob * 0.85 + (1 - white_win_prob) * 0.85:  # Black wins
            result = "0-1"
            winner = black
        else:  # Draw
            result = "1/2-1/2"
            winner = None
        
        # Generate realistic move counts based on time control and result
        if time_control == 'blitz':
            base_moves = np.random.normal(35, 10)
        elif time_control == 'rapid':
            base_moves = np.random.normal(45, 12)
        elif time_control == 'classical':
            base_moves = np.random.normal(55, 15)
        else:  # correspondence
            base_moves = np.random.normal(65, 20)
        
        # Adjust moves based on result
        if result == "1/2-1/2":
            moves = int(max(20, base_moves + np.random.normal(10, 5)))
        else:
            moves = int(max(15, base_moves + np.random.normal(0, 8)))
        
        # Opening variations
        openings = [
            "Sicilian Defense", "French Defense", "Caro-Kann Defense", 
            "Queen's Gambit", "King's Indian Defense", "English Opening",
            "Ruy Lopez", "Italian Game", "Scandinavian Defense", 
            "Alekhine Defense", "Nimzo-Indian Defense", "Catalan Opening",
            "Vienna Game", "Petrov Defense", "Benoni Defense"
        ]
        opening = random.choice(openings)
        
        # ECO codes (simplified)
        eco_codes = ["A00", "A01", "A02", "B00", "B01", "C00", "C01", "D00", "D01", "E00"]
        eco = random.choice(eco_codes)
        
        # Termination reasons
        if result == "1/2-1/2":
            termination = np.random.choice([
                "Agreement", "Stalemate", "Insufficient material", 
                "Threefold repetition", "50-move rule"
            ], p=[0.4, 0.2, 0.2, 0.15, 0.05])
        else:
            termination = np.random.choice([
                "Checkmate", "Resignation", "Time forfeit", "Abandonment"
            ], p=[0.3, 0.5, 0.15, 0.05])
        
        # Player ratings (approximate current ratings)
        white_rating = int(max(800, min(2400, white_skill + np.random.normal(0, 50))))
        black_rating = int(max(800, min(2400, black_skill + np.random.normal(0, 50))))
        
        # Add some rating inflation over time
        rating_inflation = (current_date - (datetime.now() - timedelta(days=365))).days * 0.1
        white_rating += int(rating_inflation)
        black_rating += int(rating_inflation)
        
        games.append({
            'game_id': f"game_{game_id:06d}",
            'white': white,
            'black': black,
            'result': result,
            'white_rating': white_rating,
            'black_rating': black_rating,
            'moves': moves,
            'time_control': time_control,
            'date': current_date.strftime('%Y-%m-%d'),
            'time': current_date.strftime('%H:%M:%S'),
            'opening': opening,
            'eco': eco,
            'termination': termination,
            'winner': winner if winner else 'Draw'
        })
        
        # Update games played counter
        games_played[white] += 1
        games_played[black] += 1
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(games)
    
    # Add some additional calculated columns
    df['rating_difference'] = abs(df['white_rating'] - df['black_rating'])
    df['game_length_category'] = pd.cut(df['moves'], 
                                       bins=[0, 25, 40, 60, 999], 
                                       labels=['Short', 'Medium', 'Long', 'Very Long'])
    
    # Save to CSV
    df.to_csv(filename, index=False)
    
    print(f"Created {filename} with {len(df)} games")
    print(f"Players: {len(all_players)}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Time controls: {df['time_control'].value_counts().to_dict()}")
    print(f"Results distribution: {df['result'].value_counts().to_dict()}")
    
    # Show sample data
    print("\nSample data:")
    print(df.head(10).to_string())
    
    # Show column information
    print(f"\nColumns in CSV: {list(df.columns)}")
    
    return df

def create_tournament_csv(filename="tournament_games.csv", n_tournaments=5):
    """Create a CSV with tournament-style games"""
    
    players = [f"Player_{i:03d}" for i in range(1, 21)]  # 20 players
    games = []
    game_id = 0
    
    for tournament in range(n_tournaments):
        tournament_date = datetime.now() - timedelta(days=30*tournament)
        
        # Round-robin tournament
        for i, white in enumerate(players):
            for j, black in enumerate(players[i+1:], i+1):
                
                # Simulate skill difference
                white_skill = 1200 + i * 20 + np.random.normal(0, 50)
                black_skill = 1200 + j * 20 + np.random.normal(0, 50)
                
                skill_diff = white_skill - black_skill
                white_win_prob = 1 / (1 + 10**(-skill_diff / 400))
                
                rand = random.random()
                if rand < white_win_prob * 0.8:
                    result = "1-0"
                elif rand < white_win_prob * 0.8 + (1 - white_win_prob) * 0.8:
                    result = "0-1"
                else:
                    result = "1/2-1/2"
                
                games.append({
                    'game_id': f"tournament_{tournament}_game_{game_id}",
                    'tournament': f"Tournament_{tournament + 1}",
                    'round': 1,
                    'white': white,
                    'black': black,
                    'result': result,
                    'white_rating': int(white_skill),
                    'black_rating': int(black_skill),
                    'moves': np.random.randint(25, 80),
                    'time_control': 'classical',
                    'date': tournament_date.strftime('%Y-%m-%d'),
                    'time': f"{9 + (game_id % 8)}:00:00"
                })
                
                game_id += 1
    
    df = pd.DataFrame(games)
    df.to_csv(filename, index=False)
    
    print(f"Created {filename} with {len(df)} tournament games")
    return df

if __name__ == "__main__":
    # Create main games CSV
    print("Creating main chess games CSV...")
    games_df = create_chess_games_csv("chess_games.csv", n_games=2000, n_players=100)
    
    print("\n" + "="*50)
    
    # Create tournament CSV
    print("Creating tournament games CSV...")
    tournament_df = create_tournament_csv("tournament_games.csv", n_tournaments=3)
    
    print("\n" + "="*50)
    print("CSV files created successfully!")
    print("You can now use these files with the chess rating system:")
    print("chess_system.run_full_analysis('chess_games.csv')")

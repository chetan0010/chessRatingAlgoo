import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import networkx as nx
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class EloRating:
    """Traditional ELO rating system implementation"""
    
    def __init__(self, k_factor=32, initial_rating=1200):
        self.k_factor = k_factor
        self.initial_rating = initial_rating
        self.ratings = {}
    
    def get_rating(self, player):
        return self.ratings.get(player, self.initial_rating)
    
    def expected_score(self, rating_a, rating_b):
        return 1 / (1 + 10**((rating_b - rating_a) / 400))
    
    def update_ratings(self, player_a, player_b, result):
        """
        Update ratings based on game result
        result: 1 if player_a wins, 0 if player_b wins, 0.5 for draw
        """
        rating_a = self.get_rating(player_a)
        rating_b = self.get_rating(player_b)
        
        expected_a = self.expected_score(rating_a, rating_b)
        expected_b = self.expected_score(rating_b, rating_a)
        
        new_rating_a = rating_a + self.k_factor * (result - expected_a)
        new_rating_b = rating_b + self.k_factor * ((1 - result) - expected_b)
        
        self.ratings[player_a] = new_rating_a
        self.ratings[player_b] = new_rating_b
        
        return new_rating_a, new_rating_b

class GlickoRating:
    """Glicko rating system implementation"""
    
    def __init__(self, initial_rating=1500, initial_rd=350, c=15.8):
        self.initial_rating = initial_rating
        self.initial_rd = initial_rd  # Rating deviation
        self.c = c  # Rating deviation increase per period
        self.ratings = {}
        self.rds = {}
        self.last_played = {}
    
    def get_rating(self, player):
        return self.ratings.get(player, self.initial_rating)
    
    def get_rd(self, player, current_time=None):
        if player not in self.rds:
            return self.initial_rd
        
        if current_time and player in self.last_played:
            time_diff = (current_time - self.last_played[player]).days
            rd_increase = min(self.c * np.sqrt(time_diff), 350)
            return min(self.rds[player] + rd_increase, 350)
        
        return self.rds[player]
    
    def g_function(self, rd):
        return 1 / np.sqrt(1 + 3 * (rd/400)**2 / np.pi**2)
    
    def expected_score(self, rating_a, rating_b, rd_b):
        g_rd = self.g_function(rd_b)
        return 1 / (1 + 10**(g_rd * (rating_b - rating_a) / 400))
    
    def update_rating(self, player, opponents, results, game_time=None):
        """Update Glicko rating for a player based on multiple games"""
        rating = self.get_rating(player)
        rd = self.get_rd(player, game_time)
        
        d_squared_inv = 0
        sum_term = 0
        
        for opponent, result in zip(opponents, results):
            opp_rating = self.get_rating(opponent)
            opp_rd = self.get_rd(opponent, game_time)
            
            g_rd = self.g_function(opp_rd)
            expected = self.expected_score(rating, opp_rating, opp_rd)
            
            d_squared_inv += g_rd**2 * expected * (1 - expected)
            sum_term += g_rd * (result - expected)
        
        if d_squared_inv > 0:
            d_squared = 1 / d_squared_inv
            
            new_rd = 1 / np.sqrt(1/rd**2 + 1/d_squared)
            new_rating = rating + (new_rd**2) * sum_term
            
            self.ratings[player] = new_rating
            self.rds[player] = new_rd
            if game_time:
                self.last_played[player] = game_time

class ChessRatingSystem:
    """Main chess rating system with ML and analysis features"""
    
    def __init__(self):
        self.elo_system = EloRating()
        self.glicko_system = GlickoRating()
        self.games_df = None
        self.player_stats = {}
        self.cheat_detector = None
        
    def load_data(self, csv_file):
        """Load chess games from CSV file"""
        try:
            self.games_df = pd.read_csv(csv_file)
            print(f"Loaded {len(self.games_df)} games from {csv_file}")
            print("CSV columns:", self.games_df.columns.tolist())
            return True
        except Exception as e:
            print(f"Error loading CSV: {e}")
            return False
    
    def create_sample_data(self, n_games=1000):
        """Create sample chess game data if no CSV is provided"""
        players = [f"Player_{i}" for i in range(50)]
        
        data = []
        for _ in range(n_games):
            white = np.random.choice(players)
            black = np.random.choice(players)
            while black == white:
                black = np.random.choice(players)
            
            # Simulate game result with some skill difference
            white_skill = np.random.normal(1200, 200)
            black_skill = np.random.normal(1200, 200)
            prob_white_wins = 1 / (1 + 10**((black_skill - white_skill) / 400))
            
            rand = np.random.random()
            if rand < prob_white_wins * 0.8:  # White wins
                result = "1-0"
            elif rand < prob_white_wins * 0.8 + (1 - prob_white_wins) * 0.8:  # Black wins
                result = "0-1"
            else:  # Draw
                result = "1/2-1/2"
            
            data.append({
                'white': white,
                'black': black,
                'result': result,
                'moves': np.random.randint(20, 120),
                'time_control': np.random.choice(['blitz', 'rapid', 'classical']),
                'date': datetime.now() - timedelta(days=np.random.randint(0, 365))
            })
        
        self.games_df = pd.DataFrame(data)
        print(f"Created {n_games} sample games")
    
    def process_ratings(self):
        """Process all games and calculate ratings"""
        if self.games_df is None:
            print("No data loaded!")
            return
        
        elo_history = []
        glicko_history = []
        
        for idx, game in self.games_df.iterrows():
            white = game['white']
            black = game['black']
            result = game['result']
            
            # Convert result to numeric
            if result == "1-0":
                score = 1
            elif result == "0-1":
                score = 0
            else:
                score = 0.5
            
            # Update ELO ratings
            elo_white_new, elo_black_new = self.elo_system.update_ratings(white, black, score)
            
            # Update Glicko ratings
            game_date = pd.to_datetime(game.get('date', datetime.now()))
            self.glicko_system.update_rating(white, [black], [score], game_date)
            self.glicko_system.update_rating(black, [white], [1-score], game_date)
            
            # Store history
            elo_history.append({
                'game_id': idx,
                'player': white,
                'rating': elo_white_new,
                'system': 'ELO'
            })
            elo_history.append({
                'game_id': idx,
                'player': black,
                'rating': elo_black_new,
                'system': 'ELO'
            })
            
            glicko_history.append({
                'game_id': idx,
                'player': white,
                'rating': self.glicko_system.get_rating(white),
                'rd': self.glicko_system.get_rd(white),
                'system': 'Glicko'
            })
            glicko_history.append({
                'game_id': idx,
                'player': black,
                'rating': self.glicko_system.get_rating(black),
                'rd': self.glicko_system.get_rd(black),
                'system': 'Glicko'
            })
        
        self.elo_history_df = pd.DataFrame(elo_history)
        self.glicko_history_df = pd.DataFrame(glicko_history)
        
        print("Ratings processed successfully!")
    
    def calculate_player_features(self):
        """Calculate features for each player for ML analysis"""
        features = []
        
        for player in set(list(self.games_df['white']) + list(self.games_df['black'])):
            player_games = self.games_df[
                (self.games_df['white'] == player) | (self.games_df['black'] == player)
            ]
            
            wins = len(player_games[
                ((player_games['white'] == player) & (player_games['result'] == '1-0')) |
                ((player_games['black'] == player) & (player_games['result'] == '0-1'))
            ])
            
            draws = len(player_games[player_games['result'] == '1/2-1/2'])
            total_games = len(player_games)
            
            if total_games > 0:
                win_rate = wins / total_games
                draw_rate = draws / total_games
                avg_moves = player_games['moves'].mean()
                
                # Rating progression
                player_elo_history = self.elo_history_df[self.elo_history_df['player'] == player]
                if len(player_elo_history) > 1:
                    rating_volatility = player_elo_history['rating'].std()
                    rating_trend = (player_elo_history['rating'].iloc[-1] - 
                                  player_elo_history['rating'].iloc[0]) / len(player_elo_history)
                else:
                    rating_volatility = 0
                    rating_trend = 0
                
                features.append({
                    'player': player,
                    'total_games': total_games,
                    'win_rate': win_rate,
                    'draw_rate': draw_rate,
                    'avg_moves': avg_moves,
                    'rating_volatility': rating_volatility,
                    'rating_trend': rating_trend,
                    'elo_rating': self.elo_system.get_rating(player),
                    'glicko_rating': self.glicko_system.get_rating(player),
                    'glicko_rd': self.glicko_system.get_rd(player)
                })
        
        self.player_features_df = pd.DataFrame(features)
        return self.player_features_df
    
    def detect_cheating(self):
        """Use ML to detect potential cheating based on player statistics"""
        if not hasattr(self, 'player_features_df'):
            self.calculate_player_features()
        
        # Prepare features for anomaly detection
        feature_cols = ['win_rate', 'draw_rate', 'avg_moves', 'rating_volatility', 
                       'rating_trend', 'elo_rating', 'glicko_rating']
        
        X = self.player_features_df[feature_cols].fillna(0)
        
        # Normalize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Isolation Forest for anomaly detection
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        anomaly_labels = iso_forest.fit_predict(X_scaled)
        
        self.player_features_df['is_anomaly'] = anomaly_labels == -1
        self.player_features_df['anomaly_score'] = iso_forest.score_samples(X_scaled)
        
        suspicious_players = self.player_features_df[
            self.player_features_df['is_anomaly']
        ].sort_values('anomaly_score')
        
        print(f"Detected {len(suspicious_players)} potentially suspicious players")
        return suspicious_players
    
    def create_player_network(self):
        """Create a network graph of player interactions"""
        G = nx.Graph()
        
        # Add edges based on games played
        for _, game in self.games_df.iterrows():
            white = game['white']
            black = game['black']
            
            if G.has_edge(white, black):
                G[white][black]['weight'] += 1
            else:
                G.add_edge(white, black, weight=1)
        
        # Calculate network metrics
        centrality = nx.degree_centrality(G)
        betweenness = nx.betweenness_centrality(G)
        
        # Add network metrics to player features
        if hasattr(self, 'player_features_df'):
            self.player_features_df['degree_centrality'] = self.player_features_df['player'].map(centrality)
            self.player_features_df['betweenness_centrality'] = self.player_features_df['player'].map(betweenness)
        
        self.player_network = G
        return G
    
    def visualize_ratings(self):
        """Create comprehensive visualizations"""
        if not hasattr(self, 'elo_history_df'):
            print("Process ratings first!")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Rating distribution comparison
        current_elo = [self.elo_system.get_rating(p) for p in self.elo_system.ratings.keys()]
        current_glicko = [self.glicko_system.get_rating(p) for p in self.glicko_system.ratings.keys()]
        
        axes[0, 0].hist(current_elo, alpha=0.7, label='ELO', bins=20, color='blue')
        axes[0, 0].hist(current_glicko, alpha=0.7, label='Glicko', bins=20, color='red')
        axes[0, 0].set_title('Rating Distribution Comparison')
        axes[0, 0].set_xlabel('Rating')
        axes[0, 0].set_ylabel('Number of Players')
        axes[0, 0].legend()
        
        # 2. Top players rating progression
        top_players = self.player_features_df.nlargest(5, 'elo_rating')['player'].tolist()
        for player in top_players:
            player_history = self.elo_history_df[self.elo_history_df['player'] == player]
            axes[0, 1].plot(player_history.index, player_history['rating'], 
                           label=f'{player}', marker='o', markersize=2)
        
        axes[0, 1].set_title('Top 5 Players - ELO Progression')
        axes[0, 1].set_xlabel('Game Number')
        axes[0, 1].set_ylabel('ELO Rating')
        axes[0, 1].legend()
        
        # 3. Win rate vs Rating scatter
        axes[0, 2].scatter(self.player_features_df['win_rate'], 
                          self.player_features_df['elo_rating'], 
                          alpha=0.6, color='green')
        axes[0, 2].set_title('Win Rate vs ELO Rating')
        axes[0, 2].set_xlabel('Win Rate')
        axes[0, 2].set_ylabel('ELO Rating')
        
        # 4. Rating volatility analysis
        axes[1, 0].scatter(self.player_features_df['total_games'], 
                          self.player_features_df['rating_volatility'], 
                          alpha=0.6, color='orange')
        axes[1, 0].set_title('Games Played vs Rating Volatility')
        axes[1, 0].set_xlabel('Total Games')
        axes[1, 0].set_ylabel('Rating Volatility')
        
        # 5. Cheating detection visualization
        if 'is_anomaly' in self.player_features_df.columns:
            normal_players = self.player_features_df[~self.player_features_df['is_anomaly']]
            suspicious_players = self.player_features_df[self.player_features_df['is_anomaly']]
            
            axes[1, 1].scatter(normal_players['win_rate'], normal_players['elo_rating'], 
                              alpha=0.6, color='blue', label='Normal')
            axes[1, 1].scatter(suspicious_players['win_rate'], suspicious_players['elo_rating'], 
                              alpha=0.8, color='red', label='Suspicious', s=100)
            axes[1, 1].set_title('Cheating Detection Results')
            axes[1, 1].set_xlabel('Win Rate')
            axes[1, 1].set_ylabel('ELO Rating')
            axes[1, 1].legend()
        
        # 6. Network visualization (simplified)
        if hasattr(self, 'player_network'):
            # Create a subgraph of most connected players
            degree_dict = dict(self.player_network.degree())
            top_connected = sorted(degree_dict.items(), key=lambda x: x[1], reverse=True)[:15]
            subgraph = self.player_network.subgraph([node for node, degree in top_connected])
            
            pos = nx.spring_layout(subgraph, k=1, iterations=50)
            nx.draw(subgraph, pos, ax=axes[1, 2], 
                   node_size=[degree_dict[node]*20 for node in subgraph.nodes()],
                   node_color='lightblue', 
                   with_labels=True, 
                   font_size=8,
                   edge_color='gray',
                   alpha=0.7)
            axes[1, 2].set_title('Player Network (Top Connected)')
        
        plt.tight_layout()
        plt.show()
    
    def generate_report(self):
        """Generate comprehensive analysis report"""
        if not hasattr(self, 'player_features_df'):
            self.calculate_player_features()
        
        print("="*60)
        print("CHESS RATING SYSTEM ANALYSIS REPORT")
        print("="*60)
        
        print(f"\nGAMES ANALYZED: {len(self.games_df)}")
        print(f"UNIQUE PLAYERS: {len(self.player_features_df)}")
        
        print("\nTOP 10 PLAYERS (ELO RATING):")
        print("-"*40)
        top_elo = self.player_features_df.nlargest(10, 'elo_rating')[
            ['player', 'elo_rating', 'glicko_rating', 'total_games', 'win_rate']
        ]
        for _, player in top_elo.iterrows():
            print(f"{player['player']:<15} ELO: {player['elo_rating']:.0f} "
                  f"Glicko: {player['glicko_rating']:.0f} "
                  f"Games: {player['total_games']:>3} "
                  f"Win Rate: {player['win_rate']:.2f}")
        
        print("\nTOP 10 PLAYERS (GLICKO RATING):")
        print("-"*40)
        top_glicko = self.player_features_df.nlargest(10, 'glicko_rating')[
            ['player', 'elo_rating', 'glicko_rating', 'total_games', 'win_rate']
        ]
        for _, player in top_glicko.iterrows():
            print(f"{player['player']:<15} ELO: {player['elo_rating']:.0f} "
                  f"Glicko: {player['glicko_rating']:.0f} "
                  f"Games: {player['total_games']:>3} "
                  f"Win Rate: {player['win_rate']:.2f}")
        
        if 'is_anomaly' in self.player_features_df.columns:
            suspicious = self.player_features_df[self.player_features_df['is_anomaly']]
            print(f"\nSUSPICIOUS PLAYERS DETECTED: {len(suspicious)}")
            print("-"*40)
            for _, player in suspicious.iterrows():
                print(f"{player['player']:<15} ELO: {player['elo_rating']:.0f} "
                      f"Win Rate: {player['win_rate']:.2f} "
                      f"Anomaly Score: {player['anomaly_score']:.3f}")
        
        print("\nSYSTEM STATISTICS:")
        print("-"*40)
        print(f"Average ELO Rating: {self.player_features_df['elo_rating'].mean():.1f}")
        print(f"Average Glicko Rating: {self.player_features_df['glicko_rating'].mean():.1f}")
        print(f"Rating Standard Deviation (ELO): {self.player_features_df['elo_rating'].std():.1f}")
        print(f"Rating Standard Deviation (Glicko): {self.player_features_df['glicko_rating'].std():.1f}")
        print(f"Average Games per Player: {self.player_features_df['total_games'].mean():.1f}")
        print(f"Overall Win Rate: {self.player_features_df['win_rate'].mean():.3f}")
        
    def run_full_analysis(self, csv_file=None):
        """Run complete analysis pipeline"""
        print("Starting Chess Rating System Analysis...")
        
        # Load or create data
        if csv_file:
            if not self.load_data(csv_file):
                print("Creating sample data instead...")
                self.create_sample_data()
        else:
            print("No CSV file provided. Creating sample data...")
            self.create_sample_data()
        
        # Process ratings
        print("Processing ratings...")
        self.process_ratings()
        
        # Calculate features
        print("Calculating player features...")
        self.calculate_player_features()
        
        # Detect cheating
        print("Running cheating detection...")
        self.detect_cheating()
        
        # Create network
        print("Creating player network...")
        self.create_player_network()
        
        # Generate visualizations
        print("Creating visualizations...")
        self.visualize_ratings()
        
        # Generate report
        print("Generating report...")
        self.generate_report()
        
        print("\nAnalysis complete!")

# Example usage
if __name__ == "__main__":
    # Initialize the system
    chess_system = ChessRatingSystem()
    
    # Run full analysis
    # If you have a CSV file, pass it like this:
    chess_system.run_full_analysis("tournament_games.csv")
    
    # For demonstration with sample data:
    chess_system.run_full_analysis()
    
    # You can also run individual components:
    # chess_system.load_data("games.csv")
    # chess_system.process_ratings()
    # suspicious_players = chess_system.detect_cheating()
    # chess_system.visualize_ratings()

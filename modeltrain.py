import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import scrolledtext, messagebox, simpledialog, Toplevel, ttk
from tkinter import filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from glicko2 import Glicko2
import trueskill as ts
from collections import defaultdict
import math


# --- 1. Elo Rating System Implementation ---

class EloRatingSystem:
    def __init__(self, initial_rating=1500, k_factor=30):
        self.ratings = {}  # Stores player_id -> current_rating
        self.rating_history = {}  # Stores player_id -> list of (game_number, rating) tuples
        self.initial_rating = initial_rating
        self.k_factor = k_factor
        self.game_counter = 0
        print(f"Elo System Initialized with: Initial Rating={initial_rating}, K-Factor={k_factor}")

    def get_rating(self, player_id):
        if player_id not in self.ratings:
            self.ratings[player_id] = self.initial_rating
            self.rating_history[player_id] = [(0, self.initial_rating)]
        return self.ratings[player_id]

    def _expected_score(self, rating_a, rating_b):
        return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))

    def update_game(self, white_id, black_id, winner):
        self.game_counter += 1

        rating_white = self.get_rating(white_id)
        rating_black = self.get_rating(black_id)

        expected_white = self._expected_score(rating_white, rating_black)
        expected_black = self._expected_score(rating_black, rating_white)

        if winner == 'white':
            score_white = 1
            score_black = 0
        elif winner == 'black':
            score_white = 0
            score_black = 1
        else:  # draw
            score_white = 0.5
            score_black = 0.5

        new_rating_white = rating_white + self.k_factor * (score_white - expected_white)
        new_rating_black = rating_black + self.k_factor * (score_black - expected_black)

        self.ratings[white_id] = new_rating_white
        self.ratings[black_id] = new_rating_black

        self.rating_history[white_id].append((self.game_counter, new_rating_white))
        self.rating_history[black_id].append((self.game_counter, new_rating_black))

    def get_final_ratings(self):
        return {pid: rating for pid, rating in self.ratings.items()}

    def get_history(self, player_id):
        return self.rating_history.get(player_id, [])

    def get_system_name(self):
        return "Elo"


# --- 2. Glicko2 Rating System Implementation ---

class Glicko2RatingSystem:
    def __init__(self, initial_rating=1500, initial_rd=350, initial_vol=0.06, tau=0.5):
        self.env = Glicko2(tau=tau)
        self.players = {}  # player_id -> Rating object
        self.rating_history = {}  # player_id -> list of (game_number, rating)
        self.game_counter = 0
        self.initial_rating = initial_rating
        self.initial_rd = initial_rd
        self.initial_vol = initial_vol
        print(
            f"Glicko2 System Initialized with: Initial Rating={initial_rating}, Initial RD={initial_rd}, Initial Volatility={initial_vol}, Tau={tau}")

    def get_player_obj(self, player_id):
        if player_id not in self.players:
            player = self.env.create_rating()
            player.rating = self.initial_rating
            player.rd = self.initial_rd
            player.vol = self.initial_vol
            self.players[player_id] = player
            self.rating_history[player_id] = [(0, player.getRating())]
        return self.players[player_id]


from glicko2 import Glicko2


class Glicko2RatingSystem:
    def __init__(self, initial_rating=1500, initial_rd=350, initial_vol=0.06):
        self.env = Glicko2()
        self.players = {}  # player_id -> Rating object
        self.rating_history = {}  # player_id -> list of (game_number, rating)
        self.game_counter = 0
        self.initial_rating = initial_rating
        self.initial_rd = initial_rd
        self.initial_vol = initial_vol
        print(
            f"Glicko2 System Initialized with: Initial Rating={initial_rating}, Initial RD={initial_rd}, Initial Volatility={initial_vol}"
        )

    def get_player_obj(self, player_id):
        if player_id not in self.players:
            # Create player with custom initial values
            player = self.env.create_rating()
            player.rating = self.initial_rating
            player.rd = self.initial_rd
            player.vol = self.initial_vol
            self.players[player_id] = player
            self.rating_history[player_id] = [(0, player.getRating())]
        return self.players[player_id]

    def update_game(self, white_id, black_id, winner):
        self.game_counter += 1

        player_white = self.get_player_obj(white_id)
        player_black = self.get_player_obj(black_id)

        if winner == 'white':
            score_white = 1.0
            score_black = 0.0
        elif winner == 'black':
            score_white = 0.0
            score_black = 1.0
        else:
            score_white = score_black = 0.5

        # Update both players
        player_white.update_player([player_black], [score_white])
        player_black.update_player([player_white], [score_black])

        self.rating_history[white_id].append((self.game_counter, player_white.getRating()))
        self.rating_history[black_id].append((self.game_counter, player_black.getRating()))

    def get_final_ratings(self):
        return {pid: player.getRating() for pid, player in self.players.items()}

    def get_history(self, player_id):
        return self.rating_history.get(player_id, [])


# --- 3. TrueSkill Rating System Implementation ---

class TrueSkillRatingSystem:
    def __init__(self, mu=25.0, sigma=8.333, beta=4.167, tau=0.083, draw_probability=0.10):
        # Set up TrueSkill environment
        self.env = ts.TrueSkill(
            mu=mu,
            sigma=sigma,
            beta=beta,
            tau=tau,
            draw_probability=draw_probability
        )
        ts.setup(
            mu=mu,
            sigma=sigma,
            beta=beta,
            tau=tau,
            draw_probability=draw_probability
        )

        self.players = {}  # player_id -> Rating object
        self.rating_history = {}  # player_id -> list of (game_number, rating)
        self.game_counter = 0
        self.initial_mu = mu
        self.initial_sigma = sigma
        print(
            f"TrueSkill System Initialized with: Mu={mu}, Sigma={sigma}, Beta={beta}, Tau={tau}, Draw Prob={draw_probability}")

    def get_player_obj(self, player_id):
        if player_id not in self.players:
            self.players[player_id] = ts.Rating(mu=self.initial_mu, sigma=self.initial_sigma)
            # Convert to Elo-like scale for display (multiply by 400/ln(10) ≈ 173.7 and add base)
            display_rating = self._trueskill_to_display(self.players[player_id])
            self.rating_history[player_id] = [(0, display_rating)]
        return self.players[player_id]

    def _trueskill_to_display(self, rating):
        """Convert TrueSkill rating to display scale similar to Elo"""
        # Conservative skill estimate: mu - 3*sigma
        conservative_skill = rating.mu - 3 * rating.sigma
        # Scale to Elo-like range (multiply by ~70 and add 1500 base)
        return conservative_skill * 70 + 1500

    def update_game(self, white_id, black_id, winner):
        self.game_counter += 1

        player_white = self.get_player_obj(white_id)
        player_black = self.get_player_obj(black_id)

        if winner == 'white':
            # White wins
            new_white, new_black = ts.rate_1vs1(player_white, player_black)
        elif winner == 'black':
            # Black wins
            new_black, new_white = ts.rate_1vs1(player_black, player_white)
        else:
            # Draw
            new_white, new_black = ts.rate_1vs1(player_white, player_black, drawn=True)

        self.players[white_id] = new_white
        self.players[black_id] = new_black

        # Store display ratings in history
        white_display = self._trueskill_to_display(new_white)
        black_display = self._trueskill_to_display(new_black)

        self.rating_history[white_id].append((self.game_counter, white_display))
        self.rating_history[black_id].append((self.game_counter, black_display))

    def get_final_ratings(self):
        return {pid: self._trueskill_to_display(rating) for pid, rating in self.players.items()}

    def get_history(self, player_id):
        return self.rating_history.get(player_id, [])

    def get_system_name(self):
        return "TrueSkill"


# --- 4. Enhanced GUI Implementation ---

class ChessRatingGUI:
    def __init__(self, master):
        self.master = master
        master.title("Advanced Chess Rating Calculator (Elo + Glicko-2 + TrueSkill)")
        master.geometry("1000x700")

        # Initialize rating systems
        self.elo_system = EloRatingSystem()
        self.glicko2_system = Glicko2RatingSystem()
        self.trueskill_system = TrueSkillRatingSystem()
        self.active_system = None
        self.all_systems = {}  # Store calculated systems for comparison

        self.file_path = ''

        self._create_gui()

    def _create_gui(self):
        # Main notebook for tabs
        self.notebook = ttk.Notebook(self.master)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Create tabs
        self.main_tab = ttk.Frame(self.notebook)
        self.comparison_tab = ttk.Frame(self.notebook)
        self.analysis_tab = ttk.Frame(self.notebook)

        self.notebook.add(self.main_tab, text="Rating Calculation")
        self.notebook.add(self.comparison_tab, text="System Comparison")
        self.notebook.add(self.analysis_tab, text="Advanced Analysis")

        self._create_main_tab()
        self._create_comparison_tab()
        self._create_analysis_tab()

    def _create_main_tab(self):
        main_frame = ttk.Frame(self.main_tab, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # File Input Frame
        file_frame = ttk.LabelFrame(main_frame, text="Select Game Data File", padding="10")
        file_frame.pack(pady=5, fill=tk.X)

        self.file_label = ttk.Label(file_frame, text="No file selected.")
        self.file_label.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5)

        self.browse_button = ttk.Button(file_frame, text="Browse", command=self._browse_file)
        self.browse_button.pack(side=tk.RIGHT, padx=5)

        # Algorithm Selection Frame
        algo_frame = ttk.LabelFrame(main_frame, text="Algorithm Selection", padding="10")
        algo_frame.pack(pady=5, fill=tk.X)

        ttk.Label(algo_frame, text="Select Algorithm:").pack(side=tk.LEFT, padx=5)
        self.algorithm_var = tk.StringVar(self.master)
        self.algorithm_var.set("Elo")
        self.algorithm_options = ["Elo", "Glicko-2", "TrueSkill"]
        self.algorithm_menu = ttk.OptionMenu(algo_frame, self.algorithm_var, self.algorithm_var.get(),
                                             *self.algorithm_options, command=self._update_parameter_inputs)
        self.algorithm_menu.pack(side=tk.LEFT, padx=5)

        # Parameter Customization Frame
        self.param_frame = ttk.LabelFrame(main_frame, text="Algorithm Parameters", padding="10")
        self.param_frame.pack(pady=5, fill=tk.X)

        self._create_parameter_widgets()
        self._update_parameter_inputs()

        # Action Buttons Frame
        button_frame = ttk.Frame(main_frame, padding="5")
        button_frame.pack(pady=5, fill=tk.X)

        self.calculate_button = ttk.Button(button_frame, text="Calculate Ratings", command=self.calculate_ratings)
        self.calculate_button.pack(side=tk.LEFT, expand=True, padx=2)

        self.plot_distribution_button = ttk.Button(button_frame, text="Plot Distribution",
                                                   command=self.plot_rating_distribution, state=tk.DISABLED)
        self.plot_distribution_button.pack(side=tk.LEFT, expand=True, padx=2)

        self.plot_progression_button = ttk.Button(button_frame, text="Plot Progression",
                                                  command=self.plot_player_progression, state=tk.DISABLED)
        self.plot_progression_button.pack(side=tk.LEFT, expand=True, padx=2)

        self.export_button = ttk.Button(button_frame, text="Export Results",
                                        command=self.export_results, state=tk.DISABLED)
        self.export_button.pack(side=tk.LEFT, expand=True, padx=2)

        # Results Text Area
        self.result_text = scrolledtext.ScrolledText(main_frame, width=80, height=25, wrap=tk.WORD)
        self.result_text.pack(pady=10, fill=tk.BOTH, expand=True)
        self.result_text.insert(tk.END, "Welcome to the Advanced Chess Rating Calculator!\n\n")
        self.result_text.insert(tk.END, "Features:\n")
        self.result_text.insert(tk.END, "• Elo Rating System - Classic and simple\n")
        self.result_text.insert(tk.END, "• Glicko-2 - More accurate with uncertainty tracking\n")
        self.result_text.insert(tk.END, "• TrueSkill - Microsoft's advanced Bayesian system\n\n")
        self.result_text.insert(tk.END, "Instructions:\n")
        self.result_text.insert(tk.END, "1. Select a CSV file with columns: white_id, black_id, winner\n")
        self.result_text.insert(tk.END, "2. Choose your rating algorithm and adjust parameters\n")
        self.result_text.insert(tk.END, "3. Click 'Calculate Ratings' to process games\n")
        self.result_text.insert(tk.END, "4. Use visualization and analysis tools\n\n")
        self.result_text.config(state=tk.DISABLED)

    def _create_comparison_tab(self):
        comp_frame = ttk.Frame(self.comparison_tab, padding="10")
        comp_frame.pack(fill=tk.BOTH, expand=True)

        # Comparison controls
        control_frame = ttk.LabelFrame(comp_frame, text="System Comparison", padding="10")
        control_frame.pack(pady=5, fill=tk.X)

        self.calc_all_button = ttk.Button(control_frame, text="Calculate All Systems",
                                          command=self.calculate_all_systems)
        self.calc_all_button.pack(side=tk.LEFT, padx=5)

        self.compare_button = ttk.Button(control_frame, text="Compare Top Players",
                                         command=self.compare_systems, state=tk.DISABLED)
        self.compare_button.pack(side=tk.LEFT, padx=5)

        self.correlation_button = ttk.Button(control_frame, text="Show Correlations",
                                             command=self.show_correlations, state=tk.DISABLED)
        self.correlation_button.pack(side=tk.LEFT, padx=5)

        # Comparison results
        self.comparison_text = scrolledtext.ScrolledText(comp_frame, width=80, height=30, wrap=tk.WORD)
        self.comparison_text.pack(pady=10, fill=tk.BOTH, expand=True)
        self.comparison_text.insert(tk.END, "System Comparison Results will appear here...\n\n")
        self.comparison_text.insert(tk.END,
                                    "Click 'Calculate All Systems' to run Elo, Glicko-2, and TrueSkill on your data.\n")
        self.comparison_text.config(state=tk.DISABLED)

    def _create_analysis_tab(self):
        analysis_frame = ttk.Frame(self.analysis_tab, padding="10")
        analysis_frame.pack(fill=tk.BOTH, expand=True)

        # Analysis controls
        control_frame = ttk.LabelFrame(analysis_frame, text="Advanced Analysis", padding="10")
        control_frame.pack(pady=5, fill=tk.X)

        self.rating_evolution_button = ttk.Button(control_frame, text="Rating Evolution Graph",
                                                  command=self.plot_rating_evolution, state=tk.DISABLED)
        self.rating_evolution_button.pack(side=tk.LEFT, padx=5)

        self.stability_button = ttk.Button(control_frame, text="Rating Stability",
                                           command=self.analyze_stability, state=tk.DISABLED)
        self.stability_button.pack(side=tk.LEFT, padx=5)

        self.head_to_head_button = ttk.Button(control_frame, text="Head-to-Head Analysis",
                                              command=self.head_to_head_analysis, state=tk.DISABLED)
        self.head_to_head_button.pack(side=tk.LEFT, padx=5)

        # Analysis results
        self.analysis_text = scrolledtext.ScrolledText(analysis_frame, width=80, height=30, wrap=tk.WORD)
        self.analysis_text.pack(pady=10, fill=tk.BOTH, expand=True)
        self.analysis_text.insert(tk.END, "Advanced Analysis Results will appear here...\n")
        self.analysis_text.config(state=tk.DISABLED)

    def _create_parameter_widgets(self):
        # Elo Parameters
        self.elo_params = {}
        self.elo_params['initial_rating_label'] = ttk.Label(self.param_frame, text="Initial Rating:")
        self.elo_params['initial_rating_entry'] = ttk.Entry(self.param_frame, width=10)
        self.elo_params['initial_rating_entry'].insert(0, "1500")

        self.elo_params['k_factor_label'] = ttk.Label(self.param_frame, text="K-Factor:")
        self.elo_params['k_factor_entry'] = ttk.Entry(self.param_frame, width=10)
        self.elo_params['k_factor_entry'].insert(0, "30")

        # Glicko-2 Parameters
        self.glicko2_params = {}
        self.glicko2_params['initial_rating_label'] = ttk.Label(self.param_frame, text="Initial Rating:")
        self.glicko2_params['initial_rating_entry'] = ttk.Entry(self.param_frame, width=10)
        self.glicko2_params['initial_rating_entry'].insert(0, "1500")

        self.glicko2_params['initial_rd_label'] = ttk.Label(self.param_frame, text="Initial RD:")
        self.glicko2_params['initial_rd_entry'] = ttk.Entry(self.param_frame, width=10)
        self.glicko2_params['initial_rd_entry'].insert(0, "350")

        self.glicko2_params['initial_vol_label'] = ttk.Label(self.param_frame, text="Initial Volatility:")
        self.glicko2_params['initial_vol_entry'] = ttk.Entry(self.param_frame, width=10)
        self.glicko2_params['initial_vol_entry'].insert(0, "0.06")

        self.glicko2_params['tau_label'] = ttk.Label(self.param_frame, text="Tau:")
        self.glicko2_params['tau_entry'] = ttk.Entry(self.param_frame, width=10)
        self.glicko2_params['tau_entry'].insert(0, "0.5")

        # TrueSkill Parameters
        self.trueskill_params = {}
        self.trueskill_params['mu_label'] = ttk.Label(self.param_frame, text="Initial Mu:")
        self.trueskill_params['mu_entry'] = ttk.Entry(self.param_frame, width=10)
        self.trueskill_params['mu_entry'].insert(0, "25.0")

        self.trueskill_params['sigma_label'] = ttk.Label(self.param_frame, text="Initial Sigma:")
        self.trueskill_params['sigma_entry'] = ttk.Entry(self.param_frame, width=10)
        self.trueskill_params['sigma_entry'].insert(0, "8.333")

        self.trueskill_params['beta_label'] = ttk.Label(self.param_frame, text="Beta:")
        self.trueskill_params['beta_entry'] = ttk.Entry(self.param_frame, width=10)
        self.trueskill_params['beta_entry'].insert(0, "4.167")

        self.trueskill_params['tau_label'] = ttk.Label(self.param_frame, text="Tau:")
        self.trueskill_params['tau_entry'] = ttk.Entry(self.param_frame, width=10)
        self.trueskill_params['tau_entry'].insert(0, "0.083")

        self.trueskill_params['draw_prob_label'] = ttk.Label(self.param_frame, text="Draw Probability:")
        self.trueskill_params['draw_prob_entry'] = ttk.Entry(self.param_frame, width=10)
        self.trueskill_params['draw_prob_entry'].insert(0, "0.10")

    def _browse_file(self):
        filetypes = [("CSV files", "*.csv"), ("All files", "*.*")]
        filepath = filedialog.askopenfilename(
            title="Select Chess Game CSV File",
            filetypes=filetypes
        )
        if filepath:
            self.file_path = filepath
            self.file_label.config(text=f"Selected: {filepath.split('/')[-1]}")
            self._update_result_text(f"File selected: {self.file_path}\n", append=False)
        else:
            self.file_path = ''
            self.file_label.config(text="No file selected.")

    def _update_parameter_inputs(self, *args):
        selected_algorithm = self.algorithm_var.get()

        # Hide all parameters first
        for widget_set in [self.elo_params, self.glicko2_params, self.trueskill_params]:
            for widget in widget_set.values():
                widget.pack_forget()

        # Show relevant parameters
        if selected_algorithm == "Elo":
            params = [
                ('initial_rating_label', 'initial_rating_entry'),
                ('k_factor_label', 'k_factor_entry')
            ]
            for label_key, entry_key in params:
                self.elo_params[label_key].pack(side=tk.LEFT, padx=5)
                self.elo_params[entry_key].pack(side=tk.LEFT, padx=5)

        elif selected_algorithm == "Glicko-2":
            params = [
                ('initial_rating_label', 'initial_rating_entry'),
                ('initial_rd_label', 'initial_rd_entry'),
                ('initial_vol_label', 'initial_vol_entry'),
                ('tau_label', 'tau_entry')
            ]
            for label_key, entry_key in params:
                self.glicko2_params[label_key].pack(side=tk.LEFT, padx=5)
                self.glicko2_params[entry_key].pack(side=tk.LEFT, padx=5)

        elif selected_algorithm == "TrueSkill":
            params = [
                ('mu_label', 'mu_entry'),
                ('sigma_label', 'sigma_entry'),
                ('beta_label', 'beta_entry'),
                ('tau_label', 'tau_entry'),
                ('draw_prob_label', 'draw_prob_entry')
            ]
            for label_key, entry_key in params:
                self.trueskill_params[label_key].pack(side=tk.LEFT, padx=5)
                self.trueskill_params[entry_key].pack(side=tk.LEFT, padx=5)

    def _update_result_text(self, text, append=True, target=None):
        if target is None:
            target = self.result_text
        target.config(state=tk.NORMAL)
        if not append:
            target.delete(1.0, tk.END)
        target.insert(tk.END, text)
        target.config(state=tk.DISABLED)
        target.see(tk.END)
        self.master.update_idletasks()

    def calculate_ratings(self):
        if not self.file_path:
            messagebox.showwarning("No File Selected", "Please select a CSV file first.")
            return

        selected_algorithm = self.algorithm_var.get()
        self._update_result_text(f"=== {selected_algorithm} Rating Calculation ===\n", append=False)

        try:
            # Initialize system with parameters
            if selected_algorithm == "Elo":
                initial_rating = float(self.elo_params['initial_rating_entry'].get())
                k_factor = float(self.elo_params['k_factor_entry'].get())
                self.elo_system = EloRatingSystem(initial_rating=initial_rating, k_factor=k_factor)
                self.active_system = self.elo_system

            elif selected_algorithm == "Glicko-2":
                initial_rating = float(self.glicko2_params['initial_rating_entry'].get())
                initial_rd = float(self.glicko2_params['initial_rd_entry'].get())
                initial_vol = float(self.glicko2_params['initial_vol_entry'].get())
                tau = float(self.glicko2_params['tau_entry'].get())
                self.glicko2_system = Glicko2RatingSystem(initial_rating, initial_rd, initial_vol, tau)
                self.active_system = self.glicko2_system

            elif selected_algorithm == "TrueSkill":
                mu = float(self.trueskill_params['mu_entry'].get())
                sigma = float(self.trueskill_params['sigma_entry'].get())
                beta = float(self.trueskill_params['beta_entry'].get())
                tau = float(self.trueskill_params['tau_entry'].get())
                draw_prob = float(self.trueskill_params['draw_prob_entry'].get())
                self.trueskill_system = TrueSkillRatingSystem(mu, sigma, beta, tau, draw_prob)
                self.active_system = self.trueskill_system

        except ValueError as e:
            messagebox.showerror("Parameter Error", f"Invalid parameter: {e}")
            return

        # Load and process data
        try:
            df = pd.read_csv(self.file_path)
            self._update_result_text(f"Loaded {len(df)} games from CSV\n")

            # Validate required columns
            required_columns = ['white_id', 'black_id', 'winner']
            if not all(col in df.columns for col in required_columns):
                raise ValueError(f"CSV must contain columns: {required_columns}")

            # Process games
            processed_games = 0
            for index, row in df.iterrows():
                white_player = str(row['white_id'])
                black_player = str(row['black_id'])
                game_winner = str(row['winner']).lower().strip()

                if game_winner in ['white', 'black', 'draw']:
                    self.active_system.update_game(white_player, black_player, game_winner)
                    processed_games += 1

            self._update_result_text(f"Processed {processed_games} valid games\n\n")

            # Display results
            final_ratings = self.active_system.get_final_ratings()
            sorted_ratings = sorted(final_ratings.items(), key=lambda x: x[1], reverse=True)

            self._update_result_text("=== TOP 20 PLAYERS ===\n")
            for i, (player_id, rating) in enumerate(sorted_ratings[:20], 1):
                self._update_result_text(f"{i:2d}. {player_id:<20} {rating:7.2f}\n")

            self._update_result_text(f"\nTotal players: {len(final_ratings)}\n")
            self._update_result_text(f"Average rating: {np.mean(list(final_ratings.values())):.2f}\n")
            self._update_result_text(f"Rating std dev: {np.std(list(final_ratings.values())):.2f}\n")

            # Enable buttons
            self.plot_distribution_button.config(state=tk.NORMAL)
            self.plot_progression_button.config(state=tk.NORMAL)
            self.export_button.config(state=tk.NORMAL)
            self.rating_evolution_button.config(state=tk.NORMAL)
            self.stability_button.config(state=tk.NORMAL)
            self.head_to_head_button.config(state=tk.NORMAL)

            messagebox.showinfo("Success", f"{selected_algorithm} calculation complete!")

        except Exception as e:
            messagebox.showerror("Error", f"Error processing data: {e}")
            self._update_result_text(f"Error: {e}\n")

    def calculate_all_systems(self):
        if not self.file_path:
            messagebox.showwarning("No File Selected", "Please select a CSV file first.")
            return

        self._update_result_text("=== CALCULATING ALL RATING SYSTEMS ===\n", append=False, target=self.comparison_text)

        try:
            df = pd.read_csv(self.file_path)
            systems = {
                'Elo': EloRatingSystem(),
                'Glicko-2': Glicko2RatingSystem(),
                'TrueSkill': TrueSkillRatingSystem()
            }

            # Process all games for each system
            for system_name, system in systems.items():
                self._update_result_text(f"Processing {system_name}...\n", target=self.comparison_text)

                for index, row in df.iterrows():
                    white_player = str(row['white_id'])
                    black_player = str(row['black_id'])
                    game_winner = str(row['winner']).lower().strip()

                    if game_winner in ['white', 'black', 'draw']:
                        system.update_game(white_player, black_player, game_winner)

            self.all_systems = systems
            self._update_result_text("All systems calculated successfully!\n\n", target=self.comparison_text)

            # Enable comparison buttons
            self.compare_button.config(state=tk.NORMAL)
            self.correlation_button.config(state=tk.NORMAL)

        except Exception as e:
            messagebox.showerror("Error", f"Error calculating systems: {e}")

    def compare_systems(self):
        if not self.all_systems:
            messagebox.showwarning("No Data", "Please calculate all systems first.")
            return

        self._update_result_text("=== SYSTEM COMPARISON - TOP 15 PLAYERS ===\n", target=self.comparison_text)

        # Get all player ratings from each system
        all_ratings = {}
        for system_name, system in self.all_systems.items():
            all_ratings[system_name] = system.get_final_ratings()

        # Get all unique players
        all_players = set()
        for ratings in all_ratings.values():
            all_players.update(ratings.keys())

        # Create comparison data
        comparison_data = []
        for player in all_players:
            row = {'player': player}
            for system_name in self.all_systems.keys():
                row[system_name] = all_ratings[system_name].get(player, 0)
            comparison_data.append(row)

        # Sort by Elo rating
        comparison_data.sort(key=lambda x: x['Elo'], reverse=True)

        # Display comparison table
        header = f"{'Rank':<4} {'Player':<20} {'Elo':<8} {'Glicko-2':<8} {'TrueSkill':<10}\n"
        self._update_result_text(header, target=self.comparison_text)
        self._update_result_text("-" * len(header) + "\n", target=self.comparison_text)

        for i, data in enumerate(comparison_data[:15], 1):
            line = f"{i:<4} {data['player']:<20} {data['Elo']:<8.1f} {data['Glicko-2']:<8.1f} {data['TrueSkill']:<10.1f}\n"
            self._update_result_text(line, target=self.comparison_text)

    def show_correlations(self):
        if not self.all_systems:
            messagebox.showwarning("No Data", "Please calculate all systems first.")
            return

        # Get ratings for correlation analysis
        all_ratings = {}
        for system_name, system in self.all_systems.items():
            all_ratings[system_name] = system.get_final_ratings()

        # Find common players
        common_players = set(all_ratings['Elo'].keys())
        for ratings in all_ratings.values():
            common_players &= set(ratings.keys())

        if len(common_players) < 2:
            messagebox.showwarning("Insufficient Data", "Need at least 2 common players for correlation.")
            return

        # Calculate correlations
        systems = list(all_ratings.keys())
        correlations = {}

        self._update_result_text("\n=== RATING SYSTEM CORRELATIONS ===\n", target=self.comparison_text)

        for i, sys1 in enumerate(systems):
            for j, sys2 in enumerate(systems):
                if i < j:  # Only calculate upper triangle
                    ratings1 = [all_ratings[sys1][player] for player in common_players]
                    ratings2 = [all_ratings[sys2][player] for player in common_players]

                    correlation = np.corrcoef(ratings1, ratings2)[0, 1]
                    correlations[f"{sys1} vs {sys2}"] = correlation

                    self._update_result_text(f"{sys1} vs {sys2}: {correlation:.4f}\n", target=self.comparison_text)

        self._update_result_text(f"\nBased on {len(common_players)} common players\n", target=self.comparison_text)

        # Create correlation visualization
        self._plot_correlation_matrix()

    def _plot_correlation_matrix(self):
        if not self.all_systems:
            return

        # Get ratings for all systems
        all_ratings = {}
        for system_name, system in self.all_systems.items():
            all_ratings[system_name] = system.get_final_ratings()

        # Find common players
        common_players = set(all_ratings['Elo'].keys())
        for ratings in all_ratings.values():
            common_players &= set(ratings.keys())

        # Create correlation matrix
        systems = list(all_ratings.keys())
        n_systems = len(systems)
        correlation_matrix = np.zeros((n_systems, n_systems))

        for i, sys1 in enumerate(systems):
            for j, sys2 in enumerate(systems):
                if i == j:
                    correlation_matrix[i, j] = 1.0
                else:
                    ratings1 = [all_ratings[sys1][player] for player in common_players]
                    ratings2 = [all_ratings[sys2][player] for player in common_players]
                    correlation_matrix[i, j] = np.corrcoef(ratings1, ratings2)[0, 1]

        # Plot heatmap
        plt.figure(figsize=(8, 6))
        plt.imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        plt.colorbar(label='Correlation')
        plt.xticks(range(n_systems), systems)
        plt.yticks(range(n_systems), systems)
        plt.title('Rating System Correlations')

        # Add correlation values as text
        for i in range(n_systems):
            for j in range(n_systems):
                plt.text(j, i, f'{correlation_matrix[i, j]:.3f}',
                         ha='center', va='center', color='black')

        plt.tight_layout()
        plt.show()

    def plot_rating_distribution(self):
        if self.active_system is None:
            messagebox.showwarning("No Data", "Please calculate ratings first.")
            return

        ratings = list(self.active_system.get_final_ratings().values())
        system_name = self.active_system.get_system_name()

        plt.figure(figsize=(12, 8))

        # Main histogram
        plt.subplot(2, 2, 1)
        plt.hist(ratings, bins=30, edgecolor='black', alpha=0.7, color='skyblue')
        plt.title(f'{system_name} Rating Distribution')
        plt.xlabel(f'{system_name} Rating')
        plt.ylabel('Number of Players')
        plt.grid(axis='y', alpha=0.3)

        # Box plot
        plt.subplot(2, 2, 2)
        plt.boxplot(ratings, vert=True)
        plt.title(f'{system_name} Rating Box Plot')
        plt.ylabel(f'{system_name} Rating')
        plt.grid(axis='y', alpha=0.3)

        # Statistics
        plt.subplot(2, 2, 3)
        plt.axis('off')
        stats_text = f"""
Statistics for {system_name}:
Players: {len(ratings)}
Mean: {np.mean(ratings):.2f}
Median: {np.median(ratings):.2f}
Std Dev: {np.std(ratings):.2f}
Min: {min(ratings):.2f}
Max: {max(ratings):.2f}
Q1: {np.percentile(ratings, 25):.2f}
Q3: {np.percentile(ratings, 75):.2f}
        """
        plt.text(0.1, 0.9, stats_text, transform=plt.gca().transAxes,
                 fontsize=10, verticalalignment='top', fontfamily='monospace')

        # Cumulative distribution
        plt.subplot(2, 2, 4)
        sorted_ratings = np.sort(ratings)
        cumulative = np.arange(1, len(sorted_ratings) + 1) / len(sorted_ratings)
        plt.plot(sorted_ratings, cumulative, linewidth=2)
        plt.title(f'{system_name} Cumulative Distribution')
        plt.xlabel(f'{system_name} Rating')
        plt.ylabel('Cumulative Probability')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def plot_player_progression(self):
        if self.active_system is None:
            messagebox.showwarning("No Data", "Please calculate ratings first.")
            return

        # Get player input
        player_dialog = PlayerSelectionDialog(self.master, list(self.active_system.get_final_ratings().keys()))
        self.master.wait_window(player_dialog.dialog)

        if not player_dialog.selected_players:
            return

        system_name = self.active_system.get_system_name()

        plt.figure(figsize=(14, 8))

        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']

        for i, player_id in enumerate(player_dialog.selected_players):
            history = self.active_system.get_history(player_id)
            if not history:
                continue

            game_numbers = [item[0] for item in history]
            ratings = [item[1] for item in history]

            color = colors[i % len(colors)]
            plt.plot(game_numbers, ratings, marker='o', linestyle='-',
                     markersize=3, label=player_id, color=color, linewidth=1.5)

        plt.title(f'{system_name} Rating Progression')
        plt.xlabel('Game Number')
        plt.ylabel(f'{system_name} Rating')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def plot_rating_evolution(self):
        if self.active_system is None:
            messagebox.showwarning("No Data", "Please calculate ratings first.")
            return

        # Get top 10 players for evolution plot
        final_ratings = self.active_system.get_final_ratings()
        top_players = sorted(final_ratings.items(), key=lambda x: x[1], reverse=True)[:10]

        plt.figure(figsize=(15, 10))

        colors = plt.cm.tab10(np.linspace(0, 1, len(top_players)))

        for i, (player_id, final_rating) in enumerate(top_players):
            history = self.active_system.get_history(player_id)
            if len(history) < 2:
                continue

            game_numbers = [item[0] for item in history]
            ratings = [item[1] for item in history]

            plt.plot(game_numbers, ratings, marker='o', linestyle='-',
                     markersize=4, label=f'{player_id} (Final: {final_rating:.1f})',
                     color=colors[i], linewidth=2, alpha=0.8)

        system_name = self.active_system.get_system_name()
        plt.title(f'Rating Evolution - Top 10 Players ({system_name})')
        plt.xlabel('Game Number')
        plt.ylabel(f'{system_name} Rating')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def analyze_stability(self):
        if self.active_system is None:
            messagebox.showwarning("No Data", "Please calculate ratings first.")
            return

        self._update_result_text("=== RATING STABILITY ANALYSIS ===\n", append=False, target=self.analysis_text)

        system_name = self.active_system.get_system_name()
        final_ratings = self.active_system.get_final_ratings()

        stability_data = []

        for player_id, final_rating in final_ratings.items():
            history = self.active_system.get_history(player_id)
            if len(history) < 5:  # Need at least 5 games for stability analysis
                continue

            ratings = [item[1] for item in history[1:]]  # Skip initial rating

            # Calculate stability metrics
            rating_std = np.std(ratings)
            rating_range = max(ratings) - min(ratings)
            final_10_games = ratings[-10:] if len(ratings) >= 10 else ratings
            recent_std = np.std(final_10_games)

            stability_data.append({
                'player': player_id,
                'final_rating': final_rating,
                'games': len(history) - 1,
                'std_dev': rating_std,
                'rating_range': rating_range,
                'recent_std': recent_std
            })

        # Sort by final rating
        stability_data.sort(key=lambda x: x['final_rating'], reverse=True)

        self._update_result_text(f"Rating Stability Analysis ({system_name})\n", target=self.analysis_text)
        self._update_result_text("=" * 80 + "\n", target=self.analysis_text)

        header = f"{'Player':<20} {'Rating':<8} {'Games':<6} {'Std Dev':<8} {'Range':<8} {'Recent Std':<10}\n"
        self._update_result_text(header, target=self.analysis_text)
        self._update_result_text("-" * 80 + "\n", target=self.analysis_text)

        for data in stability_data[:20]:  # Top 20
            line = f"{data['player']:<20} {data['final_rating']:<8.1f} {data['games']:<6} {data['std_dev']:<8.1f} {data['rating_range']:<8.1f} {data['recent_std']:<10.1f}\n"
            self._update_result_text(line, target=self.analysis_text)

        # Summary statistics
        all_stds = [d['std_dev'] for d in stability_data]
        all_ranges = [d['rating_range'] for d in stability_data]

        self._update_result_text(f"\nSummary:\n", target=self.analysis_text)
        self._update_result_text(f"Average rating std dev: {np.mean(all_stds):.2f}\n", target=self.analysis_text)
        self._update_result_text(f"Average rating range: {np.mean(all_ranges):.2f}\n", target=self.analysis_text)
        self._update_result_text(f"Most stable player: {min(stability_data, key=lambda x: x['std_dev'])['player']}\n",
                                 target=self.analysis_text)
        self._update_result_text(f"Most volatile player: {max(stability_data, key=lambda x: x['std_dev'])['player']}\n",
                                 target=self.analysis_text)

    def head_to_head_analysis(self):
        if not self.file_path:
            messagebox.showwarning("No File Selected", "Please select a CSV file first.")
            return

        # Get two players for head-to-head analysis
        dialog = HeadToHeadDialog(self.master)
        self.master.wait_window(dialog.dialog)

        if not dialog.player1 or not dialog.player2:
            return

        player1, player2 = dialog.player1, dialog.player2

        # Analyze head-to-head record
        try:
            df = pd.read_csv(self.file_path)

            # Filter games between these two players
            h2h_games = df[
                ((df['white_id'] == player1) & (df['black_id'] == player2)) |
                ((df['white_id'] == player2) & (df['black_id'] == player1))
                ]

            if len(h2h_games) == 0:
                messagebox.showinfo("No Games", f"No games found between {player1} and {player2}")
                return

            self._update_result_text(f"=== HEAD-TO-HEAD ANALYSIS ===\n", append=False, target=self.analysis_text)
            self._update_result_text(f"{player1} vs {player2}\n\n", target=self.analysis_text)

            # Count results
            p1_wins = 0
            p2_wins = 0
            draws = 0

            p1_as_white = 0
            p1_as_black = 0

            for _, game in h2h_games.iterrows():
                if game['white_id'] == player1:
                    p1_as_white += 1
                    if game['winner'] == 'white':
                        p1_wins += 1
                    elif game['winner'] == 'black':
                        p2_wins += 1
                    else:
                        draws += 1
                else:  # player1 is black
                    p1_as_black += 1
                    if game['winner'] == 'black':
                        p1_wins += 1
                    elif game['winner'] == 'white':
                        p2_wins += 1
                    else:
                        draws += 1

            total_games = len(h2h_games)
            p1_score = p1_wins + 0.5 * draws
            p2_score = p2_wins + 0.5 * draws

            self._update_result_text(f"Total games: {total_games}\n", target=self.analysis_text)
            self._update_result_text(
                f"{player1}: {p1_wins} wins, {draws} draws, {p2_wins} losses (Score: {p1_score}/{total_games})\n",
                target=self.analysis_text)
            self._update_result_text(
                f"{player2}: {p2_wins} wins, {draws} draws, {p1_wins} losses (Score: {p2_score}/{total_games})\n",
                target=self.analysis_text)
            self._update_result_text(f"\nColor distribution:\n", target=self.analysis_text)
            self._update_result_text(f"{player1} as White: {p1_as_white} games\n", target=self.analysis_text)
            self._update_result_text(f"{player1} as Black: {p1_as_black} games\n", target=self.analysis_text)

            if p1_score > p2_score:
                self._update_result_text(f"\n{player1} leads the head-to-head record.\n", target=self.analysis_text)
            elif p2_score > p1_score:
                self._update_result_text(f"\n{player2} leads the head-to-head record.\n", target=self.analysis_text)
            else:
                self._update_result_text(f"\nThe head-to-head record is tied.\n", target=self.analysis_text)

        except Exception as e:
            messagebox.showerror("Error", f"Error in head-to-head analysis: {e}")

    def export_results(self):
        if self.active_system is None:
            messagebox.showwarning("No Data", "Please calculate ratings first.")
            return

        # Get export file path
        filepath = filedialog.asksaveasfilename(
            title="Export Results",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )

        if not filepath:
            return

        try:
            # Prepare data for export
            final_ratings = self.active_system.get_final_ratings()
            system_name = self.active_system.get_system_name()

            export_data = []
            for player_id, rating in final_ratings.items():
                history = self.active_system.get_history(player_id)
                games_played = len(history) - 1 if history else 0

                export_data.append({
                    'player_id': player_id,
                    f'{system_name.lower()}_rating': rating,
                    'games_played': games_played
                })

            # Create DataFrame and export
            df = pd.DataFrame(export_data)
            df = df.sort_values(f'{system_name.lower()}_rating', ascending=False)
            df.to_csv(filepath, index=False)

            messagebox.showinfo("Export Complete", f"Results exported to {filepath}")

        except Exception as e:
            messagebox.showerror("Export Error", f"Error exporting results: {e}")


# --- 5. Helper Dialog Classes ---

class PlayerSelectionDialog:
    def __init__(self, parent, player_list):
        self.selected_players = []

        self.dialog = Toplevel(parent)
        self.dialog.title("Select Players for Progression Plot")
        self.dialog.geometry("400x500")
        self.dialog.transient(parent)
        self.dialog.grab_set()

        # Instructions
        ttk.Label(self.dialog, text="Select up to 8 players to plot:").pack(pady=10)

        # Listbox with scrollbar
        frame = ttk.Frame(self.dialog)
        frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        scrollbar = ttk.Scrollbar(frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.listbox = tk.Listbox(frame, yscrollcommand=scrollbar.set, selectmode=tk.MULTIPLE)
        self.listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.listbox.yview)

        # Populate listbox
        sorted_players = sorted(player_list)
        for player in sorted_players:
            self.listbox.insert(tk.END, player)

        # Buttons
        button_frame = ttk.Frame(self.dialog)
        button_frame.pack(pady=10)

        ttk.Button(button_frame, text="OK", command=self.ok_clicked).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=self.cancel_clicked).pack(side=tk.LEFT, padx=5)

    def ok_clicked(self):
        selection_indices = self.listbox.curselection()
        if len(selection_indices) > 8:
            messagebox.showwarning("Too Many Players", "Please select no more than 8 players.")
            return

        self.selected_players = [self.listbox.get(i) for i in selection_indices]
        self.dialog.destroy()

    def cancel_clicked(self):
        self.selected_players = []
        self.dialog.destroy()


class HeadToHeadDialog:
    def __init__(self, parent):
        self.player1 = None
        self.player2 = None

        self.dialog = Toplevel(parent)
        self.dialog.title("Head-to-Head Analysis")
        self.dialog.geometry("300x200")
        self.dialog.transient(parent)
        self.dialog.grab_set()

        # Player 1
        ttk.Label(self.dialog, text="Player 1 ID:").pack(pady=5)
        self.entry1 = ttk.Entry(self.dialog, width=30)
        self.entry1.pack(pady=5)

        # Player 2
        ttk.Label(self.dialog, text="Player 2 ID:").pack(pady=5)
        self.entry2 = ttk.Entry(self.dialog, width=30)
        self.entry2.pack(pady=5)

        # Buttons
        button_frame = ttk.Frame(self.dialog)
        button_frame.pack(pady=20)

        ttk.Button(button_frame, text="Analyze", command=self.analyze_clicked).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=self.cancel_clicked).pack(side=tk.LEFT, padx=5)

    def analyze_clicked(self):
        self.player1 = self.entry1.get().strip()
        self.player2 = self.entry2.get().strip()

        if not self.player1 or not self.player2:
            messagebox.showwarning("Invalid Input", "Please enter both player IDs.")
            return

        self.dialog.destroy()

    def cancel_clicked(self):
        self.dialog.destroy()


# --- 6. Main Application ---

if __name__ == "__main__":
    root = tk.Tk()
    app = ChessRatingGUI(root)
    root.mainloop()

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

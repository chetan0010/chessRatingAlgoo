class EloRating:
    def __init__(self, k_factor=32):
        self.k = k_factor
        self.players = {}

    def get_rating(self, player):
        return self.players.get(player, 1200)

    def expected_score(self, rating_a, rating_b):
        return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))

    def update_ratings(self, winner, loser):
        Ra = self.get_rating(winner)
        Rb = self.get_rating(loser)
        expected_winner = self.expected_score(Ra, Rb)
        expected_loser = self.expected_score(Rb, Ra)

        self.players[winner] = Ra + self.k * (1 - expected_winner)
        self.players[loser] = Rb + self.k * (0 - expected_loser)


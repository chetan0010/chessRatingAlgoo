import joblib
import numpy as np
import chess

# Load the trained model
model = joblib.load("models/move_predicted.pkl")

# Convert FEN to numerical features (same as in training)
def fen_to_features(fen):
    board = chess.Board(fen)
    piece_map = board.piece_map()
    features = [0] * 64
    for square, piece in piece_map.items():
        features[square] = piece.piece_type * (1 if piece.color == chess.WHITE else -1)
    return np.array(features).reshape(1, -1)

# Test FEN position (starting position)
test_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

# Predict move
predicted_move = model.predict(fen_to_features(test_fen))

print(f"✅ Predicted Move: {predicted_move[0]}")

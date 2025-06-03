import os
import joblib
import chess

# Define base directory
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
model_path = os.path.join(BASE_DIR, "models", "move_predicted.pkl")

# Check if model exists
if not os.path.exists(model_path):
    raise FileNotFoundError(f"❌ Error: Model file not found at {model_path}")

# Load model
model = joblib.load(model_path)
print(f"✅ Model loaded successfully from '{model_path}'")

# Convert FEN to numerical features
def fen_to_features(fen):
    board = chess.Board(fen)
    piece_map = board.piece_map()
    features = [0] * 64  # Initialize a list with 64 zeros

    for square, piece in piece_map.items():
        features[square] = piece.piece_type * (1 if piece.color == chess.WHITE else -1)

    return features

# Predict the best move
def predict_move(fen):
    features = fen_to_features(fen)
    prediction = model.predict([features])[0]  # Get predicted move ID (hashed)
    return prediction

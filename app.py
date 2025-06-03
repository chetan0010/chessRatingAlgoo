from flask import Flask, render_template, request, jsonify
from predict_move import predict_move

app = Flask(__name__, static_folder='../static', template_folder='../templates')

# Route for home page (index.html)
@app.route('/')
def index():
    return render_template('index.html')

# Route for leaderboard page (leaderboard.html)
@app.route('/leaderboard')
def leaderboard():
    return render_template('leaderboard.html')
import joblib  # Or use torch.load() / tensorflow.load_model() depending on your ML framework

# Load your trained model
import os

# ✅ Correct path to the model file
model_path = r"C:\moviesgames\chess-lastp\models\move_predicted.pkl"

# ✅ Check if the file exists before loading
if os.path.exists(model_path):
    print(f"✅ Model loaded successfully from '{model_path}'")
    model = joblib.load(model_path)
else:
    raise FileNotFoundError(f"❌ Model file not found at: {model_path}. Train and save the model first.")

def decode_move(move_int):
    """Converts an integer output from the model into a chess move (e.g., 'e2e4')."""
    move_mapping = {
        75154: "e2e4",
        12542: "d2d4",
        65241: "g1f3"
        # Add more mappings based on your dataset
    }
    return move_mapping.get(move_int, "unknown")  # Default to 'unknown' if not found
import numpy as np
import chess

def fen_to_features(fen):
    """Converts a FEN string into a numerical feature vector for ML input."""
    board = chess.Board(fen)
    piece_map = board.piece_map()

    feature_vector = np.zeros(64, dtype=int)  # 64 squares

    piece_encoding = {
        'p': -1, 'n': -2, 'b': -3, 'r': -4, 'q': -5, 'k': -6,  # Black
        'P': 1, 'N': 2, 'B': 3, 'R': 4, 'Q': 5, 'K': 6         # White
    }

    for square, piece in piece_map.items():
        feature_vector[square] = piece_encoding.get(piece.symbol(), 0)

    return feature_vector.reshape(1, -1)  # Reshape for model input



def predict_move(fen):
    # Ensure this function returns a move like "e2e4" instead of an integer
    predicted_move = model.predict([fen])  # Get model output

    if isinstance(predicted_move, int):  # If the output is an integer
        predicted_move = decode_move(predicted_move)


    return predicted_move  # Now it returns a proper chess move

from flask import Flask, request, jsonify
import chess

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    fen = data.get("fen", "")

    if not fen:
        return jsonify({"error": "No FEN provided"}), 400

    try:
        board = chess.Board(fen)  # Validate FEN
    except ValueError:
        return jsonify({"error": "Invalid FEN format"}), 400

    print(f"✅ Received FEN: {fen}")  # Debugging print

    # 🔥 Convert FEN to numerical features
    features = fen_to_features(fen)

    # 🔥 Predict move
    predicted_move_index = model.predict(features)[0]  # Model outputs an index
    predicted_move = decode_move(predicted_move_index)  # Convert index to chess move

    return jsonify({"predicted_move": predicted_move})




if __name__ == '__main__':
    app.run(debug=True)


#API endpoint for predicting chess moves
#API endpoint for leaderboard data
@app.route('/leaderboard-data')
def leaderboard_data():
    leaderboard = [
        {"name": "Magnus Carlsen", "elo": 2850},
        {"name": "Hikaru Nakamura", "elo": 2800},
        {"name": "Ian Nepomniachtchi", "elo": 2755}
    ]
    return jsonify(leaderboard)



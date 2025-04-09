from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import json
import re

app = Flask(__name__)

# Load tokenizer
with open("model/tokenizer.json") as f:
    tokenizer = tokenizer_from_json(f.read())

# Load model
model = tf.keras.models.load_model("model/sentiment_with_attention.keras", compile=False)

# Clean input
def clean_text(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"[^A-Za-z\s]", "", text)
    return text.lower().strip()

@app.route("/")
def index():
    return "Sentiment API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    text = data["text"]
    cleaned = clean_text(text)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=100, padding='post')
    pred = model.predict(padded)[0][0]
    sentiment = "Positive 😊" if pred > 0.5 else "Negative 😞"
    confidence = round(float(pred) * 100, 2)
    return jsonify({"sentiment": sentiment, "confidence": f"{confidence}%"})

if __name__ == "__main__":
    app.run(debug=True)

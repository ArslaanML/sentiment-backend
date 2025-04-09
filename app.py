import os
import json
import re
import numpy as np
from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.layers import Layer
from tensorflow.keras.utils import register_keras_serializable

# Register the custom attention layer
@register_keras_serializable()
class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.att_weight = self.add_weight(
            name="att_weight",
            shape=(input_shape[-1], 1),
            initializer="random_normal",
            trainable=True
        )
        self.att_bias = self.add_weight(
            name="att_bias",
            shape=(input_shape[1], 1),
            initializer="zeros",
            trainable=True
        )
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs):
        score = tf.nn.tanh(tf.tensordot(inputs, self.att_weight, axes=1) + self.att_bias)
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * inputs
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector

# Load tokenizer from JSON
with open("model/tokenizer.json", "r") as f:
    tokenizer = tokenizer_from_json(f.read())

# Load the model
model = tf.keras.models.load_model(
    "model/sentiment_model.keras",
    custom_objects={"AttentionLayer": AttentionLayer},
    compile=False
)

# Flask app
app = Flask(__name__)

# Text cleaning
def clean_text(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"[^A-Za-z\s]", "", text)
    return text.lower().strip()

# Prediction function
def predict_sentiment(text):
    cleaned = clean_text(text)
    sequence = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(sequence, maxlen=100, padding="post")
    prediction = model.predict(padded)[0][0]
    sentiment = "Positive" if prediction > 0.5 else "Negative"
    confidence = round(float(prediction) * 100, 2)
    return sentiment, confidence

# Root route
@app.route("/")
def home():
    return "Sentiment API is running!"

# API route
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text' in request"}), 400

    text = data["text"]
    sentiment, confidence = predict_sentiment(text)

    return jsonify({
        "text": text,
        "sentiment": sentiment,
        "confidence": f"{confidence}%"
    })

# Run locally (for dev)
if __name__ == "__main__":
    app.run(debug=True)

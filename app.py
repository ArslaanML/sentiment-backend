from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.layers import Layer
from tensorflow.keras.utils import register_keras_serializable
import json
import re
import os

# Register custom Attention layer
@register_keras_serializable()
class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.att_weight = self.add_weight(name="att_weight",
                                          shape=(input_shape[-1], 1),
                                          initializer="random_normal",
                                          trainable=True)
        self.att_bias = self.add_weight(name="att_bias",
                                        shape=(input_shape[1], 1),
                                        initializer="zeros",
                                        trainable=True)
        super().build(input_shape)

    def call(self, inputs):
        score = tf.nn.tanh(tf.tensordot(inputs, self.att_weight, axes=1) + self.att_bias)
        weights = tf.nn.softmax(score, axis=1)
        context = weights * inputs
        return tf.reduce_sum(context, axis=1)

# Clean text
def clean_text(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"[^A-Za-z\s]", "", text)
    return text.lower().strip()

# Load model & tokenizer
model = tf.keras.models.load_model("model/sentiment_model.keras", custom_objects={"AttentionLayer": AttentionLayer}, compile=False)

with open("model/tokenizer.json") as f:
    tokenizer = tokenizer_from_json(f.read())

# Flask app
app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "No text provided"}), 400

    text = clean_text(data["text"])
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=100, padding="post")
    prediction = float(model.predict(padded)[0][0])
    sentiment = "Positive" if prediction > 0.5 else "Negative"
    confidence = round(prediction * 100, 2)
    return jsonify({"sentiment": sentiment, "confidence": confidence})

if __name__ == "__main__":
    app.run(debug=True)

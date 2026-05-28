import json
import re
from pathlib import Path

#import the libraries
import h5py
import numpy as np
import streamlit as st

MAX_FEATURES = 10000
MAX_REVIEW_LENGTH = 500
INDEX_FROM = 3
OOV_TOKEN = 2
TOKEN_PATTERN = re.compile(r"[a-z0-9']+")
BASE_DIR = Path(__file__).resolve().parent


class NumpySimpleRNN:
    def __init__(self, model_path):
        with h5py.File(model_path, "r") as model_file:
            self.embeddings = model_file[
                "model_weights/embedding/sequential/embedding/embeddings"
            ][()].astype(np.float32)
            self.rnn_kernel = model_file[
                "model_weights/simple_rnn/sequential/simple_rnn/simple_rnn_cell/kernel"
            ][()].astype(np.float32)
            self.rnn_recurrent_kernel = model_file[
                "model_weights/simple_rnn/sequential/simple_rnn/simple_rnn_cell/recurrent_kernel"
            ][()].astype(np.float32)
            self.rnn_bias = model_file[
                "model_weights/simple_rnn/sequential/simple_rnn/simple_rnn_cell/bias"
            ][()].astype(np.float32)
            self.dense_kernel = model_file[
                "model_weights/dense/sequential/dense/kernel"
            ][()].astype(np.float32)
            self.dense_bias = model_file[
                "model_weights/dense/sequential/dense/bias"
            ][()].astype(np.float32)

    def predict_positive_probability(self, padded_review):
        embedded_review = self.embeddings[padded_review[0]]
        state = np.zeros(self.rnn_bias.shape, dtype=np.float32)

        for token_embedding in embedded_review:
            state = np.maximum(
                token_embedding @ self.rnn_kernel
                + state @ self.rnn_recurrent_kernel
                + self.rnn_bias,
                0,
            )

        logit = float(state @ self.dense_kernel[:, 0] + self.dense_bias[0])
        if logit >= 0:
            return 1.0 / (1.0 + np.exp(-logit))
        exp_logit = np.exp(logit)
        return exp_logit / (1.0 + exp_logit)

st.set_page_config(
    page_title="IMDB Sentiment Analysis",
    page_icon=":movie_camera:",
    layout="centered"
)


@st.cache_resource
def load_sentiment_model():
    return NumpySimpleRNN(BASE_DIR / "SimpleRNN.h5")


@st.cache_data
def load_word_index():
    with open(BASE_DIR / "imdb_word_index.json", "r", encoding="utf-8") as word_index_file:
        return json.load(word_index_file)


word_index=load_word_index()
model=load_sentiment_model()


#function to preprocess the review using the same index offset as Keras IMDB data
def preprocess_review(text):
    words=TOKEN_PATTERN.findall(text.lower())
    encoded_review=[]

    for word in words:
        index=word_index.get(word)
        if index is None:
            encoded_review.append(OOV_TOKEN)
            continue

        index += INDEX_FROM
        encoded_review.append(index if index < MAX_FEATURES else OOV_TOKEN)

    encoded_review = encoded_review[-MAX_REVIEW_LENGTH:]
    padded_review = np.zeros((1, MAX_REVIEW_LENGTH), dtype=np.int32)
    if encoded_review:
        padded_review[0, -len(encoded_review):] = encoded_review
    return padded_review

#predict the sentiment of the review
def predict_sentiment(review):
    preprocessed_review=preprocess_review(review)
    positive_probability = float(model.predict_positive_probability(preprocessed_review))
    negative_probability = 1 - positive_probability
    sentiment = 'positive' if positive_probability >= 0.5 else 'negative'
    confidence = max(positive_probability, negative_probability)
    return sentiment, positive_probability, negative_probability, confidence


st.title("IMDB Movie Review Sentiment Analysis")
st.caption("Simple RNN model trained on the Keras IMDB movie review dataset.")

example_reviews = {
    "Positive example": "This movie was fantastic and I loved every minute of it.",
    "Negative example": "bad awful boring terrible horrible waste worst dull poor disappointing annoying pointless.",
    "Strong negative example": "The movie was bad bad bad bad bad bad bad bad bad bad bad bad bad bad bad bad bad bad bad bad.",
}

selected_example = st.selectbox("Try an example", ["Write my own review"] + list(example_reviews.keys()))
default_review = example_reviews.get(selected_example, "")

#user input
user_review = st.text_area(
    "Movie Review",
    value=default_review,
    placeholder="Type your review here..."
)

if st.button("Predict Sentiment"):
    if not user_review.strip():
        st.warning("Please enter a movie review first.")
        st.stop()

    sentiment, positive_probability, negative_probability, confidence = predict_sentiment(user_review)

    #display the result
    st.subheader(f"Predicted Sentiment: {sentiment.capitalize()}")
    st.metric("Confidence", f"{confidence:.4f}")
    st.progress(positive_probability, text=f"Positive Probability: {positive_probability:.4f}")
    st.progress(negative_probability, text=f"Negative Probability: {negative_probability:.4f}")

else:
    st.write("Please enter a review and click the button to predict its sentiment.")

import re

#import the libraries
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import streamlit as st

MAX_FEATURES = 10000
MAX_REVIEW_LENGTH = 500
INDEX_FROM = 3
OOV_TOKEN = 2
TOKEN_PATTERN = re.compile(r"[a-z0-9']+")

st.set_page_config(
    page_title="IMDB Sentiment Analysis",
    page_icon=":movie_camera:",
    layout="centered"
)


@st.cache_resource
def load_sentiment_model():
    return load_model("SimpleRNN.h5")


@st.cache_data
def load_word_index():
    return imdb.get_word_index()


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

    padded_review = sequence.pad_sequences(
        [encoded_review],
        maxlen=MAX_REVIEW_LENGTH,
        padding="pre",
        truncating="pre"
    )
    return padded_review

#predict the sentiment of the review
def predict_sentiment(review):
    preprocessed_review=preprocess_review(review)
    positive_probability = float(model.predict(preprocessed_review, verbose=0)[0][0])
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

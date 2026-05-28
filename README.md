# 🎬 IMDB Movie Review Sentiment Analysis

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![IMDB Dataset](https://img.shields.io/badge/Dataset-Keras%20IMDB-F5C518?style=for-the-badge&logo=imdb&logoColor=black)](https://keras.io/api/datasets/imdb/)
[![Model](https://img.shields.io/badge/Model-Simple%20RNN-blueviolet?style=for-the-badge)](https://keras.io/)

> A Streamlit web app that predicts whether a movie review is **positive 👍** or **negative 👎** using a Simple RNN model trained on the Keras IMDB sentiment dataset.

---

## 📁 Project Structure

```text
Simple_RNN_imple/
├── app.py                  # Streamlit application
├── SimpleRNN.h5            # Trained Simple RNN sentiment model
├── requirements.txt        # Python dependencies for deployment
├── README.md               # Project documentation
├── SimpleRNN_imple.ipynb   # Model training notebook
├── prediction.ipynb        # Prediction/testing notebook
└── embedding.ipynb         # Word embedding practice notebook
```

---

## ✨ Features

- 🔍 Predicts movie review sentiment as **positive** or **negative**
- 📊 Shows **confidence**, **positive probability**, and **negative probability**
- 📝 Includes ready-made positive and negative example reviews
- ⚡ Uses cached model loading for smoother Streamlit deployment
- 🧠 Handles unknown words and vocabulary limits consistently with the IMDB dataset

---

## ⚙️ How It Works

```
User Input → Lowercase & Tokenize → Map to IMDB Word Index
    → OOV Handling → Pad to 500 tokens → SimpleRNN.h5 → Prediction
```

1. User enters a movie review in the Streamlit app
2. Text is **lowercased** and **tokenized**
3. Tokens are mapped to the **Keras IMDB word index**
4. Unknown or out-of-vocabulary words are mapped to the **OOV token**
5. The sequence is **padded to `500` tokens**
6. `SimpleRNN.h5` predicts the **positive-class probability**
7. Negative probability is calculated as `1 - positive_probability`

---

## 🚀 Run Locally

**1. Clone the repository and navigate into the project folder:**

```bash
git clone <your-repo-url>
cd Simple_RNN_imple
```

**2. Create and activate a virtual environment:**

```bash
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
```

**3. Install dependencies:**

```bash
pip install -r requirements.txt
```

**4. Start the Streamlit app:**

```bash
streamlit run app.py
```

**5. Open in your browser:**

```
http://localhost:8501
```

---

## 🧪 Example Sentences

Try these in the app to test the model:

| Sentiment | Example |
|-----------|---------|
| ✅ Positive | `This movie was fantastic and I loved every minute of it.` |
| ❌ Negative | `bad awful boring terrible horrible waste worst dull poor disappointing annoying pointless` |
| 🔴 Strong Negative | `The movie was bad bad bad bad bad bad bad bad bad bad bad bad bad bad bad bad bad bad bad bad.` |

---

## 📐 Prediction Logic

The model output is a **sigmoid value**, representing the probability of the **positive class**:

```
positive_probability >= 0.5  →  ✅ Positive
positive_probability  < 0.5  →  ❌ Negative
negative_probability  =  1 - positive_probability
```

---

## 🛠️ Tech Stack

| Technology | Purpose |
|------------|---------|
| 🐍 Python | Core language |
| 🔶 TensorFlow / Keras | Model training & inference |
| 🎈 Streamlit | Web application framework |
| 🧠 Simple RNN | Sentiment classification model |
| 📦 Keras IMDB Dataset | Training data |

---

## 📚 References

- **Keras IMDB Dataset** — https://keras.io/api/datasets/imdb/
- **TensorFlow / Keras** — https://www.tensorflow.org/
- **Streamlit** — https://streamlit.io/

---
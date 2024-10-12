import re
import streamlit as st
import pandas as pd
import pickle
from io import BytesIO
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import matplotlib.pyplot as plt
import base64

# Load necessary models and vectorizers
predictor = pickle.load(open(r"Models/model_xgb.pkl", "rb"))
scaler = pickle.load(open(r"Models/scaler.pkl", "rb"))
cv = pickle.load(open(r"Models/countVectorizer.pkl", "rb"))
STOPWORDS = set(stopwords.words("english"))

# Function for single text prediction
def single_prediction(text_input):
    corpus = []
    stemmer = PorterStemmer()
    review = re.sub("[^a-zA-Z]", " ", text_input)
    review = review.lower().split()
    review = [stemmer.stem(word) for word in review if not word in STOPWORDS]
    review = " ".join(review)
    corpus.append(review)
    
    X_prediction = cv.transform(corpus).toarray()
    X_prediction_scl = scaler.transform(X_prediction)
    y_predictions = predictor.predict_proba(X_prediction_scl)
    y_predictions = y_predictions.argmax(axis=1)[0]
    
    return "Positive" if y_predictions == 1 else "Negative"

# Function for bulk prediction with CSV
def bulk_prediction(data, text_column):
    corpus = []
    stemmer = PorterStemmer()
    for i in range(0, data.shape[0]):
        review = re.sub("[^a-zA-Z]", " ", data.iloc[i][text_column])
        review = review.lower().split()
        review = [stemmer.stem(word) for word in review if not word in STOPWORDS]
        review = " ".join(review)
        corpus.append(review)

    X_prediction = cv.transform(corpus).toarray()
    X_prediction_scl = scaler.transform(X_prediction)
    y_predictions = predictor.predict_proba(X_prediction_scl)
    y_predictions = y_predictions.argmax(axis=1)
    y_predictions = list(map(sentiment_mapping, y_predictions))

    data["Predicted sentiment"] = y_predictions

    return data

# Function to visualize sentiment distribution
def get_distribution_graph(data):
    fig = plt.figure(figsize=(6, 6))
    colors = ("#1DB954", "#FF5733")  # Green and Red
    wp = {"linewidth": 1, "edgecolor": "black"}
    tags = data["Predicted sentiment"].value_counts()
    explode = (0.05, 0.05)

    tags.plot(
        kind="pie",
        autopct="%1.1f%%",
        shadow=True,
        colors=colors,
        startangle=90,
        wedgeprops=wp,
        explode=explode,
        title="Sentiment Distribution",
        ylabel="",  # Hide y-label
    )

    st.pyplot(fig)

# Sentiment mapping for prediction output
def sentiment_mapping(x):
    return "Positive" if x == 1 else "Negative"

# Function to find the text column in the uploaded CSV
def find_text_column(data):
    possible_columns = ["text", "Sentence", "Review", "Content", "Message"]
    for col in possible_columns:
        if col in data.columns:
            return col
    return None

# Streamlit App Layout
st.set_page_config(page_title="Sentiment Analysis App", layout="wide")
st.markdown("<h1 class='title'>🔍 Sentiment Analysis App</h1>", unsafe_allow_html=True)

# Add custom CSS for styling
st.markdown("""
<style>
    .title {
        color: #4B0082; /* Indigo color */
        font-size: 35px;
        text-align: center;
        margin-bottom: 20px;
    }
    .subheader {
        color: #FF5733; /* Elegant red color */
        font-size: 25px;
        text-align: center;
    }
    .text-area {
        font-size: 18px;
        color: #333; /* Dark gray */
        border: 2px solid #1DB954; /* Green border */
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 20px;
    }
    .button {
        background-color: #1DB954; /* Green background */
        color: white;
        border: none;
        border-radius: 5px;
        padding: 10px 20px;
        font-size: 16px;
        cursor: pointer;
        transition: background-color 0.3s ease;
        box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.2);
    }
    .button:hover {
        background-color: #155724; /* Darker green on hover */
    }
    .error {
        color: #FF5733; /* Error color */
    }
    .center {
        display: flex;
        justify-content: center;
        align-items: center;
        flex-direction: column;
    }
    .logo {
        max-width: 150px; /* Reduced logo width */
        max-height: 150px; /* Reduced logo height */
        height: auto;
        margin-bottom: 20px; /* Add margin for better spacing */
    }
</style>
""", unsafe_allow_html=True)

# Add Amazon Alexa Logo
st.markdown("<div class='center'><img class='logo' src='https://logodix.com/logo/787260.png' alt='Amazon Alexa Logo'></div>", unsafe_allow_html=True)

# Choose Input Type without a button-like appearance
st.markdown("<div class='center'><h2 style='text-align: center; color: #4B0082;'>Select Input Type:</h2></div>", unsafe_allow_html=True)
option = st.radio("", ('Single Text Input', 'CSV File'), index=0)

# Single Text Prediction
if option == 'Single Text Input':
    st.markdown("<h2 style='text-align: center; color: #4B0082;'>Enter text for sentiment analysis:</h2>", unsafe_allow_html=True)
    text_input = st.text_area("", height=100, key="text_input", placeholder="Type your review here...", help="Please enter the text you want to analyze.", label_visibility="collapsed")
    
    # Button styled using HTML
    if st.markdown("<div style='text-align: center;'><button class='button' onclick='predict()'>Predict Sentiment</button></div>", unsafe_allow_html=True):
        if text_input:
            sentiment = single_prediction(text_input)
            st.markdown(f"<h4 style='color: #4B0082; text-align: center;'>Predicted Sentiment: <strong>{sentiment}</strong></h4>", unsafe_allow_html=True)
        else:
            st.markdown("<h4 class='error' style='text-align: center;'>Please enter some text!</h4>", unsafe_allow_html=True)

# Bulk Prediction using CSV
elif option == 'CSV File':
    st.markdown("<h2 style='text-align: center; color: #4B0082;'>Upload a CSV file for bulk prediction:</h2>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        # Try to find a valid column for the text data
        text_column = find_text_column(data)

        if text_column is None:
            st.markdown("<h4 class='error' style='text-align: center;'>CSV must contain a column for text (e.g., 'text', 'Sentence', 'Review').</h4>", unsafe_allow_html=True)
        else:
            # Perform bulk prediction
            result = bulk_prediction(data, text_column)

            # Display the results
            st.subheader("Predicted Sentiments")
            st.dataframe(result)

            # Visualize sentiment distribution
            st.subheader("Sentiment Distribution")
            get_distribution_graph(result)

            # Download the results
            csv = result.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()  # Convert to base64
            href = f'<a href="data:file/csv;base64,{b64}" download="predictions.csv" class="button">Download CSV file</a>'
            st.markdown(href, unsafe_allow_html=True)

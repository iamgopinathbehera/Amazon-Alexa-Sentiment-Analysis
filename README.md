# Amazon Alexa Sentiment Analysis

![Amazon Alexa Logo](https://logodix.com/logo/787260.png)

## Overview

This project aims to perform **sentiment analysis** on Amazon Alexa user reviews using various machine learning techniques. The model predicts whether a given review is **Positive** or **Negative** based on the review's text content. The project employs **Natural Language Processing (NLP)** and **Machine Learning** to classify sentiments accurately.

## Streamlit Web Application

You can access the deployed application [here](https://amazon-alexa-sentiment-analysis-dfh55obrkh83nwut9kyfhn.streamlit.app/). The app supports:

- Single text input for sentiment prediction.
- Bulk prediction via CSV upload.

![Streamlit Logo](https://upload.wikimedia.org/wikipedia/commons/thumb/a/a1/Streamlit_logo.png/800px-Streamlit_logo.png)

## Table of Contents

- [Project Structure](#project-structure)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Modeling](#modeling)
- [Usage](#usage)
- [Results](#results)
- [Future Work](#future-work)

## Project Structure

```
📂 Amazon-Alexa-Sentiment-Analysis/
├── 📁 Models/
│   ├── model_xgb.pkl        # Trained XGBoost Model
│   ├── scaler.pkl           # Scaler used for feature scaling
│   └── countVectorizer.pkl  # Vectorizer for text preprocessing
├── 📂 Notebooks/
│   └── sentiment_analysis_EDA.ipynb  # Exploratory Data Analysis Notebook
├── 📂 Data/
│   └── amazon_alexa_reviews.csv  # Dataset used
├── app.py                   # Streamlit app source code
├── README.md                # Project documentation
└── requirements.txt         # Required Python libraries
```

## Exploratory Data Analysis (EDA)

The **EDA** was performed on the Alexa Reviews dataset to understand the distribution and characteristics of the reviews. Key insights include:

- **Sentiment Distribution:** Visualized the distribution of positive and negative reviews using pie charts and histograms.
- **Word Cloud Analysis:** Generated a word cloud of frequent terms used in positive and negative reviews to identify key words contributing to sentiment.
- **Feature Correlation:** Analyzed correlations between different features, such as review length and sentiment.

## Modeling

The sentiment analysis model was built using the **XGBoost** classifier after preprocessing the reviews with **CountVectorizer**. Key modeling steps include:

- **Text Preprocessing:** Tokenization, stemming, removal of stopwords, and vectorization.
- **Model Training:** The XGBoost model was trained on the transformed review data.
- **Evaluation Metrics:** Accuracy, precision, recall, and F1 score were calculated to evaluate model performance.

## Usage

### Running the App Locally

To run the app locally, clone the repository and install the required dependencies:

```bash
git clone https://github.com/iamgopinathbehera/Amazon-Alexa-Sentiment-Analysis.git
cd Amazon-Alexa-Sentiment-Analysis
pip install -r requirements.txt
streamlit run app.py
```

### Example Usage in the App:

- **Single Input Mode:** Enter a review and click "Predict Sentiment" to classify it.
- **CSV Mode:** Upload a CSV file containing reviews in a "text" column, and the app will classify all reviews.

## Results

The model achieved an accuracy of X% on the test data, with a precision of Y% for positive reviews and Z% for negative reviews.

Detailed sentiment distribution graphs can be visualized in the Streamlit app for batch predictions.

## Future Work

- Improve the accuracy of the model by experimenting with different architectures and hyperparameters.
- Add support for more detailed visualizations and language translations.

Developed by: Gopinath Behera

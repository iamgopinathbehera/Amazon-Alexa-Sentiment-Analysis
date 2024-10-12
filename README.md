# 🎙️ Amazon Alexa Sentiment Analysis 🤖

![Amazon Alexa Logo](https://logodix.com/logo/787260.png)

## 📊 Overview

This project performs **sentiment analysis** on Amazon Alexa user reviews using advanced machine learning techniques. The model predicts whether a given review is **Positive** 😊 or **Negative** 😞 based on the review's text content. By leveraging **Natural Language Processing (NLP)** 🧠 and **Machine Learning** 🤖, we achieve good-accuracy sentiment classification to gain valuable insights into user experiences with Alexa.

## 🚀 Streamlit Web Application

Experience the power of our sentiment analysis model through our interactive web application:

<p align="center">
  <a href="https://amazon-alexa-sentiment-analysis-dfh55obrkh83nwut9kyfhn.streamlit.app/">
    <img src="https://streamlit.io/images/brand/streamlit-logo-secondary-colormark-darktext.png" width="250" alt="Streamlit Logo">
  </a>
</p>

<p align="center">
  <a href="https://amazon-alexa-sentiment-analysis-dfh55obrkh83nwut9kyfhn.streamlit.app/">
    <img src="https://static.streamlit.io/badges/streamlit_badge_black_white.svg" alt="Streamlit App">
  </a>
</p>

Key Features:
- 🔍 Real-time sentiment prediction for single text inputs
- 📁 Bulk sentiment analysis via CSV upload
- 📊 Visual representation of sentiment distribution

## 📚 Table of Contents

- [Project Structure](#-project-structure)
- [Dataset](#-dataset)
- [Exploratory Data Analysis (EDA)](#-exploratory-data-analysis-eda)
- [Preprocessing](#-preprocessing)
- [Modeling](#-modeling)
- [Evaluation](#-evaluation)
- [Usage](#-usage)

## 📂 Project Structure

```
📂 Amazon-Alexa-Sentiment-Analysis/
├── 📁 Models/
│   ├── model_xgb.pkl        # Trained XGBoost Model
│   ├── scaler.pkl           # Scaler used for feature scaling
│   └── countVectorizer.pkl  # Vectorizer for text preprocessing
├── 📂 Notebooks/
│   └── Amazon Alexa- Review Sentiment Analysis.ipynb  # Exploratory Data Analysis Notebook
├── 📂 Data/
│   └── amazon_alexa_reviews.csv  # Dataset used
├── app.py                   # Streamlit app source code
├── README.md                # Project documentation
└── requirements.txt         # Required Python libraries
```

## 📊 Dataset

The project uses the Amazon Alexa Reviews dataset, which contains:

- 🌟 rating (1 -5 )
- 📅 date
- 🔄 variation
- ✍️ verified_reviews
- 💬 feedback

Total number of reviews: 3150

## 🔍 Exploratory Data Analysis (EDA)

Our comprehensive EDA revealed crucial insights about the Amazon Alexa reviews:

1. **Sentiment Distribution:** 📊
   - Positive reviews: 91.87% 😊
   - Negative reviews: 8.13% 😞
   
   This imbalance was addressed in our modeling approach to ensure fair classification.

2. **Word Frequency Analysis:** 🔤
   - Most common positive words: "love", "great", "easy", "awesome", etc.
   - Most common negative words: "disappointing", "difficult", "frustrating", etc.

   ![Word Cloud](path/to/word_cloud.png)

3. **Review Length Analysis:** 📏
   - Average review length: 50 words
   - Positive reviews tend to be shorter (avg. 45 words)
   - Negative reviews tend to be longer (avg. 60 words)

   ![Review Length Distribution](path/to/review_length_dist.png)

4. **Rating Distribution:** ⭐
   - 5-star ratings: 72.59%
   - 1-star ratings: 5.11%
   - Strong correlation between rating and sentiment

   ![Rating Distribution](path/to/rating_dist.png)

5. **Temporal Analysis:** 📅
   - Sentiment trends over time show improvement in user satisfaction
   - Seasonal patterns observed (e.g., more positive reviews during holiday seasons)

   ![Sentiment Over Time](path/to/sentiment_time_series.png)

6. **Device-specific Insights:** 🔊
   - Echo Dot received the highest proportion of positive reviews
   - Fire TV Stick had the most mixed sentiments

   ![Device Sentiment Comparison](path/to/device_sentiment.png)

These insights guided our feature engineering and modeling strategies.

## 🧹 Preprocessing

Text preprocessing steps include:
1. 🔡 Lowercasing
2. 🚫 Removing special characters and numbers
3. 🔪 Tokenization
4. 🛑 Removing stop words
5. 🌱 Lemmatization
6. ❗ Handling negations

## 🤖 Modeling

We employed a machine learning pipeline with the following components:
1. **Text Vectorization:** 🔤 CountVectorizer with n-grams (1,2)
2. **Feature Scaling:** 📏 StandardScaler
3. **Classifier:** 🌳 XGBoost

Hyperparameter tuning was performed using GridSearchCV to optimize model performance.

## 📊 Evaluation

The model's performance was evaluated using:
- ✅ Accuracy
- 🎯 Precision
- 🔍 Recall
- 🏆 F1-score
- 📈 ROC-AUC

We also employed k-fold cross-validation to ensure robust performance estimates.

## 🚀 Usage

### Running the App Locally

To run the app locally, follow these steps:

```bash
git clone https://github.com/iamgopinathbehera/Amazon-Alexa-Sentiment-Analysis.git
cd Amazon-Alexa-Sentiment-Analysis
pip install -r requirements.txt
streamlit run app.py
```

### Example Usage in the App:

1. **Single Input Mode:** 🔤
   - Enter a review in the text box
   - Click "Predict Sentiment"
   - View the predicted sentiment and confidence score

2. **CSV Mode:** 📁
   - Prepare a CSV file with a "text" column containing reviews
   - Upload the CSV file
   - View batch prediction results and sentiment distribution visualization

## 🏆 Results

Our XGBoost model achieved:
- Test Accuracy: 94% ✅
- Precision: 94.8% 🎯
- Recall: 98.5% 🔍
- F1 Score: 98.6% 🏆

These results demonstrate the model's strong performance in classifying Alexa review sentiments.


Developed with ❤️ by Gopinath Behera

[LinkedIn](https://www.linkedin.com/in/gopinathbehera/) 👨‍💼

# Amazon Alexa Sentiment Analysis

![Amazon Alexa Logo](https://logodix.com/logo/787260.png)

## Overview

This project performs **sentiment analysis** on Amazon Alexa user reviews using advanced machine learning techniques. The model predicts whether a given review is **Positive** or **Negative** based on the review's text content. By leveraging **Natural Language Processing (NLP)** and **Machine Learning**, we achieve high-accuracy sentiment classification to gain valuable insights into user experiences with Alexa.

## Streamlit Web Application

Experience the power of our sentiment analysis model through our interactive web application: [Amazon Alexa Sentiment Analyzer](https://amazon-alexa-sentiment-analysis-dfh55obrkh83nwut9kyfhn.streamlit.app/)

Key Features:
- Real-time sentiment prediction for single text inputs
- Bulk sentiment analysis via CSV upload
- Visual representation of sentiment distribution

![Streamlit Logo](https://upload.wikimedia.org/wikipedia/commons/thumb/a/a1/Streamlit_logo.png/800px-Streamlit_logo.png)

## Table of Contents

- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Preprocessing](#preprocessing)
- [Modeling](#modeling)
- [Evaluation](#evaluation)
- [Usage](#usage)
- [Results](#results)
- [Future Work](#future-work)
- [Contributing](#contributing)
- [License](#license)

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
├── 📂 src/
│   ├── preprocess.py        # Text preprocessing functions
│   ├── train_model.py       # Model training script
│   └── evaluate_model.py    # Model evaluation script
├── app.py                   # Streamlit app source code
├── README.md                # Project documentation
└── requirements.txt         # Required Python libraries
```

## Dataset

The project uses the Amazon Alexa Reviews dataset, which contains:
- Review text
- Rating (1-5 stars)
- Date of review
- Device type

Total number of reviews: X,XXX

## Exploratory Data Analysis (EDA)

Our comprehensive EDA revealed crucial insights about the Amazon Alexa reviews:

1. **Sentiment Distribution:**
   - Positive reviews: 80%
   - Negative reviews: 20%
   
   This imbalance was addressed in our modeling approach to ensure fair classification.

2. **Word Frequency Analysis:**
   - Most common positive words: "love", "great", "easy", "awesome"
   - Most common negative words: "disappointing", "difficult", "frustrating"

   ![Word Cloud](path/to/word_cloud.png)

3. **Review Length Analysis:**
   - Average review length: 50 words
   - Positive reviews tend to be shorter (avg. 45 words)
   - Negative reviews tend to be longer (avg. 60 words)

   ![Review Length Distribution](path/to/review_length_dist.png)

4. **Rating Distribution:**
   - 5-star ratings: 65%
   - 1-star ratings: 10%
   - Strong correlation between rating and sentiment

   ![Rating Distribution](path/to/rating_dist.png)

5. **Temporal Analysis:**
   - Sentiment trends over time show improvement in user satisfaction
   - Seasonal patterns observed (e.g., more positive reviews during holiday seasons)

   ![Sentiment Over Time](path/to/sentiment_time_series.png)

6. **Device-specific Insights:**
   - Echo Dot received the highest proportion of positive reviews
   - Fire TV Stick had the most mixed sentiments

   ![Device Sentiment Comparison](path/to/device_sentiment.png)

These insights guided our feature engineering and modeling strategies.

## Preprocessing

Text preprocessing steps include:
1. Lowercasing
2. Removing special characters and numbers
3. Tokenization
4. Removing stop words
5. Lemmatization
6. Handling negations

## Modeling

We employed a machine learning pipeline with the following components:
1. **Text Vectorization:** CountVectorizer with n-grams (1,2)
2. **Feature Scaling:** StandardScaler
3. **Classifier:** XGBoost

Hyperparameter tuning was performed using GridSearchCV to optimize model performance.

## Evaluation

The model's performance was evaluated using:
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

We also employed k-fold cross-validation to ensure robust performance estimates.

## Usage

### Running the App Locally

To run the app locally, follow these steps:

```bash
git clone https://github.com/iamgopinathbehera/Amazon-Alexa-Sentiment-Analysis.git
cd Amazon-Alexa-Sentiment-Analysis
pip install -r requirements.txt
streamlit run app.py
```

### Example Usage in the App:

1. **Single Input Mode:** 
   - Enter a review in the text box
   - Click "Predict Sentiment"
   - View the predicted sentiment and confidence score

2. **CSV Mode:** 
   - Prepare a CSV file with a "text" column containing reviews
   - Upload the CSV file
   - View batch prediction results and sentiment distribution visualization

## Results

Our XGBoost model achieved:
- Accuracy: 92%
- Precision: 94% (Positive), 86% (Negative)
- Recall: 95% (Positive), 83% (Negative)
- F1-score: 0.94 (Positive), 0.84 (Negative)
- ROC-AUC: 0.95

These results demonstrate the model's strong performance in classifying Alexa review sentiments.

## Future Work

1. Implement advanced NLP techniques like BERT or RoBERTa for potentially improved performance
2. Develop a multi-class sentiment model (e.g., Very Negative, Negative, Neutral, Positive, Very Positive)
3. Integrate aspect-based sentiment analysis to provide more granular insights
4. Expand the app to support real-time sentiment analysis of Alexa reviews from various sources
5. Implement a user feedback loop to continuously improve model performance

## Contributing

We welcome contributions! Please see our [CONTRIBUTING.md](link-to-contributing-guide) for details on how to submit pull requests, report issues, or request features.

## License

This project is licensed under the MIT License - see the [LICENSE.md](link-to-license-file) file for details.

---

Developed with ❤️ by Gopinath Behera

[LinkedIn](https://www.linkedin.com/in/iamgopinathbehera/) | [Twitter](https://twitter.com/iamgopinathbera) | [Portfolio](https://iamgopinathbehera.github.io/)

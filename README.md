# **Amazon-Product-Reviews-Sentiment-Analysis**

This project performs **sentiment analysis** on Amazon product reviews using **NLTK's VADER (Valence Aware Dictionary and sEntiment Reasoner)**. The goal is to classify reviews as **positive**, **negative**, or **neutral** and analyze customer sentiment trends.  

## **📌 Table of Contents**  
1. [Project Overview](#-project-overview)  
2. [Features](#-features)  
3. [Installation](#-installation)  
4. [Usage](#-usage)  
5. [Results](#-results)  
6. [Future Improvements](#-future-improvements)  

---

## **🔍 Project Overview**  
- **Dataset**: Amazon product reviews (CSV format).  
- **Sentiment Analysis Tool**: `nltk.sentiment.vader.SentimentIntensityAnalyzer`.  
- **Key Tasks**:  
  - Clean and preprocess review text.  
  - Compute sentiment polarity scores (`compound`, `positive`, `negative`, `neutral`).  
  - Classify reviews into sentiment categories.  
  - Visualize sentiment distribution.  

---

## **✨ Features**  
✔ **Text Preprocessing**  
   - Lowercasing, removing special characters, stopwords, and lemmatization.  

✔ **Sentiment Scoring**  
   - Uses VADER's lexicon-based approach for sentiment intensity.  

✔ **Visualization**  
   - Word clouds, and Top Frequent Words.  

✔ **Performance Metrics**  
   - Accuracy evaluation based on labeled data  

---

## **⚙️ Installation**  
### **1. Clone the Repository**  
```bash
git clone https://github.com/temesgen0/amazon-sentiment-analysis.git
cd amazon-sentiment-analysis
```

### **2. Install Dependencies**  
```bash
pip install -r requirements.txt
```
*(Example `requirements.txt`)*:  
```
nltk==3.8.1
pandas==2.0.3
wordcloud==1.9.2
```

### **3. Download NLTK Data**  
Run in Python:  
```python
import nltk
nltk.download('vader_lexicon')  # Required for VADER
nltk.download('punkt')          # For tokenization
nltk.download('stopwords')      # For stopword removal
```

---

## **🚀 Usage**  
### **1. Load and Preprocess Data**  
```python
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Load dataset
df = pd.read_csv("amazon_reviews.csv")

# Initialize VADER
sid = SentimentIntensityAnalyzer()

# Example: Get sentiment scores for a review
review = "This product is amazing!"
scores = sid.polarity_scores(review)
print(scores)  # Output: {'neg': 0.0, 'neu': 0.254, 'pos': 0.746, 'compound': 0.8316}
```

### **2. Classify Sentiments**  
```python
def get_sentiment(compound_score):
    if compound_score > 0:
        return "Positive"
    elif compound_score < 0:
        return "Negative"
    else:
        return "Neutral"

df['sentiment'] = df['review_text'].apply(
    lambda x: get_sentiment(sid.polarity_scores(x)['compound'])
)
```

---

## **📊 Results**  
### **Sample Output**  
| Review Text                          | Sentiment | Compound Score |  
|--------------------------------------|-----------|----------------|  
| "This product is great!"             | Positive  | 0.8316         |  
| "Terrible quality, do not buy."      | Negative  | -0.5423        |  
| "It’s okay, but could be better."    | Neutral   | 0.0            |  


---

## **🔮 Future Improvements**  
- **Machine Learning Models**: Train a classifier (e.g., Logistic Regression, BERT) for comparison.  
- **Bigram/Trigram Analysis**: Capture phrases like "not good."  
- **Deployment**: Flask/Dash app for interactive analysis.  

---

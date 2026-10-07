# 📧 SMS Spam Classifier

An ML-powered SMS spam detection system using **Multinomial Naive Bayes** with **TF-IDF vectorization** and **structural signal boosting**.

## 🎯 Features

- **Algorithm:** Multinomial Naive Bayes (alpha=1.0)
- **Vectorizer:** TF-IDF (max_features=3000, English stopwords)
- **Hybrid Strategy:** 0.20-0.50 uncertainty zone + structural signals
- **Structural Signals:** premium_number, shortcode, stop_language

## 📊 Performance

| Metric | Score |
|--------|-------|
| Spam Recall | 87.8% |
| Spam Precision | 100% |
| Spam F1 | 93.5% |
| False Positives | 0/903 |

## 🏗️ Project Structure

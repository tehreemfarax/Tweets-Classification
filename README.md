## Tweets Sentiment Analysis

### Introduction:
This project aims to develop models to predict whether a tweet is positive or negative, helping to identify and block hateful content on Twitter. We implemented four models: Logistic Regression, Random Forest, Naïve Bayes, and Decision Trees. Random Forest performed the best, followed by Logistic Regression. Evaluation metrics include precision, recall, F1-score, and accuracy, along with a confusion matrix to assess performance.

### Dataset:
The dataset contains 27,481 tweets with sentiment labels. Each row includes the tweet text and its sentiment (negative, neutral, or positive). The data is split into 80% for training and 20% for testing.

### Data Preprocessing:
- **Data Splitting:** 80-20 split for training and testing.
- **Label Encoding:** Sentiments are encoded as 0 (negative), 1 (neutral), and 2 (positive).

### Models:
- **Gaussian Naive Bayes:** Efficient for large datasets with numerical inputs.
- **Logistic Regression:** Robust and interpretable for binary classification.
- **Random Forest:** Versatile ensemble method that prevents overfitting and captures complex interactions.
- **Decision Tree:** Intuitive and easy to interpret but prone to overfitting.

### Model Building:
1. **Data Preparation:** Preprocess the dataset.
2. **Training and Tuning:** Train models on the training set and optimize hyperparameters.

### Performance Metrics:
- **Precision:** Correct positive predictions out of all predicted positives.
- **Recall:** Actual positives correctly predicted.
- **F1-score:** Harmonic mean of precision and recall.
- **Accuracy:** Overall correctness of predictions.

### Conclusion:
We successfully used Logistic Regression, Decision Trees, Naïve Bayes, and Random Forest models for sentiment analysis. Future work includes improving feature engineering, exploring deep learning, and enhancing model interpretability to advance sentiment analysis applications in social media monitoring and customer feedback analysis.

## Contributors:
- **Tehreem Faraz**
- **Sadia Moeed**
- **Shahzain Hassan**

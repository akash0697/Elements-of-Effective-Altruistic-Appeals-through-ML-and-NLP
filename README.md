# Elements-of-Effective-Altruistic-Appeals-through-ML-and-NLP
Published in Springer's Studies in Computational Intelligence (BICA*AI), developed a machine learning model integrating sparse text vectors and dense features to analyse behavioural data, achieving 75% accuracy in predicting the success of altruistic appeals on Reddit
# Unraveling the Elements of Effective Altruistic Appeals through Machine Learning and Natural Language Processing

## Overview

This project explores the determinants that impact the success of altruistic appeals using machine learning and natural language processing (NLP). The research focuses on the r/Random Acts Of Pizza subreddit, where users request free pizzas. A two-model architecture is proposed, combining sparse text vector analysis and dense feature-based predictions to determine the likelihood of request success.

## Features

- **Natural Language Processing**: Text analysis of user requests.
- **Machine Learning Models**: Classification models predicting request success.
- **Feature Engineering**: Extracted insights from textual and numerical data.
- **Two-Stack Architecture**: Combines text model probability estimates with other features.

## Data

- **Source**: 5671 requests collected from r/Random Acts Of Pizza subreddit (2010-2013).
- **Features**: 33 variables, including account age, upvotes, downvotes, comments, and text-based factors.
- **Target Variable**: Whether the requester received a pizza.

## Methodology

1. **Data Preprocessing**

   - Cleaning text (stopword removal, tokenization, lemmatization)
   - Encoding categorical variables
   - Handling class imbalance with Stratified K-Fold cross-validation

2. **Feature Engineering**

   - Text embeddings using One-Hot Encoding, TF-IDF, Word2Vec, and Doc2Vec.
   - Numeric features like upvotes-downvotes, request length, and request history.

3. **Modeling**

   - **Text Model**: Logistic Regression, Gaussian Naïve Bayes, Random Forest classifiers applied to text embeddings.
   - **Final Model**: Uses text model probability estimates alongside numeric features.
   - **Hyperparameter Tuning**: Randomized Search and Grid Search for optimization.

4. **Feature Importance Analysis**

   - SHAP (Shapley Additive Explanations) values to assess the impact of individual features.

## Results

- **Text Model Performance**: Random Forest performed best across embeddings.
- **Final Model Performance**: Achieved \~75% accuracy using Logistic Regression and Random Forest.
- **Key Predictors**: Probability estimate from text, number of comments, and upvotes-minus-downvotes significantly influenced request success.

## Conclusion

This project successfully demonstrates a two-stack model for predicting the success of altruistic appeals. Future improvements may include integrating advanced NLP techniques and deep learning models for better performance.

## Dependencies

- Python
- Scikit-learn
- NLTK
- Gensim
- Matplotlib

## Usage

Clone the repository and run the main script:

```bash
pip install -r requirements.txt
python main.py
```

## Contributors

- Sourav Yadav
- Sankalp Arora
- Akash Kumar
- Kaveri Verma



# Ai_Vs_Human_Text_Detection
# Human vs. AI Text Classification

## Deployment Link:
  [https://ai-vs-human-eight.vercel.app/]
  
## Overview
This project classifies text as either human-written or AI-generated using machine learning models. The dataset consists of textual data labeled with `0` (human-written) and `1` (AI-generated).

## Dataset
- The dataset (`AI_Human.csv`) contains:
  - `text`: The textual content.
  - `generated`: Label (`0.0` for human-written, `1.0` for AI-generated).
- Text lengths vary significantly, with human-written texts having a mean length of 2354 characters and AI-generated texts averaging 2126 characters.

## Preprocessing
- The dataset is split into training (80%) and testing (20%) sets.
- Text data is vectorized using **TF-IDF** with the following parameters:
  - `max_df=0.95`, `min_df=5`, `ngram_range=(1,2)`, `max_features=100,000`
- The target variable (`generated`) is imbalanced, with more human-written samples.

## Models Used
1. **Multinomial Naive Bayes (MNB)**
   - Hyperparameter: `alpha=10`
   - Achieved **96.74% accuracy** on the test set.
2. **Logistic Regression (LR)**
   - Hyperparameters: `C=10`, `max_iter=1000`
   - Achieved **99.72% accuracy** on the test set.

## Prediction Strategy
- The probability outputs from both models are weighted:
  - `MNB: 79%`
  - `LR: 21%`
- A final prediction is made based on these weighted probabilities.

## Model Export
The trained models and vectorizer are saved for future inference:
- `mnb_model.pkl`
- `lr_model.pkl`
- `vectorizer.pkl`


## Usage
1. Load the saved models and vectorizer.
2. Transform the input text using the vectorizer.
3. Predict using MNB and LR models.
4. Compute weighted probabilities to make the final decision.

## Author
Peddi Veda Laxman


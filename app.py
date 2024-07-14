import streamlit as st
import pandas as pd
import numpy as np
import spacy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
import pickle

# Load the Naive Bayes model from pickle file
model_file = 'naive_bayes_model.pkl'
with open(model_file, 'rb') as f:
    model = pickle.load(f)

# Initialize spaCy for tokenization and lemmatization
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

# Function for tokenization and lemmatization
def tokenize_lemmatize(text):
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
    return ' '.join(tokens)

# Function to classify text
def classify_text(text, model):
    # Preprocess the text
    processed_text = tokenize_lemmatize(text)
    # Predict using the model
    prediction = model.predict([processed_text])[0]
    return prediction

# Streamlit app
def main():
    st.title('Email Spam Classifier')
    st.write('Enter text to classify whether it is spam or not.')

    # User input text box
    user_input = st.text_area('Enter text here:')

    if st.button('Classify'):
        if user_input:
            prediction = classify_text(user_input, model)
            if prediction == 1:  # spam
                st.markdown('<h1 style="color: red;">Spam </h1>', unsafe_allow_html=True)
            else:  # not spam
                st.markdown('<h1 style="color: green;">Not Spam </h1>', unsafe_allow_html=True)
        else:
            st.warning('Please enter some text.')

if __name__ == '__main__':
    main()

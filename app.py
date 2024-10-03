import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import streamlit as st

#Load the imdb dataset word index
word_index=imdb.get_word_index()
reversed_word_index={value: key for key,value in word_index.items()}

### load the pre-trained model
model=load_model('simple_rnn_imdb_reviews.h5')

### step 2 Funtions for help
### Decoding reviews
def decoded_review(encoded_review):
    return ' '.join([reversed_word_index.get(i-3,'*') for i in encoded_review])
def preprocess_text(text):
    words=text.lower().split()
    encoded_review=[word_index.get(word,2)+3 for word in words]
    padded_review=sequence.pad_sequences([encoded_review],maxlen=500)
    return padded_review


### Prediction
def predict_sentiment(review):
    processed_input=preprocess_text(review)
    prediction=model.predict(processed_input)
    sentiment='Positive' if prediction[0][0]>0.5 else 'Negative'
    return sentiment,prediction[0][0]

### Streamlit app
st.title('Rating sentiments of movie reviews')
st.write('Enter a movie review to classify it')

### user input
user_input= st.text_area('Movie Review')

if st.button('Classify'):
    processed_input=preprocess_text(user_input)

    ### make predictions
    prediction=model.predict(processed_input)
    sentiment='Positive' if prediction[0][0]>0.5 else 'Negative'

    ### Dispaly the results
    st.write(f'Sentiment:{sentiment}')
    st.write(f'prediction_score:{prediction[0][0]}')
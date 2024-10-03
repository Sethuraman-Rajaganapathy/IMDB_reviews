# IMDB_reviews
This project is a web application that classifies the sentiment of movie reviews (either **Positive** or **Negative**) using a pre-trained Recurrent Neural Network (RNN) model on the IMDb dataset. The app is built using Streamlit and TensorFlow.

## Features

- **Sentiment Classification**: The model takes a user-inputted movie review and classifies it as either positive or negative.
- **Pre-trained RNN Model**: Uses a simple RNN trained on the IMDb movie reviews dataset.
- **Web Interface**: The app provides a simple interface for users to input text and receive sentiment analysis in real-time.

## How It Works

1. **Input**: The user enters a movie review in the provided text area.
2. **Text Preprocessing**: The input is converted to lowercase, tokenized, and padded to a fixed length of 500 words.
3. **Prediction**: The RNN model predicts the sentiment of the review based on the IMDb movie reviews dataset.
4. **Output**: The result is displayed with the predicted sentiment (`Positive` or `Negative`) and the corresponding confidence score.

## Tech Stack

- **Python**: Core language used.
- **TensorFlow**: Machine learning framework used for loading the pre-trained RNN model.
- **Pandas & NumPy**: For handling data.
- **Streamlit**: For building the web interface.

## Installation

1. Clone this repository:
    ```bash
    git clone <repository-url>
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Download the pre-trained model file `simple_rnn_imdb_reviews.h5` and place it in the root directory of the project.

4. Run the Streamlit app:
    ```bash
    streamlit run app.py
    ```

## Usage

1. Open your web browser and go to `http://localhost:8501/`.
2. Enter a movie review in the text area.
3. Click the "Classify" button to see the sentiment prediction and confidence score.

## Code Overview

- **Preprocessing Function**:
    - `preprocess_text(text)`: Converts user input into a format the model can interpret by tokenizing and padding the text.
    
- **Decoding Function**:
    - `decoded_review(encoded_review)`: Decodes the IMDb encoded review back to readable text.

- **Prediction Function**:
    - `predict_sentiment(review)`: Classifies the input review as positive or negative using the pre-trained RNN model.

- **Streamlit UI**:
    - Provides a simple interface for entering a review and displays the result.

## Example

```plaintext
Input: "This movie was absolutely amazing, with stunning performances!"
Output: Sentiment: Positive
        Confidence: 0.85

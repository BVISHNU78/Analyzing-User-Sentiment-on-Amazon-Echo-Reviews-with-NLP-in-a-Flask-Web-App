# Analyzing-User-Sentiment-on-Amazon-Echo-Reviews-with-NLP-in-a-Flask-Web-App
This project focuses on analyzing customer reviews of the Amazon Echo using Natural Language Processing techniques to uncover underlying user sentiment. By building a lightweight and interactive web application with Flask, the project enables real-time sentiment classification of review data categorizing feedback into positive/negative/neutral.  

![Screenshot 2025-05-24 225102](https://github.com/user-attachments/assets/93c34582-b547-4301-ace7-aba0c071f5fd)


# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate

# Run the Flask app
python app.py

# Open the app in your browser
http://127.0.0.1:5000


A Flask web application that analyzes user sentiment on Amazon Echo product reviews using Natural Language Processing (NLP).
This project collects Amazon Echo user reviews and uses NLP techniques to analyze the sentiment (positive, negative, neutral) expressed in the reviews. The results are displayed in a simple and interactive Flask web app where users can input reviews and get instant sentiment feedback.

## Features

- Sentiment classification (Positive, Negative, Neutral) on user reviews  
- Interactive web interface built with Flask  
- Clean and user-friendly UI  
- Uses popular NLP libraries for text preprocessing and sentiment analysis  

## Tech Stack

- Python 3.x  
- Flask web framework  
- NLP libraries: NLTK, TextBlob, or similar  
- HTML, CSS, Bootstrap for frontend  
- Pandas for data manipulation  

## Installation

1. Clone the repository:

```bash
git clone https://github.com/BVISHNU78/Analyzing-User-Sentiment-on-Amazon-Echo-Reviews-with-NLP-in-a-Flask-Web-App.git
cd Analyzing-User-Sentiment-on-Amazon-Echo-Reviews-with-NLP-in-a-Flask-Web-App
How It Works
The user inputs a product review.

The text is cleaned and preprocessed (tokenization, stopword removal, etc.).

Sentiment analysis is performed using NLP libraries/models.

The predicted sentiment label is displayed on the webpage.

Data
The dataset contains Amazon Echo product reviews collected from [source, e.g., Kaggle or Amazon].

Reviews are labeled or inferred for sentiment to train/test the model.

Limitations & Future Work
Currently supports text-only input; no audio or video reviews.

Sentiment model can be improved with fine-tuning or more advanced transformers.

Adding user authentication and saving past analysis results.

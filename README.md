# Email Spam Classifier

This project is an Email Spam Classifier that uses a Naive Bayes model to predict whether an email is spam or not. The model has been trained using a dataset of emails and can classify new emails based on their content.

## Features

- **Spam Classification:** Enter an email message, and the model will classify it as either "Spam" or "Not Spam".
- **Streamlit Interface:** A user-friendly web interface built with Streamlit.
- **Tokenization and Lemmatization:** Preprocessing of email content using spaCy to improve model accuracy.

## Live Demo

You can access the live demo of the Email Spam Classifier [here](https://email-spam-classifier-rvab.onrender.com).

## Getting Started

### Prerequisites

Ensure you have the following installed:

- Python 3.7 or higher
- pip (Python package installer)

### Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/your-username/email-spam-classifier.git
    cd email-spam-classifier
    ```

2. Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

3. Download the spaCy model:

    ```bash
    python -m spacy download en_core_web_sm
    ```

### Running the App

1. Run the Streamlit app:

    ```bash
    streamlit run app.py
    ```

2. Open your web browser and go to the URL provided by Streamlit (usually `http://localhost:8501`).

### Usage

- Enter the text of an email in the provided text box.
- Click the "Classify" button.
- The app will display whether the email is "Spam" or "Not Spam" in a color-coded format (red for spam, green for not spam).



from flask import Flask, request, render_template, jsonify
import pickle
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import string

# Download NLTK stopwords (ensure it's available)
nltk.download('stopwords')

# Initialize stemmer
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

# Load trained model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))  #
# Load the same vectorizer used in training

# Text preprocessing function
def transform_text(text):
    text = text.lower()  # Lowercase
    text = nltk.word_tokenize(text)  # Tokenization

    # Remove special characters
    y = [i for i in text if i.isalnum()]

    # Remove stopwords and punctuation
    y = [i for i in y if i not in stop_words and i not in string.punctuation]

    # Apply stemming
    y = [ps.stem(i) for i in y]

    return " ".join(y)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    message = request.form["message"]

    # Apply preprocessing
    transformed_message = transform_text(message)

    # Convert to vector
    vectorized_message = vectorizer.transform([transformed_message])

    # Predict
    prediction = model.predict(vectorized_message)[0]
    result = "Spam" if prediction == 1 else "Not Spam"

    return render_template("index.html", prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
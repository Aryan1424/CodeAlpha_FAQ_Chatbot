from flask import Flask, render_template, request, jsonify
import nltk
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from faqs import faqs

nltk.download('punkt')

app = Flask(__name__)

def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text

questions = list(faqs.keys())
processed_questions = [preprocess(q) for q in questions]

vectorizer = TfidfVectorizer(tokenizer=nltk.word_tokenize)
faq_vectors = vectorizer.fit_transform(processed_questions)

def get_best_answer(user_input):
    user_input = preprocess(user_input)
    user_vector = vectorizer.transform([user_input])

    similarity = cosine_similarity(user_vector, faq_vectors)
    index = similarity.argmax()

    if similarity[0][index] < 0.3:
        return "Sorry, I couldn't understand your question."

    return faqs[questions[index]]

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json["message"]
    reply = get_best_answer(user_message)
    return jsonify({"reply": reply})

if __name__ == "__main__":
    app.run(debug=True)

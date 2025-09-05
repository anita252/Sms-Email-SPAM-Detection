from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# ---- Load your trained model and vectorizer ----
import joblib

model = joblib.load("model.joblib")
vectorizer = joblib.load("vectorizer.joblib")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        message = request.form["message"]
        data = [message]
        vect = vectorizer.transform(data).toarray()
        prediction = model.predict(vect)[0]

        result = "🚨 Spam Message" if prediction == 1 else "✅ Not Spam (Ham)"
        return render_template("index.html", prediction=result, input_text=message)

if __name__ == "__main__":
    app.run(debug=True)

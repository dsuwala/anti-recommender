from flask import Flask, render_template, request, jsonify
import requests
app = Flask(__name__)


@app.route("/")
def hello_world():
    return render_template("index.html")


@app.route("/api/recommend", methods=["POST"])
def recommend():
    data = request.json
    query = data.get("query")

    # Make request to the recommender API
    response = requests.post(
        "http://localhost:8000/recommend",
        json={
            "movie_title": query,
        }
    )
    return response.json()


if __name__ == "__main__":
    app.run(debug=True)

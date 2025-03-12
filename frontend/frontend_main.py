from flask import Flask, render_template, request, jsonify
import requests
app = Flask(__name__)


@app.route("/")
def hello_world():
    return render_template("index.html")


BACKEND_URL = "http://backend:8000"


@app.route("/api/recommend", methods=["POST"])
def recommend():
    data = request.json
    query = data.get("query")

    json = {"movie_title": query}

    if data.get("year"):
        year = data.get("year")
        json.update({"year": year})

    # Make request to the recommender API
    response = requests.post(
        f"{BACKEND_URL}/recommend",
        json=json
    )

    if "error" in response.json().keys():
        possible_matches = [f"Title: {result[0]}, year: {int(result[1])}" for result in response.json()["possible_matches"]]

        result = jsonify({
            "query": query,
            "possible_matches": possible_matches
        })

        return result
    else:

        recommendations = [f"Title: {result['standardized_title']}, year: {int(result['year'])}, rating: {result['rating']:.2f}" for result in response.json()["recommendations"]]
        result = jsonify({
            "query": query,
            "results": recommendations
        })

        return result


if __name__ == "__main__":
    app.run(debug=False)

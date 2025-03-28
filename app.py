from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from models.recommender import get_recommendations, load_data

app = Flask(__name__)

# Load dataset and model
df, df_scaled, neigh = load_data()

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        song_name = request.form["song_name"].strip()
        recommendations = get_recommendations(song_name, df, df_scaled, neigh)

        if recommendations is not None:
            return render_template("results.html", song_name=song_name, recommendations=recommendations)
        else:
            return render_template("index.html", error=f"Song '{song_name}' not found in the dataset.")

    return render_template("index.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)

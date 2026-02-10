import json
import pickle
import requests
import bs4 as bs
import numpy as np
import pandas as pd
import urllib.request
import os
from flask import Flask, render_template, request, jsonify
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

# Initialize Flask app
app = Flask(__name__)

# Load models and data
try:
    # Use environment variable or default path
    artifacts_path = os.environ.get('ARTIFACTS_PATH', 'Artifacts')
    clf = pickle.load(open(f"{artifacts_path}/nlp_model.pkl", 'rb'))
    vectorizer = pickle.load(open(f"{artifacts_path}/tranform.pkl",'rb'))
    data = pd.read_csv(f"{artifacts_path}/main_data.csv")
    
    # Create similarity matrix
    cv = CountVectorizer()
    count_matrix = cv.fit_transform(data['comb']) 
    similarity = cosine_similarity(count_matrix)
    
    print("Models and data loaded successfully")
except Exception as e:
    print(f"Error loading models/data: {e}")
    clf = None
    vectorizer = None
    data = None
    similarity = None

def rcmd(m):
    """Get movie recommendations"""
    if data is None or similarity is None:
        return ['System not ready - models not loaded']
        
    m = m.lower()
    if m not in data['movie_title'].unique():
        return ['Sorry! The movie you requested is not in our database. Please check the spelling or try with some other movies']
    else:
        i = data.loc[data['movie_title']==m].index[0]
        lst = list(enumerate(similarity[i]))
        lst = sorted(lst, key = lambda x:x[1] ,reverse=True)
        lst = lst[1:11] # excluding first item since it is the requested movie itself
        l = []
        for i in range(len(lst)):
            a = lst[i][0]
            l.append(data['movie_title'][a])
        return l

def convert_to_list(my_list):
    """Convert string representation of list to actual list"""
    my_list = my_list.split('","')
    my_list[0] = my_list[0].replace('["','')
    my_list[-1] = my_list[-1].replace('"]','')
    return my_list

def get_suggestions():
    """Get movie suggestions for autocomplete"""
    if data is None:
        return []
    return list(data['movie_title'].str.capitalize())

# Routes
@app.route("/")
@app.route("/home")
def home():
    suggestions = get_suggestions()
    return render_template('home.html', suggestions=suggestions)

@app.route("/api/similarity", methods=["POST"])
def similarity():
    try:
        movie = request.json.get('name', '')
        if not movie:
            return jsonify({'error': 'Movie name is required'}), 400
            
        rc = rcmd(movie)
        if isinstance(rc, str):
            return jsonify({'error': rc}), 404
        else:
            return jsonify({'recommendations': rc})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route("/api/recommend", methods=["POST"])
def recommend():
    try:
        # Get data from request
        title = request.json.get('title', '')
        if not title:
            return jsonify({'error': 'Title is required'}), 400
            
        # For simplicity, returning basic recommendation data
        # In a full implementation, you'd process all the movie details
        recommendations = rcmd(title)
        
        if isinstance(recommendations, str):
            return jsonify({'error': recommendations}), 404
            
        return jsonify({
            'title': title,
            'recommendations': recommendations,
            'status': 'success'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route("/api/suggestions")
def suggestions():
    """API endpoint for autocomplete suggestions"""
    try:
        suggestions_list = get_suggestions()
        return jsonify({'suggestions': suggestions_list})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Vercel serverless function handler
def handler(event, context):
    from flask import Flask
    import os
    
    # Set environment for Vercel
    os.environ['ARTIFACTS_PATH'] = '/tmp/Artifacts'
    
    # Create a new app instance for Vercel
    vercel_app = Flask(__name__)
    
    @vercel_app.route("/")
    def vercel_home():
        return "Movie Recommendation System API - Vercel Deployment"
    
    @vercel_app.route("/api/health")
    def health():
        return jsonify({
            'status': 'healthy',
            'models_loaded': clf is not None,
            'data_loaded': data is not None
        })
    
    return vercel_app

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
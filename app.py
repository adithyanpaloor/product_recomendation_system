from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


from flask import Flask, render_template

# Initialize the Flask app
app = Flask(__name__)

# Define routes below
@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True)


# Load and preprocess the data
def preprocess_data():
    data = pd.read_csv('data.csv')
    data['Dimensions'] = data['Dimensions'].str.replace(' mm', '')
    data[['Length', 'Width', 'Height']] = data['Dimensions'].str.split(' x ', expand=True).astype(float)
    data['Volume'] = data['Length'] * data['Width'] * data['Height']
    return data[['Model', 'Price', 'Weight', 'Volume', 'Category', 'Sensor Type', 'Resolution (MP)', 'ISO Range', 'Video Resolution', 'Battery Life', 'Touchscreen', 'Wi-Fi Connectivity', 'Lens Mount', 'Viewfinder Type', 'Image Stabilization']]

# Load the data
camera_data = preprocess_data()

# Calculate cosine similarity between cameras
def calculate_similarity(data):
    features = data[['Price', 'Weight', 'Volume']]
    similarity_matrix = cosine_similarity(features)
    return similarity_matrix

# Get the similarity matrix
similarity_matrix = calculate_similarity(camera_data)

@app.route('/')
def home():
    # Show the first 145 cameras in tiles on the homepage
    cameras = camera_data[[
        'Model', 'Price', 'Category', 'Resolution (MP)', 'Weight', 
        'Sensor Type', 'ISO Range', 'Video Resolution', 'Battery Life'
    ]].head(145).to_dict(orient='records')
    return render_template('index.html', cameras=cameras)


@app.route('/similar/<model>')
def get_similar(model):
    # Get the index of the selected camera
    camera_idx = camera_data[camera_data['Model'] == model].index.values
    
    if len(camera_idx) == 0:
        return "Camera not found", 404
    
    # Get the similarity scores for the selected camera
    similarity_scores = similarity_matrix[camera_idx[0]]
    
    # Sort the cameras by similarity score (excluding the selected camera)
    similar_indices = similarity_scores.argsort()[::-1][1:8]
    
    # Get the top 7 most similar cameras
    similar_cameras = camera_data.iloc[similar_indices][['Model', 'Price', 'Category', 'Sensor Type', 'Resolution (MP)', 'ISO Range', 'Video Resolution', 'Battery Life', 'Touchscreen', 'Wi-Fi Connectivity', 'Lens Mount', 'Viewfinder Type', 'Image Stabilization']].to_dict(orient='records')
    
    return render_template('similar.html', camera=model, similar_models=similar_cameras)

if __name__ == '__main__':
    app.run(debug=True)

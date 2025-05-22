from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

class CustomKNN:
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None
    
    def fit(self, X, y):
        self.X_train = X.toarray()  # Convert sparse matrix to dense
        self.y_train = y
    
    def euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))
    
    def predict(self, X):
        X = X.toarray()  # Convert sparse matrix to dense
        predictions = []
        
        for x in X:
            # Calculate distances between x and all training samples
            distances = [self.euclidean_distance(x, x_train) for x_train in self.X_train]
            
            # Get indices of k nearest neighbors
            k_indices = np.argsort(distances)[:self.k]
            
            # Get the labels of k nearest neighbors
            k_nearest_labels = [self.y_train[i] for i in k_indices]
            
            # Majority vote
            most_common = max(set(k_nearest_labels), key=k_nearest_labels.count)
            predictions.append(most_common)
        
        return np.array(predictions)
    
    def predict_proba(self, X):
        X = X.toarray()  # Convert sparse matrix to dense
        probabilities = []
        
        for x in X:
            # Calculate distances between x and all training samples
            distances = [self.euclidean_distance(x, x_train) for x_train in self.X_train]
            
            # Get indices of k nearest neighbors
            k_indices = np.argsort(distances)[:self.k]
            
            # Get the labels of k nearest neighbors
            k_nearest_labels = [self.y_train[i] for i in k_indices]
            
            # Calculate probability for each class
            unique_labels = list(set(self.y_train))
            class_counts = [k_nearest_labels.count(label) for label in unique_labels]
            probs = [count/self.k for count in class_counts]
            
            probabilities.append(probs)
        
        return np.array(probabilities)

# Initialize data and model
data = {
    'message': [
        'I love this party!',
        'Hate this so much',
        'Amazing vibes',
        'Terrible day',
        'What a beautiful moment',
        'Worst experience ever',
        'This is fantastic',
        'I am feeling great',
        'This is awful',
        'Absolutely disgusting',
        'Wonderful experience',
        'Perfect day',
        'This is terrible',
        'I feel sad',
        'Best day ever',
        'India is  Prosperous country',
        'Pakistan is terror country',
        'India is helping Country'
    ],
    'sentiment': ['positive', 'negative', 'positive', 'negative', 'positive', 'negative',
                 'positive', 'positive', 'negative', 'negative', 'positive', 'positive',
                 'negative', 'negative', 'positive', 'positive', 'negative', 'positive']
}

df = pd.DataFrame(data)
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['message'])
y = df['sentiment']

knn = CustomKNN(k=3)
knn.fit(X, y)

@app.route('/')
def home():
    return render_template('sentiment_analysis.html')

# Remove these routes as they're no longer needed
# @app.route('/sentiment-analysis')
# def sentiment_analysis_page():
#     return render_template('sentiment_analysis.html')

@app.route('/analyze-sentiment', methods=['POST'])
def analyze_sentiment():
    try:
        data = request.json
        text = data.get('text', '')
        
        # Transform the input text
        text_vector = vectorizer.transform([text])
        
        # Predict sentiment
        sentiment = knn.predict(text_vector)[0]
        
        # Get probability scores
        probabilities = knn.predict_proba(text_vector)[0]
        confidence = max(probabilities)
        
        return jsonify({
            'sentiment': sentiment,
            'confidence': float(confidence)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run()
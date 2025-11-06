from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import os
import pandas as pd
import numpy as np
from datetime import datetime
import json
import random
import string

app = Flask(__name__)
CORS(app)

# Load models
MODEL_PATH = 'models/'
lr_model = joblib.load(os.path.join(MODEL_PATH, 'lr_model.pkl'))
tfidf_vectorizer = joblib.load(os.path.join(MODEL_PATH, 'tfidf_vectorizer.pkl'))
label_encoder = joblib.load(os.path.join(MODEL_PATH, 'label_encoder.pkl'))

# In-memory storage
users = {}
complaints_db = []
complaint_counter = 1000  # Starting counter

def generate_complaint_id():
    """Generate a unique complaint ID"""
    global complaint_counter
    complaint_counter += 1
    return f"C{complaint_counter:06d}"

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "message": "Backend is running"}), 200

@app.route('/api/signup', methods=['POST'])
def signup():
    try:
        data = request.json
        email = data.get('email')
        password = data.get('password')
        name = data.get('name')
        
        if not email or not password or not name:
            return jsonify({"error": "All fields are required"}), 400
        
        if email in users:
            return jsonify({"error": "User already exists"}), 400
        
        users[email] = {
            'name': name,
            'password': password,
            'email': email,
            'created_at': datetime.now().isoformat()
        }
        
        return jsonify({
            "message": "User created successfully",
            "user": {
                "name": name,
                "email": email
            }
        }), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/login', methods=['POST'])
def login():
    try:
        data = request.json
        email = data.get('email')
        password = data.get('password')
        
        if not email or not password:
            return jsonify({"error": "Email and password are required"}), 400
        
        user = users.get(email)
        if not user or user['password'] != password:
            return jsonify({"error": "Invalid credentials"}), 401
        
        return jsonify({
            "message": "Login successful",
            "user": {
                "name": user['name'],
                "email": user['email']
            }
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/classify', methods=['POST'])
def classify_complaint():
    try:
        data = request.json
        complaint_text = data.get('complaint_text', '')
        language = data.get('language', 'English')
        
        if not complaint_text:
            return jsonify({"error": "Complaint text is required"}), 400
        
        # Generate unique complaint ID
        complaint_id = generate_complaint_id()
        
        # Transform text using TfidfVectorizer
        text_vectorized = tfidf_vectorizer.transform([complaint_text])
        
        # Predict category
        prediction = lr_model.predict(text_vectorized)
        category = label_encoder.inverse_transform(prediction)[0]
        
        # Get prediction probabilities
        probabilities = lr_model.predict_proba(text_vectorized)[0]
        confidence = float(max(probabilities) * 100)
        
        # Store complaint
        complaint_record = {
            'id': complaint_id,
            'text': complaint_text,
            'category': category,
            'confidence': confidence,
            'language': language,
            'timestamp': datetime.now().isoformat(),
            'status': 'pending'
        }
        complaints_db.append(complaint_record)
        
        return jsonify({
            "complaint_id": complaint_id,
            "category": category,
            "confidence": confidence,
            "status": "success"
        }), 200
        
    except Exception as e:
        print(f"Error in classification: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/complaints', methods=['GET'])
def get_complaints():
    try:
        return jsonify({
            "complaints": complaints_db,
            "total": len(complaints_db)
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/dashboard/stats', methods=['GET'])
def get_dashboard_stats():
    try:
        total_complaints = len(complaints_db)
        pending = sum(1 for c in complaints_db if c['status'] == 'pending')
        resolved = sum(1 for c in complaints_db if c['status'] == 'resolved')
        
        # Category distribution
        categories = {}
        for complaint in complaints_db:
            cat = complaint['category']
            categories[cat] = categories.get(cat, 0) + 1
        
        # Critical complaints (low confidence)
        critical = sum(1 for c in complaints_db if c['confidence'] < 70)
        
        return jsonify({
            "total_complaints": total_complaints,
            "pending": pending,
            "resolved": resolved,
            "critical": critical,
            "categories": categories
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/complaint/<complaint_id>', methods=['PUT'])
def update_complaint(complaint_id):
    try:
        data = request.json
        status = data.get('status')
        
        for complaint in complaints_db:
            if complaint['id'] == complaint_id:
                if status:
                    complaint['status'] = status
                return jsonify({
                    "message": "Complaint updated successfully",
                    "complaint": complaint
                }), 200
        
        return jsonify({"error": "Complaint not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    app.run(debug=True, host='0.0.0.0', port=5000)
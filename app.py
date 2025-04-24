# app.py

import os
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify, send_from_directory
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import joblib
import logging
import json
from datetime import datetime, timedelta
import uuid
import time

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder='static')

# Create necessary directories
if not os.path.exists('models'):
    os.makedirs('models')
if not os.path.exists('static'):
    os.makedirs('static')
if not os.path.exists('data'):
    os.makedirs('data')

# History data file
HISTORY_FILE = 'data/history.json'

# Initialize history if not exists
if not os.path.exists(HISTORY_FILE):
    with open(HISTORY_FILE, 'w') as f:
        json.dump([], f)

# Function to get history
def get_history():
    try:
        with open(HISTORY_FILE, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error reading history: {str(e)}")
        return []

# Function to add to history
def add_to_history(entry):
    try:
        history = get_history()
        # Add timestamp and unique ID
        entry['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        entry['id'] = str(uuid.uuid4())
        
        # Add to history (limit to 100 recent entries)
        history.append(entry)
        if len(history) > 100:
            history = history[-100:]
            
        with open(HISTORY_FILE, 'w') as f:
            json.dump(history, f)
    except Exception as e:
        logger.error(f"Error saving to history: {str(e)}")

# Function to get model statistics
def get_model_stats():
    return {
        "model_type": "Isolation Forest",
        "contamination": "5%",
        "training_samples": 1000,
        "last_trained": datetime.now().strftime('%Y-%m-%d'),
        "parameters": {
            "n_estimators": 100,
            "max_samples": "auto",
            "random_state": 42
        }
    }

# Function to train or load the model
def get_model():
    model_path = 'models/isolation_forest.joblib'
    
    # Check if model already exists
    if os.path.exists(model_path):
        logger.info("Loading existing model from file")
        model = joblib.load(model_path)
        scaler = joblib.load('models/scaler.joblib')
        return model, scaler
    
    logger.info("Training new model")
    # Generate synthetic temperature data for training
    np.random.seed(42)
    min_temps = np.random.normal(10, 5, 1000)
    max_temps = min_temps + np.random.normal(15, 3, 1000)
    center_temps = (min_temps + max_temps) / 2
    
    # Combine features into a dataset
    data = np.column_stack((min_temps, max_temps, center_temps))
    
    # Create a scaler object
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    
    # Train isolation forest model
    model = IsolationForest(contamination=0.05, random_state=42)
    model.fit(data_scaled)
    
    # Save the model and scaler
    joblib.dump(model, model_path)
    joblib.dump(scaler, 'models/scaler.joblib')
    
    return model, scaler

# Function to predict if the temperature is an anomaly
def predict_anomaly(min_temp, max_temp, center_temp):
    try:
        logger.debug(f"Predicting with inputs: min={min_temp}, max={max_temp}, center={center_temp}")
        model, scaler = get_model()
        
        # Scale input data
        data = np.array([[float(min_temp), float(max_temp), float(center_temp)]])
        logger.debug(f"Input data shape: {data.shape}")
        
        data_scaled = scaler.transform(data)
        logger.debug(f"Scaled data: {data_scaled}")
        
        # Predict
        prediction = model.predict(data_scaled)
        score = model.score_samples(data_scaled)
        
        # Calculate threshold
        threshold = -0.5  # Default threshold
        
        # Calculate additional metrics for insights
        avg_temp = (float(min_temp) + float(max_temp) + float(center_temp)) / 3
        temp_range = float(max_temp) - float(min_temp)
        
        logger.debug(f"Prediction: {prediction}, Score: {score}")
        
        # Convert prediction to boolean (1 is normal, -1 is anomaly)
        is_anomaly = prediction[0] == -1
        
        # Determine severity level based on the anomaly score
        severity = "Low"
        if is_anomaly:
            if score[0] < -0.8:
                severity = "High"
            elif score[0] < -0.65:
                severity = "Medium"
                
        # Add insights based on the data
        insights = []
        if is_anomaly:
            if temp_range > 25:
                insights.append("The temperature range is unusually large")
            if center_temp > max_temp or center_temp < min_temp:
                insights.append("Center temperature is outside the min-max range")
            if abs(center_temp - (min_temp + max_temp) / 2) > 5:
                insights.append("Center temperature deviates significantly from the arithmetic mean")
        
        # Calculate confidence level (normalized score)
        confidence = max(0, min(100, int(100 * (1 - abs(score[0] + 0.5) / 2))))
        
        result = {
            "is_anomaly": bool(is_anomaly),
            "anomaly_score": float(score[0]),
            "threshold": threshold,
            "severity": severity,
            "confidence": confidence,
            "insights": insights,
            "avg_temp": round(avg_temp, 2),
            "temp_range": round(temp_range, 2)
        }
        
        # Add to history
        add_to_history({
            "min_temp": float(min_temp),
            "max_temp": float(max_temp),
            "center_temp": float(center_temp),
            "is_anomaly": bool(is_anomaly),
            "anomaly_score": float(score[0]),
            "severity": severity
        })
        
        return result
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}", exc_info=True)
        return {
            "error": str(e)
        }

# Function to generate synthetic trend data
def generate_trend_data():
    # Generate synthetic data for last 30 days
    data = []
    today = datetime.now()
    
    # Generate normal and anomaly counts
    np.random.seed(42)
    for i in range(30):
        day = today - timedelta(days=30-i)
        normal_count = int(np.random.normal(25, 5))
        anomaly_count = int(np.random.normal(3, 2))
        if anomaly_count < 0:
            anomaly_count = 0
            
        # Inject a spike for demonstration
        if i == 25:
            anomaly_count = 15
            
        data.append({
            'date': day.strftime('%Y-%m-%d'),
            'normal': normal_count,
            'anomaly': anomaly_count
        })
        
    return data

# Function to generate insights from historical data
def generate_insights():
    try:
        history = get_history()
        if not history:
            return []
            
        insights = []
        
        # Count anomalies
        anomaly_count = sum(1 for entry in history if entry.get('is_anomaly', False))
        if anomaly_count > 0:
            insights.append({
                "title": "Anomaly Frequency",
                "content": f"{anomaly_count} anomalies detected in recent history ({len(history)} records)",
                "icon": "chart-line"
            })
            
        # Check for patterns in anomalies
        if anomaly_count >= 3:
            # Simple analysis: calculate average temperatures for anomalies
            anomalies = [entry for entry in history if entry.get('is_anomaly', False)]
            avg_min = sum(entry.get('min_temp', 0) for entry in anomalies) / len(anomalies)
            avg_max = sum(entry.get('max_temp', 0) for entry in anomalies) / len(anomalies)
            
            insights.append({
                "title": "Anomaly Pattern",
                "content": f"Anomalies tend to occur with min temp around {avg_min:.1f}°C and max temp around {avg_max:.1f}°C",
                "icon": "search"
            })
            
        # Time-based insights
        if len(history) >= 5:
            recent_count = sum(1 for entry in history[-5:] if entry.get('is_anomaly', False))
            if recent_count >= 3:
                insights.append({
                    "title": "Recent Trend",
                    "content": "High frequency of anomalies in recent measurements. Consider system inspection.",
                    "icon": "exclamation-triangle"
                })
                
        return insights
    except Exception as e:
        logger.error(f"Error generating insights: {str(e)}")
        return []

# Function to cluster data for patterns
def cluster_data():
    try:
        history = get_history()
        if len(history) < 10:
            return {"error": "Not enough data for clustering"}
            
        # Extract features
        features = np.array([[
            entry.get('min_temp', 0),
            entry.get('max_temp', 0),
            entry.get('center_temp', 0)
        ] for entry in history])
        
        # Determine optimal number of clusters (simple approach)
        n_clusters = min(3, len(features) // 5)
        if n_clusters < 2:
            n_clusters = 2
            
        # Apply K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(features)
        centers = kmeans.cluster_centers_
        
        # Group data by cluster
        clusters = []
        for i in range(n_clusters):
            cluster_points = features[labels == i]
            anomaly_count = sum(1 for j, entry in enumerate(history) if labels[j] == i and entry.get('is_anomaly', False))
            total_count = sum(1 for j in range(len(labels)) if labels[j] == i)
            
            clusters.append({
                "id": i,
                "center": {
                    "min_temp": round(centers[i][0], 2),
                    "max_temp": round(centers[i][1], 2),
                    "center_temp": round(centers[i][2], 2)
                },
                "points": len(cluster_points),
                "anomaly_percentage": round(anomaly_count / total_count * 100 if total_count > 0 else 0, 1)
            })
            
        return {"clusters": clusters}
    except Exception as e:
        logger.error(f"Error clustering data: {str(e)}")
        return {"error": str(e)}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        logger.debug(f"Received data: {data}")
        
        # Simulate processing time for better UX
        time.sleep(0.5)
        
        min_temp = float(data['min_temp'])
        max_temp = float(data['max_temp'])
        center_temp = float(data['center_temp'])
        
        result = predict_anomaly(min_temp, max_temp, center_temp)
        
        if "error" in result:
            return jsonify({"error": result["error"]}), 400
            
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error in predict route: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 400

@app.route('/generate_data', methods=['GET'])
def generate_data():
    try:
        # Generate some sample data for visualization
        np.random.seed(42)
        data = []
        
        # Generate normal data
        for _ in range(100):
            min_temp = np.random.normal(10, 3)
            max_temp = min_temp + np.random.normal(15, 2)
            center_temp = (min_temp + max_temp) / 2
            data.append({
                'min_temp': round(min_temp, 1),
                'max_temp': round(max_temp, 1),
                'center_temp': round(center_temp, 1)
            })
        
        # Generate some anomalies
        for _ in range(10):
            min_temp = np.random.normal(10, 10)
            max_temp = min_temp + np.random.normal(25, 8)
            center_temp = np.random.normal((min_temp + max_temp) / 2, 5)
            data.append({
                'min_temp': round(min_temp, 1),
                'max_temp': round(max_temp, 1),
                'center_temp': round(center_temp, 1)
            })
        
        return jsonify(data)
    except Exception as e:
        logger.error(f"Error generating data: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 400

@app.route('/history', methods=['GET'])
def get_history_data():
    try:
        history = get_history()
        return jsonify(history)
    except Exception as e:
        logger.error(f"Error retrieving history: {str(e)}")
        return jsonify({"error": str(e)}), 400

@app.route('/trends', methods=['GET'])
def get_trends():
    try:
        trend_data = generate_trend_data()
        return jsonify(trend_data)
    except Exception as e:
        logger.error(f"Error generating trends: {str(e)}")
        return jsonify({"error": str(e)}), 400

@app.route('/model_info', methods=['GET'])
def get_model_information():
    try:
        model_info = get_model_stats()
        return jsonify(model_info)
    except Exception as e:
        logger.error(f"Error retrieving model info: {str(e)}")
        return jsonify({"error": str(e)}), 400

@app.route('/insights', methods=['GET'])
def get_data_insights():
    try:
        insights = generate_insights()
        return jsonify(insights)
    except Exception as e:
        logger.error(f"Error generating insights: {str(e)}")
        return jsonify({"error": str(e)}), 400

@app.route('/clusters', methods=['GET'])
def get_clusters():
    try:
        clusters = cluster_data()
        return jsonify(clusters)
    except Exception as e:
        logger.error(f"Error clustering data: {str(e)}")
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

import pickle
import requests
import subprocess
import os
import time

# Test Cases
MESSAGES = [
    r"Just a friendly reminder that the deadline for our current project is next Friday, March 24th. Please make sure all your work is submitted by 5:00 PM on that day.",
    r"This is to confirm our meeting scheduled for tomorrow, March 13th, at 10:00 AM in the conference room. We'll be discussing the new marketing strategy.",
    r"Congratulations! You've been selected to receive a free vacation to the Bahamas. Click here to claim your prize now!",
]

def test_docker():
    """Test Flask app running inside a Docker container."""
    # Stop and remove any existing container
    subprocess.run(["docker", "stop", "spam_classifier"], check=False)
    subprocess.run(["docker", "rm", "spam_classifier"], check=False)
    
    # Build and run the container
    subprocess.run(["docker", "build", "-t", "flask-spam-classifier", "."], check=True)
    subprocess.run(["docker", "run", "-d", "-p", "5000:5000", "--name", "spam_classifier", "flask-spam-classifier"], check=True)
    
    time.sleep(5)  # Give the server time to start

    # Test the /home endpoint
    response = requests.get("http://127.0.0.1:5000/")
    assert response.status_code == 200
    assert "Spam Classifier" in response.text

    # Test the /score endpoint
    for message in MESSAGES:
        response = requests.post("http://127.0.0.1:5000/score", data={"text": message})
        assert response.status_code == 200
        data = response.json()
        assert "prediction" in data and "propensity" in data
        assert isinstance(data["prediction"], bool)
        assert isinstance(data["propensity"], float)
        assert 0 <= data["propensity"] <= 1
        assert data["prediction"] in [0, 1]
    
    # Stop and remove the container
    subprocess.run(["docker", "stop", "spam_classifier"], check=True)
    subprocess.run(["docker", "rm", "spam_classifier"], check=True)
import requests
import base64
import json


# Base URL for vLLM server
base_url = "http://localhost:8000"


# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def start_profile():
    """
    Send a request to start profiling the vLLM server.
    """
    url = f"{base_url}/start_profile"
    try:
        response = requests.post(url)
        print("Profile started:", response.text)
        return response.text
    except Exception as e:
        print(f"Error starting profile: {e}")
        return None


def stop_profile():
    """
    Send a request to stop profiling the vLLM server and get the results.
    """
    url = f"{base_url}/stop_profile"
    try:
        response = requests.post(url)
        print("Profile stopped. Results:", response.text)
        return response.text
    except Exception as e:
        print(f"Error stopping profile: {e}")
        return None


# Path to your image
image_path = "/home/aidan/fireworks/aledade-image.png"

# Start profiling before the vision request
print(f"Starting vLLM profiling on {base_url}...")
start_profile()

# Getting the base64 string
# base64_image = encode_image(image_path)

with open("./aledade-prod.base64", "r") as file:
    base64_image = file.read()

# Payload for the API
payload = {
    "model": "microsoft/Phi-3-vision-128k-instruct",
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}},
            ],
        }
    ],
    "max_tokens": 80,
    "min_tokens": 80,
    "do_sample": True,
}

# Headers
headers = {"Content-Type": "application/json"}

# Making the vision request
print("Sending vision request...")
response = requests.post(f"{base_url}/v1/chat/completions", headers=headers, json=payload)

# Print the vision response
print("Vision response:", response.json())

# Stop profiling after the vision request
print("Stopping profile...")
stop_profile()

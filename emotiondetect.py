import os
import requests

# Hugging Face token from env
HF_TOKEN = os.environ.get("HUGGINGFACEHUB_API_TOKEN")

API_URL = "https://api-inference.huggingface.co/models/j-hartmann/emotion-english-distilroberta-base"
headers = {"Authorization": f"Bearer {HF_TOKEN}"}

sentence = "I am feeling really excited and happy today!"

response = requests.post(API_URL, headers=headers, json={"inputs": sentence})
results = response.json()

# Hugging Face API returns a nested list
if isinstance(results, list) and len(results) > 0 and isinstance(results[0], list):
    results = results[0]

# Print emotions
for r in results:
    print(f"Emotion: {r['label']}, Score: {r['score']:.4f}")

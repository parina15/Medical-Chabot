import json
import random
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score

# Load the dataset from the JSON file
with open('medical_dataset.json') as f:
    data = json.load(f)

# Extract patterns and tags
patterns = []
tags = []

for intent in data['intents']:
    for pattern in intent['patterns']:
        patterns.append(pattern)
        tags.append(intent['tag'])

# Create a DataFrame
df = pd.DataFrame({'patterns': patterns, 'tags': tags})

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['patterns'], df['tags'], test_size=0.2, random_state=42)

# Create a pipeline with TF-IDF and Logistic Regression
model = make_pipeline(TfidfVectorizer(), LogisticRegression(max_iter=1000))

# Train the model
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

def get_response(user_input):
    # Predict the tag for the user input
    predicted_tag = model.predict([user_input])[0]
    
    # Find the corresponding response
    for intent in data['intents']:
        if intent['tag'] == predicted_tag:
            return random.choice(intent['responses'])
    
    return "I'm sorry, I don't understand that."

def chat():
    print("Welcome to the Medical Chatbot! Type 'exit' to end the chat.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Chatbot: Goodbye!")
            break
        response = get_response(user_input)
        print(f"Chatbot: {response}")

# Start the chat
chat()
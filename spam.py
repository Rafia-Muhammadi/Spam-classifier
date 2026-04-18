import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Sample dataset
data = {
    'message': [
        'Win money now',
        'Hello friend how are you',
        'Free offer just click',
        'Let’s meet tomorrow',
        'Claim your prize now',
        'Are you coming to class'
    ],
    'label': [1, 0, 1, 0, 1, 0]  # 1 = spam, 0 = not spam
}

df = pd.DataFrame(data)

# Convert text to numbers
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['message'])

# Train model
model = MultinomialNB()
model.fit(X, df['label'])

# Test
while True:
    msg = input("Enter message: ")
    msg_vec = vectorizer.transform([msg])
    prediction = model.predict(msg_vec)

    if prediction[0] == 1:
        print("Spam 🚫")
    else:
        print("Not Spam ✅")

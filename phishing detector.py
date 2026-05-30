# ============================================
#  PHISHING EMAIL DETECTION SYSTEM
# ============================================

# Import required libraries
import pandas as pd
import numpy as np
import re

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ============================================
# 1. LOAD DATASET
# ============================================
data = pd.read_csv("emails.csv")

print("\n Dataset Loaded Successfully")
print(data.head())

# ============================================
# 2. FEATURE ENGINEERING (IMPORTANT PART)
# ============================================

def extract_features(text):
    """
    This function extracts security-related features from email text.
    """

    features = {}

    #  Count number of URLs in email
    features['url_count'] = len(re.findall(r'http[s]?://', text))

    #  Detect suspicious phishing words
    suspicious_words = [
        'urgent', 'free', 'click', 'login',
        'password', 'verify', 'update', 'account'
    ]

    features['suspicious_word_count'] = sum(
        word in text.lower() for word in suspicious_words
    )

    return features


# Apply feature extraction to dataset
feature_data = data['text'].apply(lambda x: pd.Series(extract_features(x)))

print("\n Extracted Features Sample:")
print(feature_data.head())

# ============================================
# 3. TEXT VECTORIZATION (TF-IDF)
# ============================================

vectorizer = TfidfVectorizer(stop_words='english', max_features=3000)

X_text = vectorizer.fit_transform(data['text']).toarray()

print("\n TF-IDF Vectorization Completed")

# ============================================
# 4. COMBINE FEATURES
# ============================================

X = np.hstack((X_text, feature_data.values))

# Convert labels into numeric format
y = data['label'].map({'safe': 0, 'phishing': 1})

print("\n Feature Matrix Shape:", X.shape)

# ============================================
# 5. TRAIN-TEST SPLIT
# ============================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

print("\n Data Split Completed")
print("Training Samples:", len(X_train))
print("Testing Samples:", len(X_test))

# ============================================
# 6. MODEL TRAINING
# ============================================

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

print("\n Model Training Completed")

# ============================================
# 7. MODEL PREDICTION
# ============================================

y_pred = model.predict(X_test)

# ============================================
# 8. EVALUATION
# ============================================

accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("\n====================================")
print("MODEL EVALUATION")
print("====================================")

print("\n Accuracy:", accuracy)

print("\n Confusion Matrix:\n", cm)

print("\n Classification Report:\n")
print(classification_report(y_test, y_pred))

# ============================================
# 9. REAL-TIME EMAIL TEST FUNCTION
# ============================================

def predict_email(email_text):

    # Extract features
    feat = extract_features(email_text)

    # Convert email text into TF-IDF vector
    text_vec = vectorizer.transform([email_text]).toarray()

    # Combine both features
    combined = np.hstack((text_vec, np.array(list(feat.values())).reshape(1, -1)))

    # Predict
    prediction = model.predict(combined)

    return "Phishing Email" if prediction[0] == 1 else "✅ Safe Email"


# ============================================
# 10. TEST THE MODEL
# ============================================

sample_email = """
URGENT! Your bank account is locked.
Click http://fake-login.com immediately to verify your password.
"""

print("\n====================================")
print(" SAMPLE TEST RESULT")
print("====================================")

print(predict_email(sample_email))

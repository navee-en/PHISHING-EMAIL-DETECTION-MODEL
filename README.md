# PHISHING-EMAIL-DETECTION-MODEL
Phishing Email Detection Model (ML Project)
 Objective

Build a machine learning model to classify emails as:

Phishing
Safe (Legitimate)

using text features like URLs, keywords, and email content.

 1. Dataset Format

You need a CSV file like this:

emails.csv
text	label
"Your account is suspended, click http://fake.com"	phishing
"Meeting scheduled for tomorrow at 10 AM"	safe
 4. Confusion Matrix Meaning
True Safe → Correct safe emails
True Phishing → Correct phishing emails
Errors → misclassified emails
 5. Key Features Implemented

✔ TF-IDF Text Analysis
✔ URL Detection
✔ Suspicious Keyword Analysis
✔ Logistic Regression Model
✔ Accuracy Evaluation
✔ Confusion Matrix
✔ Real-time email prediction

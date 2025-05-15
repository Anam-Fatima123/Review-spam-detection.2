import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import string
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.metrics import confusion_matrix
import joblib

# Step 1: Load the dataset
file_path = "deceptive-opinion.csv.csv"  # Replace with the correct file path
data = pd.read_csv(file_path)
data['label'] = data['label'].map({'CG':1,'OR':0})

print(data.columns)
print(data[['text','label']].head(100))
print(data)
# Ensure column names are correct
if 'text' not in data.columns or 'label' not in data.columns:
    raise ValueError("The dataset must have 'text' and 'label' columns.")

# Drop rows with missing values
data = data.dropna(subset=['text', 'label'])

# Remove punctuations and convert text to lowercase
def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

data['text'] = data['text'].apply(clean_text)

# Ensure label column is numeric
if data['label'].dtype == 'object':
    data['label'] = data['label'].map({'fake': 1, 'not_fake': 0})  # Adjust based on actual label names

# Step 3: Prepare features and target
X = data['text']  # Review text
y = data['label'].astype(int)  # Target labels (1 for fake, 0 for not fake)

# Step 4: Vectorize the text data
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X = vectorizer.fit_transform(X)

# Step 5: Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 6: Train the SVM model
svm = SVC(kernel='linear', C=1, random_state=42)
svm.fit(X_train, y_train)

# Step 7: Make predictions
y_pred = svm.predict(X_test)

# Step 8: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("\nModel Accuracy:", accuracy)
print("\nClassification Report:\n", classification_report(y_test, y_pred))
# Save the trained model
joblib.dump(svm, "svm_model.pkl")
print("Model saved as svm_model.pkl")

# Save the vectorizer
joblib.dump(vectorizer, "vectorizer.pkl")
print("Vectorizer saved as vectorizer.pkl")
# while True:
#     review = input("\nEnter a review (or type 'exit' to quit): ")
#     if review.lower() == "exit":
#         print("Exiting the program.")
#         break

#     # Transform the user input into the same format as training data
#     input_vector = vectorizer.transform([review])

#     # Predict using the trained model
#     prediction = svm.predict(input_vector)

#     # Display result
#     if prediction[0] == 1:
#         print("The review is genuine.")
#     elif prediction[0] == 0:
#         print("The review is fake")
#     else:
#         print("error:Unexpected prediction value.")
#Plot the confusion score



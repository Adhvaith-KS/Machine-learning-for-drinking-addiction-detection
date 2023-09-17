import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import psutil

df_cleaned = pd.read_csv('smoking_driking_dataset_Ver01.csv')

# The below is to define the target variable (y)
target_variable = 'DRK_YN'
y = df_cleaned[target_variable]

# dropping the target variable to include everything in X except y
X = df_cleaned.drop(columns=[target_variable])

# This is to split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

classifiers = {
#     "Logistic Regression": LogisticRegression(max_iter=1000),
#     "Decision Tree": DecisionTreeClassifier(),
#     "Random Forest": RandomForestClassifier(),
#     "Support Vector Machine": SVC(),
    "Naive Bayes": GaussianNB(),
    "K-Nearest Neighbors": KNeighborsClassifier()
}

for name, classifier in classifiers.items():
    print(f"Training {name}...")
    start_time = time.time()
    
    classifier.fit(X_train, y_train)
    end_time = time.time()
    training_time = end_time - start_time
    
    # To predict on the test data
    y_pred = classifier.predict(X_test)
    
    # To calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    # To calculate memory usage
    memory_usage = (psutil.Process().memory_info().rss / 1024)/1000  # in MB
    
    print(f"{name}:")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Training Time: {training_time:.2f} seconds")
    print(f"Memory Usage: {memory_usage:.2f} MB")
    # To print other relevant metrics
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    print("\n")

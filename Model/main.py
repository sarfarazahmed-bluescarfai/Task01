import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report


class QuestionClassifier:
    def __init__(self, data):
        self.data = data
        self.model = None
        self.vectorizer = TfidfVectorizer()

    def preprocess_data(self):
        self.data['label'] = self.data['Question'].apply(self._label_question)
        X = self.vectorizer.fit_transform(self.data['Question'])
        y = self.data['label']
        return train_test_split(X, y, test_size=0.2, random_state=42)

    def _label_question(self, question):
        keywords = ['mortgage', 'dividend',
                    'market capitalization', 'financial metrics', 'rating']
        return int(any(keyword in question.lower() for keyword in keywords))

    def train_model(self, X_train, y_train):
        self.model = MultinomialNB()
        self.model.fit(X_train, y_train)

    def evaluate_model(self, X_test, y_test):
        predictions = self.model.predict(X_test)
        print("Accuracy:", accuracy_score(y_test, predictions))
        print("Classification Report:\n",
              classification_report(y_test, predictions))

    def predict(self, question):
        if self.model is None:
            raise Exception("Model has not been trained yet!")
        question_vector = self.vectorizer.transform([question])
        return self.model.predict(question_vector)[0]


# Load data
file_path = 'App\data'
data = pd.read_excel(file_path)

# Initialize the classifier
classifier = QuestionClassifier(data)

# Preprocess the data and split into training and test sets
X_train, X_test, y_train, y_test = classifier.preprocess_data()

# Train the model
classifier.train_model(X_train, y_train)

# Evaluate the model
classifier.evaluate_model(X_test, y_test)

# Make a prediction
sample_question = "What is the overall latest rating for Amazon.com Inc?"
prediction = classifier.predict(sample_question)
print(
    f"Prediction for sample question: {'Stock Performance Related' if prediction else 'Not Stock Performance Related'}")

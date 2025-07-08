import pandas as pd
import string
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords')

# Load dataset (you can use a Kaggle dataset like 'Fake and Real News Dataset')
# Format: dataframe with 'title', 'text', 'label' (label = 'FAKE' or 'REAL')
data = pd.read_csv('news.csv')  # Ensure this CSV has 'title', 'text', 'label'

# Combine title and text
data['content'] = data['title'] + ' ' + data['text']

# Preprocessing function
def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    tokens = text.split()
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    return " ".join(tokens)

# Apply cleaning
data['content'] = data['content'].apply(clean_text)

# Labels (convert 'FAKE' to 0, 'REAL' to 1)
data['label'] = data['label'].map({'FAKE': 0, 'REAL': 1})

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    data['content'], data['label'], test_size=0.2, random_state=42
)

# TF-IDF Vectorization
tfidf = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Train Logistic Regression model
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# Predict
y_pred = model.predict(X_test_tfidf)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Prediction Function
def predict_news(title, text):
    combined = clean_text(title + ' ' + text)
    vec = tfidf.transform([combined])
    prediction = model.predict(vec)[0]
    confidence = model.predict_proba(vec).max() * 100
    label = "Real News" if prediction == 1 else "Fake News"
    print(f"\nPredicted Output:\n {label} (Confidence: {confidence:.2f}%)")

# Example Inputs
predict_news(
    "Government launches new policy to boost economy.",
    "The government has introduced a set of new economic reforms aimed at improving GDP growth by 5% in the upcoming year."
)

predict_news(
    "Aliens discovered working with government officials.",
    "A secret document has been leaked, revealing that extraterrestrial beings have been collaborating with high-ranking officials."
)

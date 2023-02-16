import pandas as pd
import re
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the dataset into a Pandas dataframe
data = pd.read_csv('emails.csv', encoding='latin-1')

# drop & rename columns
data = data.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1)
data = data.rename(columns={'v1': 'label', 'v2': 'text'})

# Convert labels to binary 0 and 1, where 1 represents spam and 0 represents ham
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# data cleaner and stemming
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()
def preprocess_text(text):
    """Convert text to lower case, remove punctuation, stop words, digits,
     special characters and stem remaining text"""
    text = text.lower()
    text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
    text = re.sub('\d+', '', text)
    text = ' '.join([word for word in text.split() if word not in stop_words])
    text = ' '.join([ps.stem(word) for word in text.split()])
    return text
data['text'] = data['text'].apply(preprocess_text)

# Use TfidfVectorizer to convert the preprocessed text into a matrix of TF-IDF features
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(data['text'])

# Split the dataset into training and testing sets
y = data['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the pipeline for feature extraction and model training
pipeline = Pipeline([ #chain vectorizer and multinomialNB
    ('tfidf', TfidfVectorizer()), #convert mail text into numerical
    ('model', MultinomialNB()) #classifier to check if spam
])

# Fit the pipeline to the training data
pipeline.fit(X_train, y_train)

# Evaluate the model on the testing data
accuracy = pipeline.score(X_test, y_test)
print(f"Accuracy: {accuracy:.2f}")

# Use the model to predict whether a new email is spam or not
new_email = "Get 50% off your next purchase at our store!"
prediction = pipeline.predict([new_email])
print(f"Prediction: {prediction[0]}")

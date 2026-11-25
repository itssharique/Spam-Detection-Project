import string
import pickle

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Make sure the NLTK data exists
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)

ps = PorterStemmer()
stop = set(stopwords.words('english'))


def transform_text(text):
    text = nltk.word_tokenize(str(text).lower())
    text = [w for w in text if w.isalnum()]
    text = [w for w in text if w not in stop and w not in string.punctuation]
    return " ".join(ps.stem(w) for w in text)


# 1. Load data
df = pd.read_csv('spam.csv', encoding='latin-1')[['v1', 'v2']]
df.columns = ['target', 'text']
df = df.drop_duplicates()
df['target'] = (df['target'] == 'spam').astype(int)

# 2. Preprocess
df['transformed'] = df['text'].apply(transform_text)

# 3. Fit the vectorizer, then transform
tfidf = TfidfVectorizer(max_features=3000)
tfidf.fit(df['transformed'])
X = tfidf.transform(df['transformed'])
print("Works before saving:", tfidf.transform(['free prize win']).shape)

# 4. Train the model
model = MultinomialNB()
model.fit(X, df['target'])
print("Training accuracy:", model.score(X, df['target']))

# 5. Save
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf, f)
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

# 6. Check that the saved files load and work
with open('vectorizer.pkl', 'rb') as f:
    v2 = pickle.load(f)
with open('model.pkl', 'rb') as f:
    m2 = pickle.load(f)

test = v2.transform([transform_text("Congratulations! You won a free prize. Call now to claim")])
print("Works after loading:", test.shape, "-> prediction:", m2.predict(test)[0], "(1 = spam)")
print("Done.")

import nltk
import pandas as pd

from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer


def tokenize(text):
    return [stemmer.stem(w) for w in nltk.tokenize.word_tokenize(text)]


stemmer = nltk.stem.PorterStemmer()
nltk.download('stopwords')
nltk.download('punkt')

news = fetch_20newsgroups(subset='all')

x, y = news.data, news.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=32)

nb_clf = Pipeline([('vectorizer', TfidfVectorizer(stop_words=nltk.corpus.stopwords.words('english'), tokenizer=tokenize)),
                   ('classifier', MultinomialNB(alpha=0.005))])

nb_clf.fit(x_train, y_train)

print("Accuracy", nb_clf.score(x_test, y_test))
print("Confusion Matrix")
confusion_matrix_df = pd.DataFrame(confusion_matrix(y_test, nb_clf.predict(x_test)))
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(confusion_matrix_df)
confusion_matrix_df.to_csv("out_nb.csv", sep='\t', encoding='utf-8')
import sys
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string
import joblib

def load_data(database_filepath):
    df = pd.read_sql_table('cleaned_data', database_filepath)
    X = df.message
    Y = df.drop(['id', 'original', 'message', 'genre'], axis=1)
    return X, Y, Y.columns


def tokenize(text, stopwords_=stopwords.words('english')):
    text = text.lower()
    # remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Reduce words to their root form
    words = text.split()
    words = [w for w in words if w not in stopwords_]
    lemmed = [WordNetLemmatizer().lemmatize(w) for w in words]
    return lemmed


def build_model():
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 3),
                                  tokenizer=tokenize)),
        ('clf', MultiOutputClassifier(estimator=RandomForestClassifier(max_depth=4,
                                                                       n_estimators=200,
                                                                       random_state=42)))])

    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    Y_pred = model.predict(X_test)
    Y_pred = pd.DataFrame(Y_pred, columns=category_names)
    for col in category_names:
        print(col)
        print(classification_report(Y_test[col], Y_pred[col]))
    return


def save_model(model, model_filepath):
    joblib.dump(model, model_filepath)
    return


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
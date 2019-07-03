""" ML Pipeline for Udacity desaster response excercise.
Loads data from sqlite data base, trains & optimizes a classifier, and saves
the optimized model to a pickle file.

"""
import sys
import pandas as pd
import numpy as np

from sqlalchemy import create_engine

import nltk
import re
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def load_data(database_filepath):
    """Loading cleaned data from sqlite database

    Args:
        database_filepath: .db file holding the cleanded desaster response dataset

    Returns:
        X: Array of messages (n x 1)
        Y: Array of message category data (n x 36)
        category_names: list of names of the 36 categories in Y

    """
    # connect to sqlite database
    engine = create_engine(''.join(['sqlite:///', database_filepath]))

    # read complete data set from table
    df = pd.read_sql_query("select * from DisasterResponse",engine)

    # select input & output colummns
    X = df['message'].values
    Y = df.drop(columns=['id','message', 'genre', 'original'])

    # get list of category names
    category_names = list(Y.columns.values)

    # return values only
    Y = Y.values

    return X, Y, category_names


def tokenize(text):
    """Tokenizing a text message. Removes URLs & stopwords, converts to lower
    case and applies lemmatizer.

    Args: string containing raw text message

    Returns: array of clean tokens

    """
    # regular expression to detect URLs
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

    # delete urls
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "")

    # stop words
    stop_words = stopwords.words("english")

    # normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    # tokenize and apply lemmatizer
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]

    return clean_tokens


def build_model():
    """Builing the model using pipeline & grid search to optimize parameters of
    ML pipeline

    Args:
        none

    Returns:
        model: sklearn pipeline & grid search optimization

    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(estimator=RandomForestClassifier()))
    ])

    # parameters of the pipeline
    # parameters = {
    #     'vect__ngram_range': ((1, 1), (1, 2)),
    #     'vect__max_df': (0.5, 0.75, 1.0),
    #     'tfidf__use_idf': (True, False),
    #     'clf__estimator__min_samples_split': [2, 3, 4],
    # }
    parameters = {
        'vect__ngram_range': ((1, 1), (1, 2)),
    }

    # optimal model using grid search
    model = GridSearchCV(pipeline, param_grid=parameters)

    return model


def evaluate_model(model, X_test, Y_test, category_names):
    """ evaluating the model on test data set and prints classification report
    for each output dimension

    Args:
        model: trained sklearn model
        X_test: array of test data (text messages)
        Y_test: array of test output (36 columns)
        category_names: list of column names of Y

    """
    # predict
    Y_pred = pipeline.predict(X_test)

    # print classification report
    for i in range(0,len(category_names)):
        print("Category: {}".format(category_names[i]))
        print(classification_report(Y_test[:,i],Y_pred[:,i]))


def save_model(model, model_filepath):
    """Saves model to pickle file.

    Args:
        model: trained sklearn model

        model_filepath: path to write model file to

    """
    pickle.dump(model, open(model_filepath, 'wb'))


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

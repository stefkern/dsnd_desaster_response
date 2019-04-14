import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """Loading the message and category data files.

    Args:
        messages_filepath: .csv file holding the message data
        categories_filepath: .csv file holding the category data

    Returns:
        merged_data : pandas datafram holding the raw data of the two data files merged
            based on message id

    """
    # load data files
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    # merge datasets
    merged_data = messages.merge(categories, how='left', on='id')

    return merged_data


def clean_data(df):
    """Cleaning the data
    Actions performed:
    - splitting categories and converting to 0 or 1
    - removing duplicates

    Args:
        df: pandas dataframe holding raw merged dataset

    Returns:
        df: pandas dataframe holding cleaned dataset

    """
    # splitting category string into columns
    categories = df['categories'].str.split(pat=';', expand=True)

    # getting category names from first row
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x: x[:-2])

    # rename the columns of `categories`
    categories.columns = category_colnames

    # loop over columns and convert values to just numbers 0 or 1
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).apply(lambda x: x[-1:])

        # convert column from string to numeric
        categories[column] = categories[column].astype(float)

    # drop the original categories column from `df`
    df.drop(columns=['categories'], inplace=True)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)

    # remove duplicates
    df.drop_duplicates(inplace=True, keep='first')

    return df


def save_data(df, database_filename):
    """Saving the cleaned data a sqlite database

    Args:
        df: pandas dataframe holding the cleaned data
        database_filename: a sqlite database file

    """
    engine = create_engine(''.join(['sqlite:///', database_filename]))
    df.to_sql(database_filename.replace(".db", ""), engine, index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()

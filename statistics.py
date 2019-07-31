import pymongo
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns


def collection_to_df(collection):
    data = pd.DataFrame(list(collection.find()))
    # Drop the _id column
    data = data.drop('_id', axis=1)
    return data


def countplot(col, title):
    sns.countplot(col)
    plt.title(title)
    plt.show()


def main():
    client = pymongo.MongoClient('localhost', 27017)
    db = client['yelp_db']
    collection_name = 'users_reviews_restaurants_join'
    # user stats
    collection = db[collection_name]
    df = collection_to_df(collection)

    #Stats
    countplot(df.stars, 'Distribution of stars' )

    countplot(df.funny, 'Distribution of funny reviews')
    countplot(df.useful, 'Distribution of useful reviews')
    countplot(df.cool, 'Distribution of cool reviews')



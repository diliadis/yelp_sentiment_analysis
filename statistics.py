import pymongo
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
import text_yelp


def collection_to_df(collection):
    data = pd.DataFrame(list(collection.find()))
    # Drop the _id column
    data = data.drop('_id', axis=1)
    return data


def countplot(col, title):
    sns.countplot(col)
    plt.title(title)
    plt.show()


def compute_plots(df):
    # Boxplot sentiment scores by class
    sns.boxplot(x='stars', y='polarity', data=df)
    plt.title('Stars/polarity')
    plt.show()

    sns.boxplot(x='stars', y='swnscore', data=df)
    plt.title('Stars/swnscore')
    plt.show()

    sns.boxplot(x='stars', y='compound', data=df)
    plt.title('Stars/compound')
    plt.show()

    # Attribute correlations with stars, heatmap
    stars = df.groupby('stars').mean()
    stars.corr()
    sns.heatmap(data=df.corr(), annot=True)
    plt.title('Heatmap: Correlation between features and stars')
    plt.show()

    # Word Frequencies
    # print('Word frequencies for whole text data:', text_yelp.word_freq(df['text']))
    text_yelp.word_cloud(df['lsswords'], 'word_clouds/full')
    # print('Word frequencies for class Stars-1 text data:', text_yelp.word_freq(stars1['text']))

    for star_rating in range(1, 6, 1):
        # print('Word frequencies for class '+str(star_rating)+' text data:', text_yelp.word_freq(df[df['stars'] == star_rating]['joined_lsswords']))
        text_yelp.word_cloud(df[df['stars'] == star_rating]['joined_lsswords'], 'word_clouds/rated_'+str(star_rating)+'_text')


def main():
    client = pymongo.MongoClient('localhost', 27017)
    db = client['yelp_db']
    collection_name = 'users_reviews_restaurants_join'
    collection = db[collection_name]
    df = collection_to_df(collection)

    # Stats
    countplot(df.stars, 'Distribution of stars' )
    countplot(df.funny, 'Distribution of funny reviews')
    countplot(df.useful, 'Distribution of useful reviews')
    countplot(df.cool, 'Distribution of cool reviews')



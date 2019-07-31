import pymongo
import pandas as pd


def main():
    client = pymongo.MongoClient('localhost', 27017)
    db = client['yelp_db']

    reviews_collection = db['reviews']
    # reviews_df = pd.DataFrame(list(reviews_collection.find()))

    restaurants_collection = db['restaurants']
    restaurants_df = pd.DataFrame(list(restaurants_collection.find()))

    users_collection = db['users']
    # users_df = pd.DataFrame(list(users_collection.find()))

    # find the top ten restaurant categories
    # top_categories = get_restaurant_categories(restaurants_df, 10)

    # filter the restaurant collection to get only the mexican ones
    mexican_restaurants_collection_name = 'mexican_restaurants'
    filter_on_restaurant_category(db, restaurants_df, 'Mexican', mexican_restaurants_collection_name)

    # join the mexican restaurants with the reviews collection
    joined_collection_name = 'reviews_mexican_restaurants_join'
    reviews_restaurants_join(db, reviews_collection, mexican_restaurants_collection_name, joined_collection_name)

    final_joined_collection_name = 'users_reviews_restaurants_join'
    # join the users collection with the result of the mexican_restaurants-reviews join
    users_reviews_restaurants_join(db, users_collection, db[joined_collection_name], final_joined_collection_name)


def reviews_restaurants_join(db, reviews_collection, restaurants_collection_name, joined_collection_name):
    # create pipeline that joins the reviews and restaurant collections
    pipeline1 = [
        {'$lookup': {'from': restaurants_collection_name, 'localField': 'business_id', 'foreignField': 'business_id',
                     'as': 'business'}},
        {'$unwind': '$business'},
        {'$project':
             {'funny': 1, 'user_id': 1, 'review_id': 1, 'text': 1, 'business_id': 1, 'stars': 1, 'useful': 1, 'cool': 1,
              'business.latitude': 1, 'business.longitude': 1, 'business.review_count': 1, 'business.stars': 1}}
    ]

    reviews_restaurants_join = db[joined_collection_name]

    for doc in reviews_collection.aggregate(pipeline1):
        reviews_restaurants_join.insert_one(doc)

    reviews_restaurants_join_df = pd.DataFrame(list(reviews_restaurants_join.find()))

    temp_df = pd.DataFrame(list(reviews_restaurants_join_df['business']))
    temp_df = temp_df.rename(columns={"stars": "average_stars"})
    final_df = pd.concat([reviews_restaurants_join_df[['funny', 'user_id', 'review_id', 'text', 'stars', 'useful', 'cool', 'business_id']], temp_df[['longitude', 'latitude', 'average_stars', 'review_count']]], axis=1)
    db[joined_collection_name].drop()
    db[joined_collection_name].insert_many(final_df.to_dict('records'))


# get all the different restaurant categories and find the top 10 must used
def get_restaurant_categories(df, n):
    cat = (df.categories.apply(pd.Series)
              .stack()
              .reset_index(level=1, drop=True)
              .to_frame('categories'))
    cat['index'] = cat.index
    top_categories = cat.groupby('categories')['index'].nunique().sort_values(ascending=False)
    return top_categories[:n]


# filter the collection and create a new one that contains a specific value in the categories attribute
def filter_on_restaurant_category(db, df, category, new_collection_name):
    # create a function that checks if the categories list attribute contains a specific name-category
    collist = db.list_collection_names()
    if new_collection_name not in collist:
        open_df = df[df['is_open'] == 1]
        category_function = lambda x: x.__contains__(category)
        # apply the above function to the restaurant dataframe. The result is a dataframe(filtered_df)
        # with only the mexican restaurants
        filtered_df = open_df.loc[df['categories'].map(category_function)].filter(items=['_id', 'business_id',
                                                                                    'latitude', 'longitude', 'name',
                                                                                    'review_count', 'stars'])
        # insert the new updated dataframe to a collection in mongodb
        db[new_collection_name].insert_many(filtered_df.to_dict('records'))


# filter the bad quality users and join them with the joined collection of restaurants and reviews
def users_reviews_restaurants_join(db, users_collection, joined_collection, new_collection_name):
    df = pd.DataFrame(list(users_collection.find()))

    # calculate number of friends and elite years
    num_function = lambda x: len(x)
    df['num_of_friends'] = df['friends'].map(num_function)
    df['elite_years'] = df['elite'].map(num_function)
    fdf = df.loc[(df['num_of_friends'] != 0) | (df['compliment_cool'] != 0) | (df['compliment_cute'] != 0) | (
            df['compliment_funny'] != 0) | (df['compliment_hot'] != 0) | (df['compliment_list'] != 0) | (
                       df['compliment_more'] != 0) | (df['compliment_note'] != 0) | (df['compliment_photos'] != 0) | (
                       df['compliment_plain'] != 0) | (df['compliment_profile'] != 0) | (
                       df['compliment_writer'] != 0) | (df['funny'] != 0) | (df['fans'] != 0) | (df['useful'] != 0)]
    renamed_fdf = fdf.rename(columns={"cool": "user_cool", "average_stars": "user_average_stars", "funny": "user_funny", "review_count": "user_review_count", "useful": "user_useful"})

    # load the reviews_mexican_restaurants_join collection to a dataframe

    rdf = pd.DataFrame(list(joined_collection.find()))

    # join the two dataframes to create the final
    final_df = pd.merge(rdf, renamed_fdf[['user_id', 'yelping_since','user_average_stars', 'compliment_cool', 'compliment_cute', 'compliment_funny',
                             'compliment_hot', 'compliment_list', 'compliment_more', 'compliment_note',
                             'compliment_photos', 'compliment_plain', 'compliment_profile', 'compliment_writer', 'user_cool',
                             'elite_years', 'elite', 'friends', 'fans', 'user_funny', 'name', 'user_review_count', 'user_useful', 'num_of_friends']],
                        on='user_id', how='inner')

    # print(final_df.groupby('stars')['_id'].nunique())
    db[new_collection_name].insert_many(final_df.to_dict('records'))
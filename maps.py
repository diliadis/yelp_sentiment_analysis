from pymongo import MongoClient
import pandas as pd
import numpy as np
import gmplot


def get_restaurants_heatmap(over_limit=2, output_filename='heatmap'):
    client = MongoClient('localhost', 27017)
    db = client.yelp_db
    collection = db.restaurant
    data = pd.DataFrame(list(collection.find()))

    #Set coordinates of Las Vegas
    restaur_map = gmplot.GoogleMapPlotter(36.115, -115.173, 11)

    over_lat = list(data.loc[data['stars'] > over_limit, 'latitude'])
    over_long = list(data.loc[data['stars'] > over_limit, 'longitude'])

    restaur_map.heatmap(over_lat, over_long)
    restaur_map.draw(output_filename+".html")


def visualize_best_and_worst_restaurants(n=10):
    client = MongoClient('localhost', 27017)
    db = client.yelp_db
    restaurants_collection_name = 'mexican_restaurants'
    restaurants_collection = db[restaurants_collection_name]
    df = pd.DataFrame(list(restaurants_collection.find()))

    coordinates = df.sort_values('stars')[['longitude', 'latitude', 'stars']]

    print('The '+str(n)+' best rated restaurants have a mean star rating of '+str(np.mean(coordinates.tail(n)['stars'].tolist())))
    print('The '+str(n)+' worst rated restaurants have a mean star rating of '+str(np.mean(coordinates.head(n)['stars'].tolist())))

    n_best_coordinates = coordinates.tail(n)[['longitude', 'latitude']]
    n_worst_coordinates = coordinates.head(n)[['longitude', 'latitude']]

    restaur_map = gmplot.GoogleMapPlotter(36.115, -115.173, 11)
    restaur_map.scatter(n_best_coordinates['latitude'].tolist(), n_best_coordinates['longitude'].tolist(), c='b', s=200, marker=False)
    restaur_map.scatter(n_worst_coordinates['latitude'].tolist(), n_worst_coordinates['longitude'].tolist(), c='r', s=200, marker=False)

    restaur_map.draw('/maps/n_'+str(n)+"_blue_best_red_worst"+ ".html")


def get_restaurants_on_map(num_of_categories=5, output_filename='standard_map'):
    client = MongoClient('localhost', 27017)
    db = client.yelp_db
    collection = db.restaurant
    data = pd.DataFrame(list(collection.find()))

    #Set coordinates of Las Vegas
    restaur_map = gmplot.GoogleMapPlotter(36.115, -115.173, 11)

    #Take the latitudes and longitudes of all restaurants
    lat = list(data['latitude'].values)
    long = list(data['longitude'].values)

    if num_of_categories == 3:
        # 3 different colors for restaurants according to their stars
        # Take the latitudes and longitudes of excellent restaurants (stars > 3.5)
        excellent_lat = list(data.loc[data['stars']>3.5,'latitude'])
        excellent_long = list(data.loc[data['stars']>3.5,'longitude'])

        # Take the latitudes and longitudes of good restaurants (2.5 <= stars <= 3.5)
        good_lat = list(data.loc[(data['stars']>=2.5) & (data['stars']<=3.5),'latitude'])
        good_long = list(data.loc[(data['stars']>=2.5) & (data['stars']<=3.5),'longitude'])

        # Take the latitudes and longitudes of poor restaurants (stars < 2.5)
        poor_lat = list(data.loc[data['stars']<2.5,'latitude'])
        poor_long = list(data.loc[data['stars']<2.5,'longitude'])

        # 3 categories of restaurants with different colors
        restaur_map.scatter(excellent_lat, excellent_long, c='r', s=100, marker=False)
        restaur_map.scatter(good_lat, good_long, c='b', s=100, marker=False)
        restaur_map.scatter(poor_lat, poor_long, color='c', s=100, marker=False)
    elif num_of_categories == 5:
        # 5 different colors for restaurants according to their stars
        # Take the latitudes and longitudes of excellent restaurants (stars > 4)
        excellent_lat = list(data.loc[data['stars']>4,'latitude'])
        excellent_long = list(data.loc[data['stars']>4,'longitude'])

        # Take the latitudes and longitudes of very good restaurants (3 < stars <= 4)
        very_good_lat = list(data.loc[(data['stars']>3) & (data['stars']<=4),'latitude'])
        very_good_long = list(data.loc[(data['stars']>3) & (data['stars']<=4),'longitude'])

        # Take the latitudes and longitudes of good restaurants (2 < stars <= 3)
        good_lat = list(data.loc[(data['stars']>2) & (data['stars']<=3),'latitude'])
        good_long = list(data.loc[(data['stars']>2) & (data['stars']<=3),'longitude'])

        # Take the latitudes and longitudes of poor restaurants (1 < stars <= 2)
        poor_lat = list(data.loc[(data['stars']>1) & (data['stars']<=2),'latitude'])
        poor_long = list(data.loc[(data['stars']>1) & (data['stars']<=2),'longitude'])

        # Take the latitudes and longitudes of very poor restaurants (stars <= 1)
        very_poor_lat = list(data.loc[data['stars']<=1,'latitude'])
        very_poor_long = list(data.loc[data['stars']<=1,'longitude'])

        # 5 categories of restaurants with different colors
        restaur_map.scatter(excellent_lat, excellent_long, c='c', s=100, marker=False)
        restaur_map.scatter(very_good_lat, very_good_long, c='b', s=100, marker=False)
        restaur_map.scatter(good_lat, good_long, c='r', s=100, marker=False)
        restaur_map.scatter(poor_lat, poor_long, c='m', s=100, marker=False)
        restaur_map.scatter(very_poor_lat, very_poor_long, c='y', s=100, marker=False)
    else:
        print('invalid number of rating categories to be visualized on a map')
    if num_of_categories == 5 or num_of_categories == 3:
        restaur_map.draw('/maps/'+output_filename+".html")
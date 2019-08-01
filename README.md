# Emotion detection in the Yelp Dataset Challenge.

This is a python project is an attempt to create a classifier that performs sentiment analysis in dataset provided by the 
[Yelp Dataset Challenge](https://www.yelp.com/dataset/challenge).

## Description of dataset 

The data, as provided by the Yelp Dataset Challenge, is divided into three separate datasets. These datasets were stored
in seperate collections in a mongoDB database.

### 1) yelp_restaurant_Las_Vegas

Data for 5899 restaurants in the Las Vegas area. Contains information about average rating, exact coordinates, address, 
opening hours, city, postal code, number of total ratings, restaurant category (eg Italian) etc.


### 2) reviews_Las_Vegas_restaurants

906K restaurant reviews in the Las Vegas area. It contains information such as, date of rating, user who added it, 
stars (numeric rating), text rating, etc.

### 3) users_restaurant_reviews_Las_Vegas

337K users who have reviewed restaurants in Las Vegas. It contains information such as each user's id, all the ratings 
he has given, his friends (other users of the platform) and many more features.


## Preprocessing steps

Because of its substantial size we had to sample a manageable portion of it. We decided to focus only on mexican 
restaurants that are located in the Las Vegas area (restaurants that have the categories: Mexican attribute: value 
combo in Yelp_restaurant_Las_Vegas.json).

To filter out low-quality ratings, we decided to ignore users who had zero activity on the platform.
To achieve this, we discarded users that has the value 0 in compliment type fields like funny, useful, cool, fans 
and in a field that we defined defined as the number of friends a user has (via the friends feature).

These restrictions reduced the number of restaurants from 5899 to 550. Correspondingly, a total of 337K users were 
also reduced to 44K users. These remaining users were relatively active (had given a rating to a Mexican restaurant).
The number of ratings also dropped from 906K to 70K.

To take advantage of the features of all three files, a join operation was applied to all three files. Because joins 
through pymongo created nested fields, we decided to use the join function offered by pandas dataframes. 
We also used pandas to apply the various filters mentioned above.


## Useful statistics

From the ratings distribution in our reviews dataset we observe that there is a significant imbalance between the
absolute values (1-5). The 5-star rating class has over 26k instances while the 2-star rating class comes last with only
6k instances. The absolute values for all 5 classes are:

* 1-star: 7841 instances
* 2-star: 6005 instances
* 3-star: 8225 instances
* 4-star: 15042 instances
* 5-star: 26062 instances

<p align="center">
    <img src="https://github.com/diliadis/yelp_sentiment_analysis/blob/master/plots/stars_distribution.png">
</p>

To mitigate the negative effects of class imbalance, we randomly sampled the 5 classes and ended up with a total of
4k instances.

After that, we produced the plots for the distributions of reviews that were characterized as funny, useful and cool.
In all three plots that are presented below we could see that they follow a power-law distribution. Also, the majority of
reviews for the mexican restaurants in the Las Vegas area are not characterized as funny, useful or cool

<p align="center">
    <img src="https://github.com/diliadis/yelp_sentiment_analysis/blob/master/plots/Distribution_of_useful_reviews.png">
</p>

<p align="center">
    <img src="https://github.com/diliadis/yelp_sentiment_analysis/blob/master/plots/Distribution_of_funny_reviews.png">
</p>

<p align="center">
    <img src="https://github.com/diliadis/yelp_sentiment_analysis/blob/master/plots/Distribution_of_cool_reviews.png">
</p>


We also created several wordclouds. The five different word clouds correspond to the 5 different star rating classes.

1-star rating wordcloud
<p align="center">
    <img src="https://github.com/diliadis/yelp_sentiment_analysis/blob/master/word_clouds/rated_1_text.png">
</p>

2-star rating wordcloud
<p align="center">
    <img src="https://github.com/diliadis/yelp_sentiment_analysis/blob/master/word_clouds/rated_2_text.png">
</p>

3-star rating wordcloud
<p align="center">
    <img src="https://github.com/diliadis/yelp_sentiment_analysis/blob/master/word_clouds/rated_3_text.png">
</p>

4-star rating wordcloud
<p align="center">
    <img src="https://github.com/diliadis/yelp_sentiment_analysis/blob/master/word_clouds/rated_4_text.png">
</p>

5-star rating wordcloud
<p align="center">
    <img src="https://github.com/diliadis/yelp_sentiment_analysis/blob/master/word_clouds/rated_5_text.png">
</p>
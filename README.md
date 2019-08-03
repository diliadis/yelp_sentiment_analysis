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


## Text preprocessing and extraction of sentiment info

There are two main approaches used to extract information about the sentiment in text. The first approach uses scores 
about the individual words and the second approach uses the characteristics and structure of the text. The dictionary-based
approach involves calculating the semantic orientation of words or phrases in the document using word dictionaries for 
which the semantic score of emotional information has already been determined. Such dictionaries can be created manually
or even automatically, using seed words to extend the list of words. The total score of a sentence is essentially the 
sum of scores of the individual words that make up the sentence.

On the other hand, the approach of classification using the text's characteristics, is based on machine learning models
that are trained on datasets with known scores. The most common technique used in the literature, are SVMs that are trained
in datasets comprised of features that are produced from unigrams, bigrams, word frequencies, tfidf vectors etc. Methods 
that use supervised learning can detect the polarity of text with very high accuracies.

There are also hybrid methods, that use combinations of the aforementioned approaches to achieve even better results.

### Text preprocessing

For this step we relied heavily in the [NLTK library](https://www.nltk.org/). This library provides tools for processing 
language data with capabilities for classification, tokenization, stemming, tagging, parsing and semantic reasoning.

Initially, we use the word_tokenize function to split each review into a list it's words. Then, each word in the list is
converted to lowecase letters, and words that appear frequently in English are removed from it. Finnaly, we use a Porter
Stemmer and a word Lemmatizer to reduce the total number of distinct words 

### Extraction of sentiment info

The extraction of emotional information from texts can be achieved in a variety of methods that have been reported in the 
literature. We chose to use the bag of words model with tfidfs and n-grams, as well as dictionaries such as SentiWordNet 
and pre-trained dictionaries.

## Classification

The purpose of this study was to investigatye which approach can yield better results in the extraction of sentiment 
information in order to predict the star rating of reviews. We experimented with the dictinary-based approach, the
text characteristics extraction approach, as well as various combinations of them. Results from the literature suggested
that the hybrid method yields the best results and that the unigrams are the strongest features.

The most commonly used classification algorithm that yields the best results is the SVM and, although various tests were
performed with other algorithms (AdaBoost, Logistic Regression) it was confirmed that SVMs give the best results. Therefore,
we used that classifier in all our experiments to find the best feature combination. Finally, we used the best classifier 
and feature combination to train binary problems (class 1 as negative and class 5 as positive) as well as multi-class
problems (class 3 as neutral to form a total of 3 classes or all 5 classes to form a 5 class multi-class problem).


### Feature selection

The features used for training can be divided into 3 categories:

* A) Text features: tfidf, bigrams, trigrams.
* B) Dictionary features: scores from SentiWordNet, TextBlob and Vader.
* C) Features provided from yelp for the Reviews, Users and Restaurants.

To select the best attributes available to the three files (Reviews, Users and Restaurants), a Chi-squared test was applied
(visualization in a heatmap presented below). The attributes that we eventually used were the averate rating of each user,
the average rating of each restaurant and the number of reviews of each restaurant.

<p align="center">
    <img src="https://github.com/diliadis/yelp_sentiment_analysis/blob/master/plots/heat_map_stars_users.png">
</p>

<p align="center">
    <img src="https://github.com/diliadis/yelp_sentiment_analysis/blob/master/plots/heat_map_stars_business.png">
</p>

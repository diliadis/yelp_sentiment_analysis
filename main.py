import pymongo
import pandas as pd
import ml_yelp
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

def main():
    # Load mongodb collection
    client = pymongo.MongoClient('localhost', 27017)
    db = client['yelp_db']
    collection_name = 'users_reviews_restaurants_join'
    collection = db[collection_name]
    df = pd.DataFrame(list(collection.find()))

    df = ml_yelp.equal_class_sampling(df, sample_size_per_class=4000)

    ada = AdaBoostClassifier(DecisionTreeClassifier(max_depth=4), n_estimators=200, learning_rate=1.5, algorithm="SAMME", random_state=10)
    weight = {1: 2., 2: 3., 3: 4., 4: 3., 5: 2.}
    svm = LinearSVC(class_weight=weight, C=0.01)
    lr = LogisticRegression(multi_class='multinomial', solver='lbfgs', random_state=10)
    mlp = MLPClassifier(activation='relu', learning_rate='constant', random_state=10)
    pa = PassiveAggressiveClassifier(random_state=10, class_weight=weight)
    # only with tfidf and bigrams
    nb = MultinomialNB(class_prior=[0.25, 0.25, 0.25, 0.25, 0.25])

    ml_yelp.run_models(df, base_classifier=svm, chi2_filter_k=500, special_features_list=['polarity', 'compound',
                      'swnscore', 'business_average_stars', 'user_average_stars'], train_mode='standard')













from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection
from sklearn.preprocessing import MinMaxScaler
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.feature_selection import chi2, SelectKBest
import ml_yelp
import text_yelp
import scipy as sp
import pandas as pd
import numpy as np
import os
import itertools
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def identity_tokenizer(text):
    return text


def n_grams(col, l1, l2):

    cv = CountVectorizer(ngram_range=(l1, l2), tokenizer=identity_tokenizer, preprocessor=identity_tokenizer, token_pattern=r'\b\w+\b', min_df=1)
    ngrams = cv.fit_transform(col)
    return ngrams


def tfidf(col):
    vect = TfidfVectorizer(analyzer='word', tokenizer=identity_tokenizer, preprocessor=identity_tokenizer, token_pattern=None)
    tfidf = vect.fit_transform(col)
    return tfidf


def train_clf(clf, X, y, score):
    scores = model_selection.cross_val_score(clf, X, y, cv=10, scoring=score)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()))


def equal_class_sampling(df, sample_size_per_class=4000):
    # Sample the dataset so that every class will have an equal number of instances
    # check if the user's sample size is bigger than the actual size of the minority class
    smallest_class_size = min(df.groupby('stars')['stars'].count().tolist())
    if smallest_class_size < sample_size_per_class:
        sample_size_per_class = smallest_class_size

    stars1 = df.loc[df['stars'] == 1].sample(n=sample_size_per_class, replace = False, axis =0)
    stars2 = df.loc[df['stars'] == 2].sample(n=sample_size_per_class, replace = False, axis =0)
    stars3 = df.loc[df['stars'] == 3].sample(n=sample_size_per_class, replace = False, axis =0)
    stars4 = df.loc[df['stars'] == 4].sample(n=sample_size_per_class, replace = False, axis =0)
    stars5 = df.loc[df['stars'] == 5].sample(n=sample_size_per_class, replace = False, axis =0)

    if os.path.isfile('sampled_df.pkl'):
        df = pd.read_pickle('sampled_df.pkl')
    else:
        df = pd.concat([stars1, stars2, stars3, stars4, stars5])
        df.to_pickle('sampled_df.pkl')

    # Text Features Extraction
    if os.path.isfile('preproc_df.pkl'):
        df = pd.read_pickle('preproc_df.pkl')
    else:
        text_yelp.text_processing(df, text_col_name='text')
        text_yelp.get_textblob_score(df, text_col_name='text')
        text_yelp.get_vader_score(df, text_col_name='text')
        text_yelp.get_senti_word_net_score(df, text_col_name='text')
        df.to_pickle('preproc_df.pkl')


def run_models(df, base_classifier, chi2_filter_k=500, special_features_list=['polarity','compound', 'swnscore', 'business_average_stars', 'user_average_stars'], train_mode='standard'):
    # Features extraction
    # compute tfidf
    tfidf = ml_yelp.tfidf(df['sswords'])
    # compute bigrams and trigrams
    bitrigrams = ml_yelp.n_grams(df['text'], 2, 3)
    # compute chi2 for each feature - test how closely each feature is correlated with it's class
    tfidf_new = SelectKBest(chi2, k=chi2_filter_k).fit_transform(tfidf, df['stars'])
    bitri_new = SelectKBest(chi2, k=chi2_filter_k).fit_transform(bitrigrams, df['stars'])

    #Data scaling
    scaler = MinMaxScaler(feature_range=(1,5))
    df[['polarity', 'compound', 'neg', 'neu', 'pos', 'swnscore']] = scaler.fit_transform(df[special_features_list])

    #Combine different types of features
    X = sp.sparse.hstack((bitri_new, tfidf_new, df[['polarity','compound', 'swnscore', 'business_average_stars', 'user_average_stars']].values),format='csr') #textblob, vader, SentiwordNet
    # X2 = sp.sparse.hstack((bitri_new, tfidf_new, df[['polarity','swnscore']].values),format='csr') #textblob, SentiwordNet
    # X3 = sp.sparse.hstack((bitri_new, tfidf_new, df[['polarity','neg', 'neu', 'pos']].values),format='csr') #textblob, vader
    # X4 = sp.sparse.hstack((bitri_new, tfidf_new, df[['neg', 'neu', 'pos', 'swnscore']].values),format='csr')  #vader, SentiwordNet
    # X5 = sp.sparse.hstack((bitri_new, tfidf_new, df[['neg', 'neu', 'pos']].values),format='csr')  #vader
    # X6 = sp.sparse.hstack((bitri_new, tfidf_new, df[['swnscore']].values),format='csr')  #SentiwordNet
    # X7 = sp.sparse.hstack((bitri_new, tfidf_new, df[['polarity']].values),format='csr') #textblob

    # One versus Rest
    ovr = OneVsRestClassifier(base_classifier)

    # One versus One
    ovo = OneVsOneClassifier(base_classifier)

    print("Training Classifiers....")
    if train_mode == 'standard':
        ml_yelp.train_clf_split(base_classifier, X, df['stars'])
    elif train_mode == 'cross_val':
        ml_yelp.train_clf(base_classifier, X, df['stars'], 'accuracy')


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')




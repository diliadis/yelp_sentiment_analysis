import pandas as pd
from nltk import pos_tag, sent_tokenize
from nltk.corpus import sentiwordnet as swn
from nltk.corpus import wordnet as wn
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk import FreqDist
import matplotlib
matplotlib.use('TkAgg')
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from pandas.plotting import scatter_matrix
from textblob import TextBlob


# compute polarity score based on Textblob method
def get_textblob_score(df, text_col_name='text'):
    text_col = df[text_col_name]

    sentiment = text_col.apply(lambda x: TextBlob(x).sentiment)
    sentiment_series = sentiment.tolist()
    df['polarity'] = [sentiment_score.polarity for sentiment_score in sentiment_series]
    df['subjectivity'] = [sentiment_score.subjectivity for sentiment_score in sentiment_series]

    '''
    g = sns.FacetGrid(data=df, col='stars')
    g.map(plt.hist, 'polarity', bins=50)
    plt.show()
    g = sns.FacetGrid(data=df, col='stars')
    g.map(plt.hist, 'subjectivity', bins=50)
    plt.show()
    '''

# compute sentiment score based on Vader method
def get_vader_score(df, text_col_name='text'):
    text_col = df[text_col_name]

    sid = SentimentIntensityAnalyzer()
    vader_scores = text_col.apply(lambda x: sid.polarity_scores(x))
    vader_scores_series = vader_scores.tolist()
    df['neg'] = [vader_score['neg'] for vader_score in vader_scores_series]
    df['neu'] = [vader_score['neu'] for vader_score in vader_scores_series]
    df['pos'] = [vader_score['pos'] for vader_score in vader_scores_series]
    df['compound'] = [vader_score['compound'] for vader_score in vader_scores_series]

    '''
    g = sns.FacetGrid(data=df, col='stars')
    g.map(plt.hist, 'neg', bins=50)
    plt.show()
    g = sns.FacetGrid(data=df, col='stars')
    g.map(plt.hist, 'neu', bins=50)
    plt.show()
    g = sns.FacetGrid(data=df, col='stars')
    g.map(plt.hist, 'pos', bins=50)
    plt.show()
    g = sns.FacetGrid(data=df, col='stars')
    g.map(plt.hist, 'compound', bins=50)
    plt.show()
    '''


# Convert between the PennTreebank tags to simple Wordnet tags
def penn_to_wn(tag):

    if tag.startswith('J'):
        return wn.ADJ
    elif tag.startswith('N'):
        return wn.NOUN
    elif tag.startswith('R'):
        return wn.ADV
    elif tag.startswith('V'):
        return wn.VERB
    return None


# compute sentiment score based on SentiwordNet for the entire dataframe
def get_senti_word_net_score(df, text_col_name='text'):
    text_col = df[text_col_name]
    df['swnscore'] = text_col.apply(lambda x: sentiment_swn(x))


# compute sentiment score based on SentiwordNet for individual sentence
def sentiment_swn(text_col):

    lemmatizer = WordNetLemmatizer()
    sentiment = 0.0
    text = text_col.replace("<br />", " ")
    raw_sentences = sent_tokenize(text)
    for raw_sentence in raw_sentences:
        tagged_sentence = pos_tag(word_tokenize(raw_sentence))
        for word, tag in tagged_sentence:
            wn_tag = penn_to_wn(tag)
            if wn_tag not in (wn.NOUN, wn.ADJ, wn.ADV):
                continue
            lemma = lemmatizer.lemmatize(word, pos=wn_tag)
            if not lemma:
                continue
            synsets = wn.synsets(lemma, pos=wn_tag)
            if not synsets:
                continue
            # Take the first sense, the most common
            synset = synsets[0]
            swn_synset = swn.senti_synset(synset.name())
            sentiment += swn_synset.pos_score() - swn_synset.neg_score()
    return sentiment


def word_freq(text_col, n = 30):
    text = ' '.join(text_col)
    words_list = word_tokenize(text)
    word_dist = FreqDist(words_list)
    return word_dist.most_common(n)


# creates the wordcloud plots for the given text
def word_cloud(text_col,file_name):
    text = ' '.join(text_col)
    cloud = WordCloud(max_font_size=15, min_font_size=8, background_color='white', max_words=40).generate(text)
    plt.figure()
    plt.imshow(cloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()
    plt.savefig(file_name+'.png')


def text_processing(df, text_col_name='text'):
    text_col = df[text_col_name]

    # remove punctuations
    df[text_col_name] = df[text_col_name].str.replace('[^\w\s]','')
    # Tokenization
    df['words'] = text_col.apply(word_tokenize)
    #to lowercase
    df['words'] = df['words'].apply(lambda x: [token.lower() for token in x])
    # Remove stopwords
    stop_words = stopwords.words('english')
    df['swords'] = df['words'].apply(lambda x: [item for item in x if item not in stop_words])
    df['joined_swords'] = df['swords'].apply(lambda x: " ".join(x))
    # Stemming
    ps = PorterStemmer()
    df['sswords'] = df['swords'].apply(lambda x: [ps.stem(y) for y in x])
    df['joined_sswords'] = df['sswords'].apply(lambda x: " ".join(x))
    # Lemmatization
    wl = WordNetLemmatizer()
    df['lswords'] = df['swords'].apply(lambda x: [wl.lemmatize(y) for y in x])
    df['joined_lswords'] = df['lswords'].apply(lambda x: " ".join(x))
    # Lemmatization on Stemmed words
    wl = WordNetLemmatizer()
    df['lsswords'] = df['sswords'].apply(lambda x: [wl.lemmatize(y) for y in x])
    df['joined_lsswords'] = df['lsswords'].apply(lambda x: " ".join(x))


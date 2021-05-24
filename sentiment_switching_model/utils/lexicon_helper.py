from nltk.corpus import stopwords
from spacy.lang.en.stop_words import STOP_WORDS as spacy_stopwords
from sentiment_switching_model.config import global_config


def get_sentiment_words():
    with open(file=global_config.sentiment_words_file_path, mode='r', encoding='ISO-8859-1') as sentiment_words_file:
        words = sentiment_words_file.readlines()
    words = set(word.strip() for word in words)
    return words

def get_stopwords():
    nltk_stopwords = set(stopwords.words('english'))
    all_stopwords = set()
    all_stopwords |= spacy_stopwords
    all_stopwords |= nltk_stopwords
    return all_stopwords

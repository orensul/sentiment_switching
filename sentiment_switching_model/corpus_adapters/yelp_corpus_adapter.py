import re

from sentiment_switching_model.config import global_config
from sentiment_switching_model.utils import log_initializer

logger = log_initializer.setup_custom_logger(global_config.logger_name, "INFO")

dev_pos_reviews_file_path = "sentiment_switching_model/corpus_adapters/data/yelp/sentiment.dev.1"
dev_neg_reviews_file_path = "sentiment_switching_model/corpus_adapters/data/yelp/sentiment.dev.0"
test_pos_reviews_file_path = "sentiment_switching_model/corpus_adapters/data/yelp/sentiment.test.1"
test_neg_reviews_file_path = "sentiment_switching_model/corpus_adapters/data/yelp/sentiment.test.0"
train_pos_reviews_file_path = "sentiment_switching_model/corpus_adapters/data/yelp/sentiment.train.1"
train_neg_reviews_file_path = "sentiment_switching_model/corpus_adapters/data/yelp/sentiment.train.0"

train_text_file_path = "sentiment_switching_model/corpus_adapters/data/yelp/reviews-train.txt"
train_labels_file_path = "sentiment_switching_model/corpus_adapters/data/yelp/sentiment-train.txt"
val_text_file_path = "sentiment_switching_model/corpus_adapters/data/yelp/reviews-val.txt"
val_labels_file_path = "sentiment_switching_model/corpus_adapters/data/yelp/sentiment-val.txt"
test_text_file_path = "sentiment_switching_model/corpus_adapters/data/yelp/reviews-test.txt"
test_labels_file_path = "sentiment_switching_model/corpus_adapters/data/yelp/sentiment-test.txt"


def clean_text(string):
    string = re.sub(r"\.", "", string)
    string = re.sub(r"\\n", " ", string)
    string = re.sub(r"\'m", " am", string)
    string = re.sub(r"\'ve", " have", string)
    string = re.sub(r"n\'t", " not", string)
    string = re.sub(r"\'re", " are", string)
    string = re.sub(r"\'d", " would", string)
    string = re.sub(r"\'ll", " will", string)
    string = re.sub(r'\d+', "number", string)
    string = string.replace("\r", " ")
    string = string.replace("\n", " ")
    string = string.strip().lower()

    return string


def get_file_paths(dataset):
    if dataset == 'validation':
        text_file_path = val_text_file_path
        labels_file_path = val_labels_file_path
        pos_reviews_file_path = dev_pos_reviews_file_path
        neg_reviews_file_path = dev_neg_reviews_file_path
    elif dataset == 'test':
        text_file_path = test_text_file_path
        labels_file_path = test_labels_file_path
        pos_reviews_file_path = test_pos_reviews_file_path
        neg_reviews_file_path = test_neg_reviews_file_path
    elif dataset == 'train':
        text_file_path = train_text_file_path
        labels_file_path = train_labels_file_path
        pos_reviews_file_path = train_pos_reviews_file_path
        neg_reviews_file_path = train_neg_reviews_file_path

    return text_file_path, labels_file_path, pos_reviews_file_path, neg_reviews_file_path


def write_dataset(dataset):
    text_file_path, labels_file_path, pos_reviews_file_path, neg_reviews_file_path = get_file_paths(dataset)
    with open(text_file_path, 'w') as text_file, open(labels_file_path, 'w') as labels_file:
        with open(pos_reviews_file_path, 'r') as reviews_file:
            for line in reviews_file:
                text_file.write(clean_text(line) + '\n')
                labels_file.write("pos" + '\n')
        with open(neg_reviews_file_path, 'r') as reviews_file:
            for line in reviews_file:
                text_file.write(clean_text(line) + '\n')
                labels_file.write("neg" + '\n')

    logger.info("Processing complete")


write_dataset('validation')
logger.info("Writing validation dataset")
write_dataset('test')
logger.info("Writing test dataset")
write_dataset('train')
logger.info("Writing train dataset")
import math
import sys

import argparse
import logging
import tensorflow as tf
from typing import Any

from sentiment_switching_model.config import global_config
from sentiment_switching_model.utils import log_initializer, lexicon_helper

logger = logging.getLogger()


class Options(argparse.Namespace):

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.logging_level = None
        self.text_file_path = None
        self.label_file_path = None


def read_labels(label_file_path):
    labels = set()
    empty_template = dict()

    with open(label_file_path) as label_file:
        for label_line in label_file:
            label = label_line.strip()
            labels.add(label)
    logger.debug(labels)

    for label in labels:
        empty_template[label] = 0
    logger.debug(empty_template)

    return labels, empty_template


def read_vocab(text_file_path):
    text_tokenizer = tf.keras.preprocessing.text.Tokenizer()
    with open(text_file_path) as text_file:
        text_tokenizer.fit_on_texts(text_file)
    vocab = text_tokenizer.word_index.keys()
    del text_tokenizer
    stopwords = lexicon_helper.get_stopwords()
    vocab -= stopwords
    logger.debug(vocab)
    return vocab


def set_word_label_occurrences(text_file_path, label_file_path, vocab, empty_template):
    word_occurrences = dict()
    with open(text_file_path) as text_file, open(label_file_path) as label_file:
        for text_line, label_line in zip(text_file, label_file):
            words = text_line.strip().split()
            label = label_line.strip()
            for word in words:
                if len(word) > 3 and word in vocab:
                    if word not in word_occurrences:
                        word_occurrences[word] = empty_template.copy()
                    label_occurrences = word_occurrences[word]
                    label_occurrences[label] += 1
    logger.debug(word_occurrences)
    return word_occurrences, label_occurrences


def calc_label_word_scores(labels, word_occurrences):
    label_word_scores = dict()
    for label in labels:
        word_scores = list()
        for word in word_occurrences:
            try:
                label_occurrences = word_occurrences[word]
                positive_count = label_occurrences[label]
                negative_count = sum(label_occurrences.values()) - positive_count
                kld = positive_count * (math.log(positive_count) / math.log(negative_count))
                word_scores.append((kld, word))
            except Exception:
                logger.debug("error while processing word '{}' for label '{}'".format(word, label))
        label_word_scores[label] = word_scores

    logger.debug(label_word_scores)
    return label_word_scores


def log_correlated_words_label(label_word_scores):
    for label in label_word_scores:
        word_scores = label_word_scores[label]
        word_scores.sort(key=lambda x: x[0], reverse=True)
        most_correlated = [x[1] for x in word_scores[:100]]
        logger.info("For label '{}'".format(label))
        logger.info("Most correlated words: {}".format(most_correlated))


def build_word_statistics(text_file_path, label_file_path):
    vocab = read_vocab(text_file_path)
    labels, empty_template = read_labels(label_file_path)
    word_occurrences, label_occurrences = set_word_label_occurrences(text_file_path, label_file_path, vocab,
                                                                     empty_template)
    label_word_scores = calc_label_word_scores(labels, word_occurrences)
    log_correlated_words_label(label_word_scores)


def main(argv):
    options = Options()

    parser = argparse.ArgumentParser()
    parser.add_argument("--text-file-path", type=str, required=True)
    parser.add_argument("--label-file-path", type=str, required=True)
    parser.add_argument("--logging-level", type=str, required=True)
    parser.parse_args(args=argv, namespace=options)

    global logger
    logger = log_initializer.setup_custom_logger(global_config.logger_name, options.logging_level)
    build_word_statistics(options.text_file_path, options.label_file_path)


if __name__ == '__main__':
    main(sys.argv[1:])
import sys

import argparse
import json
import numpy as np
import os
import pickle
import tensorflow as tf

from sentiment_switching_model.config import global_config
from sentiment_switching_model.config.model_config import mconf
from sentiment_switching_model.config.options import Options
from sentiment_switching_model.models import adversarial_autoencoder
from sentiment_switching_model.utils import bleu_scorer, data_processor, log_initializer, word_embedder, tf_session_helper

logger = None


def get_word_embeddings(embedding_model_path, word_index):
    encoder_embedding_matrix = np.random.uniform(size=(global_config.vocab_size, global_config.embedding_size),
                                                 low=-0.05, high=0.05).astype(dtype=np.float32)
    logger.debug("encoder_embedding_matrix: {}".format(encoder_embedding_matrix.shape))

    decoder_embedding_matrix = np.random.uniform(size=(global_config.vocab_size, global_config.embedding_size),
                                                 low=-0.05, high=0.05).astype(dtype=np.float32)
    logger.debug("decoder_embedding_matrix: {}".format(decoder_embedding_matrix.shape))

    if embedding_model_path:
        logger.info("Loading pretrained embeddings")
        encoder_embedding_matrix, decoder_embedding_matrix = \
            word_embedder.add_word_vectors_to_embeddings(word_index, encoder_embedding_matrix, decoder_embedding_matrix,
                                                         embedding_model_path)
    return encoder_embedding_matrix, decoder_embedding_matrix


def get_data(options):
    [word_index, padded_sequences, text_sequence_lengths, text_tokenizer, inverse_word_index] = \
        data_processor.get_text_sequences(options.text_file_path, options.vocab_size, global_config.vocab_save_path)
    logger.debug("text_sequence_lengths: {}".format(text_sequence_lengths.shape))
    logger.debug("padded_sequences: {}".format(padded_sequences.shape))

    [one_hot_labels, num_labels] = data_processor.get_labels(options.label_file_path, True, global_config.save_directory)
    logger.debug("one_hot_labels.shape: {}".format(one_hot_labels.shape))

    return [word_index, padded_sequences, text_sequence_lengths, one_hot_labels, num_labels, text_tokenizer,
            inverse_word_index]


def train_save_model(options):
    os.makedirs(global_config.save_directory)
    with open(global_config.model_config_file_path, 'w') as model_config_file:
        json.dump(obj=mconf.__dict__, fp=model_config_file, indent=4)
    logger.info("Saved model config to {}".format(global_config.model_config_file_path))

    # Retrieve all data
    logger.info("Reading data ...")
    [word_index, padded_sequences, text_sequence_lengths, one_hot_labels, num_labels,
     text_tokenizer, inverse_word_index] = get_data(options)
    data_size = padded_sequences.shape[0]

    encoder_embedding_matrix, decoder_embedding_matrix = \
        get_word_embeddings(options.training_embeddings_file_path, word_index)

    # Build model
    logger.info("Building model architecture ...")
    network = adversarial_autoencoder.AdversarialAutoencoder()
    network.build_model(word_index, encoder_embedding_matrix, decoder_embedding_matrix, num_labels)
    logger.info("Training model ...")
    sess = tf_session_helper.get_tensorflow_session()

    [_, validation_actual_word_lists, validation_sequences, validation_sequence_lengths] = \
        data_processor.get_test_sequences(options.validation_text_file_path, text_tokenizer, word_index,
                                          inverse_word_index)
    [_, validation_labels] = data_processor.get_test_labels(options.validation_label_file_path,
                                                            global_config.save_directory)

    network.train(sess, data_size, padded_sequences, text_sequence_lengths, one_hot_labels, num_labels,
                  word_index, encoder_embedding_matrix, decoder_embedding_matrix, validation_sequences,
                  validation_sequence_lengths, validation_labels, inverse_word_index, validation_actual_word_lists,
                  options)
    sess.close()

    logger.info("Training complete!")


def main(argv):
    options = Options()

    parser = argparse.ArgumentParser()
    parser.add_argument("--logging-level", type=str, default="INFO")
    run_mode = parser.add_mutually_exclusive_group(required=True)
    run_mode.add_argument("--train-model", action="store_true", default=False)
    run_mode.add_argument("--transform-text", action="store_true", default=False)
    run_mode.add_argument("--generate-novel-text", action="store_true", default=False)

    parser.parse_known_args(args=argv, namespace=options)

    parser.add_argument("--vocab-size", type=int, default=1000)
    parser.add_argument("--training-epochs", type=int, default=10)
    parser.add_argument("--text-file-path", type=str, required=True)
    parser.add_argument("--label-file-path", type=str, required=True)
    parser.add_argument("--validation-text-file-path", type=str, required=True)
    parser.add_argument("--validation-label-file-path", type=str, required=True)
    parser.add_argument("--training-embeddings-file-path", type=str)
    parser.add_argument("--validation-embeddings-file-path", type=str, required=True)
    parser.add_argument("--dump-embeddings", action="store_true", default=False)
    parser.add_argument("--classifier-saved-model-path", type=str, required=True)
    parser.parse_known_args(args=argv, namespace=options)

    global logger
    logger = log_initializer.setup_custom_logger(global_config.logger_name, options.logging_level)

    global_config.training_epochs = options.training_epochs
    logger.info("experiment_timestamp: {}".format(global_config.experiment_timestamp))

    train_save_model(options)



if __name__ == "__main__":
    main(sys.argv[1:])

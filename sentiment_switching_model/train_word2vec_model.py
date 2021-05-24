import argparse
import sys

from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

from sentiment_switching_model.config import global_config
from sentiment_switching_model.utils import log_initializer

logger = None


def train_word2vec_model(text_file_path, model_file_path):
    logger.info("Loading input file and training mode ...")
    model = Word2Vec(sentences=LineSentence(text_file_path), min_count=1, size=global_config.embedding_size)
    logger.info("Model Details: {}".format(model))
    model.wv.save_word2vec_format(model_file_path, binary=True)
    logger.info("Model saved")


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--text-file-path", type=str, required=True)
    parser.add_argument("--model-file-path", type=str, required=True)
    parser.add_argument("--logging-level", type=str, default="INFO")
    options = vars(parser.parse_args(args=argv))

    global logger
    logger = log_initializer.setup_custom_logger(global_config.logger_name, options['logging_level'])
    train_word2vec_model(options['text_file_path'], options['model_file_path'])
    logger.info("Training Complete!")


if __name__ == "__main__":
    main(sys.argv[1:])

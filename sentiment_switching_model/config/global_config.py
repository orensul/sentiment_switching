from datetime import datetime as dt
logger_name = "sentiment_switching"

embedding_size = 300
max_sequence_length = 15
validation_interval = 1
tsne_sample_limit = 1000

filter_sentiment_words = True
filter_stopwords = True
vocab_size = None  # set by runtime param
bow_size = None  # set by runtime params and exclusions
tokenizer_filters = '!"#$%&()*+,-./:;=?@[\\]^_`{|}~\t\n'

bleu_score_weights = {1: (1.0, 0.0, 0.0, 0.0),
                      2: (0.5, 0.5, 0.0, 0.0),
                      3: (0.34, 0.33, 0.33, 0.0),
                      4: (0.25, 0.25, 0.25, 0.25)}

unk_token = "<unk>"
sos_token = "<sos>"
eos_token = "<eos>"
experiment_timestamp = dt.now().strftime("%Y%m%d%H%M%S")
save_directory = "./saved-models/{}".format(experiment_timestamp)
classifier_save_directory = "./saved-models-classifier/{}".format(experiment_timestamp)
predefined_word_index = {unk_token: 0, sos_token: 1, eos_token: 2}

log_directory = "./tensorflow-logs/{}".format(experiment_timestamp)

all_style_embeddings_path = save_directory + "/all_style_embeddings.npy"
all_content_embeddings_path = save_directory + "/all_content_embeddings.npy"
all_shuffled_labels_path = save_directory + "/all_shuffled_labels_path.pkl"
label_mapped_style_embeddings_path = save_directory + "/label_mapped_style_embeddings.pkl"

tsne_plot_folder = save_directory + "/tsne_plots/"
style_embedding_plot_file = "tsne_embeddings_plot_style_{}.svg"
content_embedding_plot_file = "tsne_embeddings_plot_content_{}.svg"
style_embedding_custom_plot_file = "tsne_embeddings_custom_plot_style.svg"
content_embedding_custom_plot_file = "tsne_embeddings_custom_plot_content.svg"

model_save_file = "sentiment_switching_model.ckpt"
model_save_path = save_directory + "/" + model_save_file

model_config_file = "model_config.json"
model_config_file_path = save_directory + "/" + model_config_file

vocab_save_file = "vocab.json"
vocab_save_path = save_directory + "/" + vocab_save_file
classifier_vocab_save_path = classifier_save_directory + "/" + vocab_save_file


index_to_label_dict_file = "index_to_label_dict.json"
label_to_index_dict_file = "label_to_index_dict.json"
index_to_label_dict_path = save_directory + "/" + index_to_label_dict_file
label_to_index_dict_path = save_directory + "/" + label_to_index_dict_file


style_coordinates_file = "style_coordinates.pkl"
content_coordinates_file = "content_coordinates.pkl"
style_coordinates_path = save_directory + "/" + style_coordinates_file
content_coordinates_path = save_directory + "/" + content_coordinates_file

sentiment_words_file_path = "sentiment_switching_model/corpus_adapters/data/opinion-lexicon/sentiment-words.txt"

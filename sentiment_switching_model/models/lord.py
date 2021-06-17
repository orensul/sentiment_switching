import os
import pdb
import pickle
from collections import namedtuple

import logging
import numpy as np
import tensorflow as tf

from keras.layers import Input

from sentiment_switching_model.config import global_config
from sentiment_switching_model.config.lord_config import lconf

from sentiment_switching_model.config.model_config import mconf

from tensorflow.keras import optimizers, losses, regularizers
from tensorflow.keras.layers import Input, Reshape, Embedding, GaussianNoise
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, Callback


logger = logging.getLogger(global_config.logger_name)



class Lord:

    def build_embedding(self, input_dim, output_dim, name):
        idx = Input(shape=(1,))

        embedding = Embedding(input_dim, output_dim)(idx)
        embedding = Reshape(target_shape=(output_dim,))(embedding)

        model = Model(inputs=idx, outputs=embedding, name='%s-embedding' % name)

        print('%s-embedding:' % name)
        model.summary()

        return model

    def build_regularized_embedding(self, input_dim, output_dim, std, decay, name):
        idx = Input(shape=(1,))

        embedding = Embedding(input_dim, output_dim, activity_regularizer=regularizers.l2(decay))(idx)
        embedding = Reshape(target_shape=(output_dim,))(embedding)

        embedding = GaussianNoise(stddev=std)(embedding)

        model = Model(inputs=idx, outputs=embedding, name='%s-embedding' % name)

        print('%s-embedding:' % name)
        model.summary()

        return model

    def build_model(self, word_index, data_size, encoder_embedding_matrix, decoder_embedding_matrix, num_labels):
        # model inputs
        self.input_sequence = tf.placeholder(
            dtype=tf.int32, shape=[None, global_config.max_sequence_length],
            name="input_sequence")
        logger.debug("input_sequence: {}".format(self.input_sequence))

        batch_size = tf.shape(self.input_sequence)[0]
        logger.debug("batch_size: {}".format(batch_size))

        self.input_label = tf.placeholder(
            dtype=tf.float32, shape=[None, num_labels], name="input_label")
        logger.debug("input_label: {}".format(self.input_label))

        self.sequence_lengths = tf.placeholder(
            dtype=tf.int32, shape=[None], name="sequence_lengths")
        logger.debug("sequence_lengths: {}".format(self.sequence_lengths))

        self.input_bow_representations = tf.placeholder(
            dtype=tf.float32, shape=[None, global_config.bow_size],
            name="input_bow_representations")
        logger.debug("input_bow_representations: {}".format(self.input_bow_representations))

        self.inference_mode = tf.placeholder(dtype=tf.bool, name="inference_mode")
        logger.debug("inference_mode: {}".format(self.inference_mode))

        self.generation_mode = tf.placeholder(dtype=tf.bool, name="generation_mode")
        logger.debug("generation_mode: {}".format(self.generation_mode))

        self.recurrent_state_keep_prob = tf.cond(
            pred=tf.math.logical_or(self.inference_mode, self.generation_mode),
            true_fn=lambda: 1.0,
            false_fn=lambda: mconf.recurrent_state_keep_prob)

        self.fully_connected_keep_prob = tf.cond(
            pred=tf.math.logical_or(self.inference_mode, self.generation_mode),
            true_fn=lambda: 1.0,
            false_fn=lambda: mconf.fully_connected_keep_prob)

        self.sequence_word_keep_prob = tf.cond(
            pred=tf.math.logical_or(self.inference_mode, self.generation_mode),
            true_fn=lambda: 1.0,
            false_fn=lambda: mconf.sequence_word_keep_prob)

        self.conditioning_embedding = tf.placeholder(
            dtype=tf.float32, shape=[None, mconf.style_embedding_size],
            name="conditioning_embedding")
        logger.debug("conditioning_embedding: {}".format(self.conditioning_embedding))

        self.sampled_content_embedding = tf.placeholder(
            dtype=tf.float32, shape=[None, mconf.content_embedding_size],
            name="sampled_content_embedding")
        logger.debug("sampled_content_embedding: {}".format(self.sampled_content_embedding))

        self.epoch = tf.placeholder(dtype=tf.float32, shape=(), name="epoch")
        logger.debug("epoch: {}".format(self.epoch))

        decoder_input = tf.concat(
            values=[tf.fill(dims=[batch_size, 1], value=word_index[global_config.sos_token]),
                    self.input_sequence], axis=1, name="decoder_input")

        with tf.device('/cpu:0'):
            with tf.variable_scope("embeddings", reuse=tf.AUTO_REUSE):
                # word embeddings matrices
                encoder_embeddings = tf.get_variable(
                    initializer=encoder_embedding_matrix, dtype=tf.float32,
                    trainable=True, name="encoder_embeddings")
                logger.debug("encoder_embeddings: {}".format(encoder_embeddings))

                decoder_embeddings = tf.get_variable(
                    initializer=decoder_embedding_matrix, dtype=tf.float32,
                    trainable=True, name="decoder_embeddings")
                logger.debug("decoder_embeddings: {}".format(decoder_embeddings))


                # embedded sequences
                encoder_embedded_sequence = tf.nn.dropout(
                    x=tf.nn.embedding_lookup(params=encoder_embeddings, ids=self.input_sequence),
                    keep_prob=self.sequence_word_keep_prob,
                    name="encoder_embedded_sequence")
                logger.debug("encoder_embedded_sequence: {}".format(encoder_embedded_sequence))

                decoder_embedded_sequence = tf.nn.dropout(
                    x=tf.nn.embedding_lookup(params=decoder_embeddings, ids=decoder_input),
                    keep_prob=self.sequence_word_keep_prob,
                    name="decoder_embedded_sequence")
                logger.debug("decoder_embedded_sequence: {}".format(decoder_embedded_sequence))


        content_embedding = self.build_regularized_embedding(data_size, lconf.content_embedding_size,
                                                             lconf.content_std, lconf.content_decay, name='content')

        style_embedding = self.build_embedding(num_labels, lconf.style_embedding_size, name='style')

        pdb.set_trace()
        sentence_id = Input(shape=(1,))
        style_id = Input(shape=(1,))
        
        self.content_embedding = content_embedding(sentence_id)
        self.style_embedding = style_embedding(style_id)

        # concatenated generative embedding
        generative_embedding = tf.layers.dense(
            inputs=tf.concat(values=[self.style_embedding, self.content_embedding], axis=1),
            units=mconf.decoder_rnn_size, activation=tf.nn.leaky_relu,
            name="generative_embedding")
        logger.debug("generative_embedding: {}".format(generative_embedding))

        # sequence predictions
        with tf.name_scope('sequence_prediction'):
            training_output, self.inference_output, self.final_sequence_lengths = \
                self.generate_output_sequence(
                    decoder_embedded_sequence, generative_embedding, decoder_embeddings,
                    word_index, batch_size)
            logger.debug("training_output: {}".format(training_output))
            logger.debug("inference_output: {}".format(self.inference_output))


    def generate_output_sequence(self, embedded_sequence, generative_embedding,
                                 decoder_embeddings, word_index, batch_size):

        decoder_cell = tf.nn.rnn_cell.DropoutWrapper(
            cell=tf.contrib.rnn.GRUCell(num_units=mconf.decoder_rnn_size),
            input_keep_prob=self.recurrent_state_keep_prob,
            output_keep_prob=self.recurrent_state_keep_prob,
            state_keep_prob=self.recurrent_state_keep_prob)

        projection_layer = tf.layers.Dense(units=global_config.vocab_size, use_bias=False)

        init_state = decoder_cell.zero_state(batch_size=batch_size, dtype=tf.float32)

        training_decoder_scope_name = "training_decoder"
        with tf.name_scope(training_decoder_scope_name):
            training_helper = tf.contrib.seq2seq.TrainingHelper(
                inputs=embedded_sequence,
                sequence_length=self.sequence_lengths)

            training_decoder = custom_decoder.CustomBasicDecoder(
                cell=decoder_cell, helper=training_helper,
                initial_state=init_state,
                latent_vector=generative_embedding,
                output_layer=projection_layer)
            training_decoder.initialize(training_decoder_scope_name)

            training_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(
                decoder=training_decoder, impute_finished=True,
                maximum_iterations=global_config.max_sequence_length,
                scope=training_decoder_scope_name)

        inference_decoder_scope_name = "inference_decoder"
        with tf.name_scope(inference_decoder_scope_name):
            greedy_embedding_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                embedding=decoder_embeddings,
                start_tokens=tf.fill(dims=[batch_size],
                                     value=word_index[global_config.sos_token]),
                end_token=word_index[global_config.eos_token])

            inference_decoder = custom_decoder.CustomBasicDecoder(
                cell=decoder_cell, helper=greedy_embedding_helper,
                initial_state=init_state,
                latent_vector=generative_embedding,
                output_layer=projection_layer)
            inference_decoder.initialize(inference_decoder_scope_name)

            inference_decoder_output, _, final_sequence_lengths = \
                tf.contrib.seq2seq.dynamic_decode(
                    decoder=inference_decoder, impute_finished=True,
                    maximum_iterations=global_config.max_sequence_length,
                    scope=inference_decoder_scope_name)

        return [training_decoder_output.rnn_output, inference_decoder_output.sample_id, final_sequence_lengths]


    # def train(self, sentences, classes, batch_size, n_epochs):
    #     sentence_id = Input(shape=(1, ))
    #     class_id = Input(shape=(1, ))
    #
    #     content_code = self.content_embedding(sentence_id)
    #     class_code = self.class_embedding(class_id)
    #
    #     generated_sentence = self.generator([content_code, class_code, sentence_shape])
    #
    #     model = Model(inputs=[sentence_id, class_id], outputs=generated_sentence)
    #
    #     model.compile(
    #         optimizer=LRMultiplier(
    #             optimizer=optimizers.Adam(beta_1=0.5, beta_2=0.999),
    #             multipliers={
    #                 'content-embedding': 10,
    #                 'class-embedding': 10
    #             }
    #         ),
    #         # what is the loss
    #         loss='sparse_categorical_crossentropy',
    #         # loss=self.__perceptual_loss_multiscale
    #     )
    #
    #     lr_scheduler = CosineLearningRateScheduler(max_lr=3e-4, min_lr=1e-5, total_epochs=n_epochs)
    #     early_stopping = EarlyStopping(monitor='loss', mode='min', min_delta=1, patience=100, verbose=1)
    #
    #     model.fit(
    #         x=[np.arange(sentences.shape[0]), classes], y=sentences, batch_size=batch_size, epochs=n_epochs,
    #         callbacks=[lr_scheduler, early_stopping], verbose=1
    #     )

   
    
    

  


# class CosineLearningRateScheduler(Callback):
#
#     def __init__(self, max_lr, min_lr, total_epochs):
#         super().__init__()
#
#         self.max_lr = max_lr
#         self.min_lr = min_lr
#         self.total_epochs = total_epochs
#
#     def on_train_begin(self, logs=None):
#         K.set_value(self.model.optimizer.lr, self.max_lr)
#
#     def on_epoch_end(self, epoch, logs=None):
#         fraction = epoch / self.total_epochs
#         lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1 + np.cos(fraction * np.pi))
#
#         K.set_value(self.model.optimizer.lr, lr)
#         logs['lr'] = lr

import os
import pickle
from collections import namedtuple

import numpy as np
import tensorflow as tf

from keras import backend as K
from keras import optimizers, losses, regularizers
from keras.layers import Conv2D, Dense, UpSampling2D, LeakyReLU, Activation
from keras.layers import Layer, Input, Reshape, Lambda, Flatten, Concatenate, Embedding, GaussianNoise
from keras.models import Model, load_model
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, Callback
from keras_lr_multiplier import LRMultiplier


Config = namedtuple(
    typename='Config',
    field_names=['sentence_shape',
                 'n_sentences',
                 'n_classes',
                 'content_dim',
                 'class_dim',
                 'content_std',
                 'content_decay']
)



class Lord:

    @classmethod
    def build(cls, config):
        content_embedding = cls.__build_regularized_embedding(config.n_sentences, config.content_dim,
                                                              config.content_std, config.content_decay, name='content')

        class_embedding = cls.__build_embedding(config.n_classes, config.class_dim, name='class')

        # class_modulation = cls.__build_class_modulation(config.class_dim, config.n_adain_layers, config.adain_dim)

        generator = cls.__build_generator(config.content_dim)

        return Lord(config, content_embedding, class_embedding, generator)

    def __init__(self, config,
                 content_embedding, class_embedding, generator, content_encoder=None, class_encoder=None):
        self.config = config
        self.content_embedding = content_embedding
        self.class_embedding = class_embedding
        # self.class_modulation = class_modulation
        self.generator = generator
        self.content_encoder = content_encoder
        self.class_encoder = class_encoder

        # self.vgg = self.__build_vgg()

    def train(self, sentences, classes, batch_size, n_epochs):
        sentence_id = Input(shape=(1, ))
        class_id = Input(shape=(1, ))

        content_code = self.content_embedding(sentence_id)
        class_code = self.class_embedding(class_id)

        # class_adain_params = self.class_modulation(class_code)

        generated_sentence = self.generator([content_code, # class_adain_params
                                             ])

        model = Model(inputs=[sentence_id, class_id], outputs=generated_sentence)

        model.compile(
            optimizer=LRMultiplier(
                optimizer=optimizers.Adam(beta_1=0.5, beta_2=0.999),
                multipliers={
                    'content-embedding': 10,
                    'class-embedding': 10
                }
            ),
            # what is the loss
            loss='sparse_categorical_crossentropy',
            # loss=self.__perceptual_loss_multiscale
        )

        lr_scheduler = CosineLearningRateScheduler(max_lr=3e-4, min_lr=1e-5, total_epochs=n_epochs)
        early_stopping = EarlyStopping(monitor='loss', mode='min', min_delta=1, patience=100, verbose=1)

        model.fit(
            x=[np.arange(sentences.shape[0]), classes], y=sentences, batch_size=batch_size, epochs=n_epochs,
            callbacks=[lr_scheduler, early_stopping], verbose=1
        )

    @classmethod
    def __build_regularized_embedding(cls, input_dim, output_dim, std, decay, name):
        idx = Input(shape=(1,))

        embedding = Embedding(input_dim, output_dim, activity_regularizer=regularizers.l2(decay))(idx)
        embedding = Reshape(target_shape=(output_dim,))(embedding)

        embedding = GaussianNoise(stddev=std)(embedding)

        model = Model(inputs=idx, outputs=embedding, name='%s-embedding' % name)

        print('%s-embedding:' % name)
        model.summary()

        return model

    @classmethod
    def __build_embedding(cls, input_dim, output_dim, name):
        idx = Input(shape=(1,))

        embedding = Embedding(input_dim, output_dim)(idx)
        embedding = Reshape(target_shape=(output_dim,))(embedding)

        model = Model(inputs=idx, outputs=embedding, name='%s-embedding' % name)

        print('%s-embedding:' % name)
        model.summary()

        return model
    
    

    @classmethod
    def __build_generator(cls, content_dim, sentence_shape):
        pass

  


class CosineLearningRateScheduler(Callback):

    def __init__(self, max_lr, min_lr, total_epochs):
        super().__init__()

        self.max_lr = max_lr
        self.min_lr = min_lr
        self.total_epochs = total_epochs

    def on_train_begin(self, logs=None):
        K.set_value(self.model.optimizer.lr, self.max_lr)

    def on_epoch_end(self, epoch, logs=None):
        fraction = epoch / self.total_epochs
        lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1 + np.cos(fraction * np.pi))

        K.set_value(self.model.optimizer.lr, lr)
        logs['lr'] = lr

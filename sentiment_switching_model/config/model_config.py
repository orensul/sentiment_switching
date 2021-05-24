class ModelConfig():
    def __init__(self):
        # batch settings
        self.batch_size = 128

        # layer sizes
        self.encoder_rnn_size = 256
        self.decoder_rnn_size = 256
        self.style_embedding_size = 8
        self.content_embedding_size = 128



mconf = ModelConfig()
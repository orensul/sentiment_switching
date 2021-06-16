class LordConfig():
	def __init__(self):
		
		self.style_embedding_size = 8
		self.content_embedding_size = 128

		self.train_batch_size = 64
		self.train_n_epochs = 100

		self.train_encoders_batch_size = 64
		self.train_encoders_n_epochs = 20

		self.content_std = 1
		self.content_decay = 0.001

	def init_from_dict(self, previous_config):
		for key in previous_config:
			setattr(self, key, previous_config[key])


lconf = LordConfig()



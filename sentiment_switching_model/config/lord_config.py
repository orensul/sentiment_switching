default_config = dict(
	content_dim=128,
	class_dim=8,

	content_std=1,
	content_decay=0.001,

	train=dict(
		batch_size=64,
		n_epochs=100
	),

	train_encoders=dict(
		batch_size=64,
		n_epochs=20
	)
)
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.regularizers import l2
from tensorflow.keras.saving import register_keras_serializable

@register_keras_serializable(package='Custom')
class ArcFace(layers.Layer):
	def __init__(self, num_classes, regularizer=None, **kwargs):
		super(ArcFace, self).__init__(**kwargs)
		self.num_classes = num_classes
		self.regularizer = regularizer
	
	def build(self, input_shape):
		self.w = self.add_weight(
			shape=(input_shape[-1], self.num_classes),
			initializer='glorot_uniform',
			trainable=True,
			regularizer=self.regularizer,
			name='arcface_weights'
		)
	
	def call(self, inputs):
		# Normalize features and weights
		x = tf.nn.l2_normalize(inputs, axis=1)
		w = tf.nn.l2_normalize(self.w, axis=0)
		
		# Calculate cosine similarity
		cos_sim = tf.matmul(x, w)
		
		return cos_sim
	
	def get_config(self):
		config = super().get_config()
		config.update({
			'num_classes': self.num_classes,
			'regularizer': tf.keras.regularizers.serialize(self.regularizer)
		})
		return config

@register_keras_serializable(package='Custom')
def l2_distance(vectors):
	x, y = vectors
	return tf.sqrt(tf.reduce_sum(tf.square(x - y), axis=-1, keepdims=True))

class FaceRecognitionModel:
	def __init__(self, num_classes, input_shape=(160, 160, 3)):
		self.num_classes = num_classes
		self.input_shape = input_shape
	def build_base_model(self):
		base_model = ResNet50(
			input_shape=self.input_shape,
			include_top=False,
			weights='imagenet'
		)
		# Unfreeze last 10 layers
		base_model.trainable = True
		for layer in base_model.layers[:-10]:
			layer.trainable = False
		return base_model
	
 
	def build_embedding_model(self):
		base_model = self.build_base_model()
		inputs = tf.keras.Input(shape=self.input_shape)
		x = base_model(inputs, training=False)
		x = layers.GlobalAveragePooling2D()(x)
		x = layers.Dense(1024, activation='relu')(x)
		x = layers.BatchNormalization()(x)
		x = layers.Dropout(0.4)(x)
		x = layers.Dense(512, activation='relu')(x)
		outputs = layers.Dense(256)(x)
		return Model(inputs, outputs)
	def build_classification_model(self):
		"""Build the classification model with ArcFace layer"""
		embedding_model = self.build_embedding_model()
		
		inputs = tf.keras.Input(shape=self.input_shape)
		embeddings = embedding_model(inputs)
		outputs = ArcFace(self.num_classes, regularizer=l2(1e-4))(embeddings)
		
		return Model(inputs, outputs, name='face_recognition_model')
	
	def build_siamese_model(self):
		"""Build a siamese model for verification tasks"""
		embedding_model = self.build_embedding_model()
		
		input_1 = layers.Input(shape=self.input_shape)
		input_2 = layers.Input(shape=self.input_shape)
		
		embedding_1 = embedding_model(input_1)
		embedding_2 = embedding_model(input_2)
		
		# Use the registered function
		distance = layers.Lambda(l2_distance, name='l2_distance')([embedding_1, embedding_2])
		
		# Output is 1 if same person, 0 if different
		output = layers.Dense(1, activation='sigmoid')(distance)
		
		return Model(inputs=[input_1, input_2], outputs=output, name='siamese_model')

	
if __name__ == '__main__':
	# Example usage
	model = FaceRecognitionModel(num_classes=10)
	classification_model = model.build_classification_model()
	classification_model.summary()
	
	siamese_model = model.build_siamese_model()
	siamese_model.summary()
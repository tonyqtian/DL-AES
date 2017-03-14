import numpy as np
import logging

import keras.backend as K
from keras.layers.embeddings import Embedding
from keras.models import Model
from keras.layers.core import Activation, Dense, Dropout
from nea.my_layers import MeanOverTime, Conv1DWithMasking
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2, activity_l2
from keras.engine.topology import Input, merge
from keras.layers.wrappers import Bidirectional

logger = logging.getLogger(__name__)

def create_model(args, initial_mean_value, overal_maxlen, vocab):
	
	###############################################################################################################################
	## Recurrence unit type
	#

	if args.recurrent_unit == 'lstm':
		from keras.layers.recurrent import LSTM as RNN
	elif args.recurrent_unit == 'gru':
		from keras.layers.recurrent import GRU as RNN
	elif args.recurrent_unit == 'simple':
		from keras.layers.recurrent import SimpleRNN as RNN

	###############################################################################################################################
	## Create Model
	#
	
	if args.dropout_w > 0:
		dropout_W = args.dropout_w
	else:
		dropout_W = args.dropout_prob		# default=0.5
	if args.dropout_u > 0:
		dropout_U = args.dropout_u
	else:
		dropout_U = args.dropout_prob		# default=0.1
	
	cnn_border_mode='same'
	
	if args.model_type == 'reg':
		if initial_mean_value.ndim == 0:
			initial_mean_value = np.expand_dims(initial_mean_value, axis=1)
		num_outputs = len(initial_mean_value)
	else:
		num_outputs = initial_mean_value

	###############################################################################################################################
	## Initialize embeddings if requested
	#
		
	if args.emb_path:
		def my_init(shape, name=None):
			from nea.w2vEmbReader import W2VEmbReader as EmbReader
			logger.info('Initializing lookup table')
			emb_reader = EmbReader(args.emb_path, emb_dim=args.emb_dim)
			emb_matrix = np.random.random(shape)
# 			logger.info(' initial matrix \n %s ' % (emb_matrix,))
			emb_matrix = emb_reader.get_emb_matrix_given_vocab(vocab, emb_matrix)
# 			from keras.backend import set_value, get_value
# 			set_value(model.layers[model.emb_index].W, get_value(emb_reader.get_emb_matrix_given_vocab(vocab, model.layers[model.emb_index].W)))
# 			model.layers[model.emb_index].W.set_value(emb_reader.get_emb_matrix_given_vocab(vocab, model.layers[model.emb_index].W.get_value()))
# 			logger.info(' pre-trained matrix \n %s ' % (emb_matrix,))
			return K.variable(emb_matrix, name=name)
		logger.info(' Use pre-trained embedding')
	else:
		my_init = 'uniform'
		logger.info(' Use default initializing embedding')
	
	###############################################################################################################################
	## Model Stacking
	#
	
	if args.model_type == 'cls':
		logger.info('Building a CLASSIFICATION model with POOLING')
		dense_activation = 'tanh'
		dense_init = 'glorot_normal'
		if args.loss == 'cnp':
			final_activation = 'softmax'
			final_init = 'glorot_uniform'
		elif args.loss == 'hng':
			final_activation = 'linear'
			final_init = 'glorot_uniform'
	elif args.model_type == 'reg':
		logger.info('Building a REGRESSION model with POOLING')
		dense_activation = 'tanh'
		dense_init = 'he_normal'
		if args.normalize:
			final_activation = 'sigmoid'
			final_init = 'he_normal'
		else:
			final_activation = 'relu'
			final_init = 'he_uniform'
	else:
		raise NotImplementedError
	
	sequence = Input(shape=(overal_maxlen,), dtype='int32')
	x = Embedding(len(vocab), args.emb_dim, mask_zero=True, init=my_init, trainable=args.embd_train)(sequence)
	
	# Conv Layer
	if args.cnn_dim > 0:
		x = Conv1DWithMasking(nb_filter=args.cnn_dim, filter_length=args.cnn_window_size, border_mode=cnn_border_mode, subsample_length=1)(x)
		
	# RNN Layer
	if args.rnn_dim > 0:
		rnn_layer = RNN(args.rnn_dim, return_sequences=True, consume_less=args.rnn_opt, dropout_W=dropout_W, dropout_U=dropout_U)
		if args.bi:
			rnn_layer = Bidirectional(rnn_layer)
		x = rnn_layer(x)
		if args.dropout_prob > 0:
			x = Dropout(args.dropout_prob)(x)
			
		# Stack 2 Layers
		if args.rnn_2l or args.rnn_3l:
			rnn_layer2 = RNN(args.rnn_dim, return_sequences=True, consume_less=args.rnn_opt, dropout_W=dropout_W, dropout_U=dropout_U)
			if args.bi:
				rnn_layer2 = Bidirectional(rnn_layer2)
			x = rnn_layer2(x)
			if args.dropout_prob > 0:
				x = Dropout(args.dropout_prob)(x)
			# Stack 3 Layers
			if args.rnn_3l:
				rnn_layer3 = RNN(args.rnn_dim, return_sequences=True, consume_less=args.rnn_opt, dropout_W=dropout_W, dropout_U=dropout_U)
				if args.bi:
					rnn_layer3 = Bidirectional(rnn_layer3)
				x = rnn_layer3(x)
				if args.dropout_prob > 0:
					x = Dropout(args.dropout_prob)(x)
			
	# Mean over Time
	if args.aggregation == 'mot':
		x = MeanOverTime(mask_zero=True)(x)
	elif args.aggregation == 'sot':
		x = MeanOverTime(mask_zero=True)(x)
	else:
		raise NotImplementedError
				
	# Augmented TF/IDF Layer
	if args.tfidf > 0:
		pca_input = Input(shape=(args.tfidf,), dtype='float32')
		merged = merge([x,pca_input], mode='concat')
	else:
		merged = x
	
	# Augmented Numerical Features
	if args.features:
		ftr_input = Input(shape=(13,), dtype='float32')
		merged = merge([merged,ftr_input], mode='concat')
				
	# Optional Dense Layer	
	if args.dense > 0:
		if args.loss == 'hng':
			merged = Dense(num_outputs, init=dense_init, W_regularizer=l2(0.001), activity_regularizer=activity_l2(0.001) )(merged)
		else:
			merged = Dense(num_outputs, init=dense_init)(merged)
		if final_activation == 'relu' or final_activation == 'linear':
			merged = BatchNormalization()(merged)
		merged = Activation(dense_activation)(merged)
		if args.dropout_prob > 0:
			merged = Dropout(args.dropout_prob)(merged)
		
	# Final Prediction Layer
	if args.loss == 'hng':
		merged = Dense(num_outputs, init=final_init, W_regularizer=l2(0.001), activity_regularizer=activity_l2(0.001) )(merged)
	else:
		merged = Dense(num_outputs, init=final_init)(merged)
	if final_activation == 'relu' or final_activation == 'linear':
		merged = BatchNormalization()(merged)
	predictions = Activation(final_activation)(merged)
	
	# Model Input/Output
	model_input = [sequence,]	
	if args.tfidf > 0:
		model_input.append(pca_input)
	if args.features:
		model_input.append(ftr_input)

	model = Model(input=model_input, output=predictions)

	logger.info('  Model Done')
	return model

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
		final_init = 'glorot_uniform'
		if args.loss == 'cnp':
			final_activation = 'softmax'
		elif args.loss == 'hng':
			final_activation = 'linear'
	elif args.model_type == 'reg':
		logger.info('Building a REGRESSION model with POOLING')
		if args.normalize:
			final_activation = 'sigmoid'
			final_init = 'he_normal'
			dense_activation = 'tanh'
			dense_init = 'he_normal'
		else:
			final_activation = 'relu'
			final_init = 'he_uniform'
			dense_activation = 'tanh'
			dense_init = 'he_uniform'
	else:
		raise NotImplementedError
	
	sequence = Input(shape=(overal_maxlen,), dtype='int32')
	x = Embedding(len(vocab), args.emb_dim, mask_zero=True, init=my_init, trainable=args.embd_train)(sequence)
	
	# Conv Layer
	if args.cnn_dim > 0:
		x = Conv1DWithMasking(nb_filter=args.cnn_dim, filter_length=args.cnn_window_size, border_mode=cnn_border_mode, subsample_length=1)(x)
		
	# RNN Layer
	if args.rnn_dim > 0:
		forwards = RNN(args.rnn_dim, return_sequences=True, dropout_W=dropout_W, dropout_U=dropout_U)(x)
		if args.bi:
			backwards = RNN(args.rnn_dim, return_sequences=True, dropout_W=dropout_W, dropout_U=dropout_U, go_backwards=True)(x)
		if args.dropout_prob > 0:
			forwards = Dropout(args.dropout_prob)(forwards)
			if args.bi:
				backwards = Dropout(args.dropout_prob)(backwards)
		# Stack 2 Layers
		if args.rnn_2l or args.rnn_3l:
			if args.bi:
				merged = merge([forwards, backwards], mode='concat', concat_axis=-1)
			else:
				merged = forwards
			forwards = RNN(args.rnn_dim, return_sequences=True, dropout_W=dropout_W, dropout_U=dropout_U)(merged)
			if args.bi:
				backwards = RNN(args.rnn_dim, return_sequences=True, dropout_W=dropout_W, dropout_U=dropout_U, go_backwards=True)(merged)
			if args.dropout_prob > 0:
				forwards = Dropout(args.dropout_prob)(forwards)
				if args.bi:
					backwards = Dropout(args.dropout_prob)(backwards)
			# Stack 3 Layers
			if args.rnn_3l:
				if args.bi:
					merged = merge([forwards, backwards], mode='concat', concat_axis=-1)
				else:
					merged = forwards
				forwards = RNN(args.rnn_dim, return_sequences=True, dropout_W=dropout_W, dropout_U=dropout_U)(merged)
				if args.bi:
					backwards = RNN(args.rnn_dim, return_sequences=True, dropout_W=dropout_W, dropout_U=dropout_U, go_backwards=True)(merged)
				if args.dropout_prob > 0:
					forwards = Dropout(args.dropout_prob)(forwards)
					if args.bi:
						backwards = Dropout(args.dropout_prob)(backwards)
		
		if args.aggregation == 'mot':
			forwards = MeanOverTime(mask_zero=True)(forwards)
			if args.bi:
				backwards = MeanOverTime(mask_zero=True)(backwards)
				merged = merge([forwards, backwards], mode='concat', concat_axis=-1)
			else:
				merged = forwards
		else:
			raise NotImplementedError
		
		# Augmented TF/IDF Layer	
		if args.tfidf > 0:
			pca_input = Input(shape=(args.tfidf,), dtype='float32')
			tfidfmerged = merge([merged,pca_input], mode='concat')
		else:
			tfidfmerged = merged
			
		# Optional Dense Layer	
		if args.dense > 0:
			if args.loss == 'hng':
				tfidfmerged = Dense(num_outputs, init=dense_init, W_regularizer=l2(0.001), activity_regularizer=activity_l2(0.001) )(tfidfmerged)
			else:
				tfidfmerged = Dense(num_outputs, init=dense_init)(tfidfmerged)
			if final_activation == 'relu' or final_activation == 'linear':
				tfidfmerged = BatchNormalization()(tfidfmerged)
			tfidfmerged = Activation(dense_activation)(tfidfmerged)
			if args.dropout_prob > 0:
				tfidfmerged = Dropout(args.dropout_prob)(tfidfmerged)
			
		# Final Prediction Layer
		if args.loss == 'hng':
			tfidfmerged = Dense(num_outputs, init=final_init, W_regularizer=l2(0.001), activity_regularizer=activity_l2(0.001) )(tfidfmerged)
		else:
			tfidfmerged = Dense(num_outputs, init=final_init)(tfidfmerged)
		if final_activation == 'relu' or final_activation == 'linear':
			tfidfmerged = BatchNormalization()(tfidfmerged)
		predictions = Activation(final_activation)(tfidfmerged)
		
	else: # if no rnn
		if args.dropout_prob > 0:
			x = Dropout(args.dropout_prob)(x)
		# Mean over Time
		if args.aggregation == 'mot':
			x = MeanOverTime(mask_zero=True)(x)
		else:
			raise NotImplementedError
		# Augmented TF/IDF Layer
		if args.tfidf > 0:
			pca_input = Input(shape=(args.tfidf,), dtype='float32')
			z = merge([x,pca_input], mode='concat')
		else:
			z = x
		# Optional Dense Layer
		if args.dense > 0:
			if args.loss == 'hng':
				z = Dense(args.dense, init=dense_init, W_regularizer=l2(0.001), activity_regularizer=activity_l2(0.001) )(z)
			else:
				z = Dense(args.dense, init=dense_init)(z)
			if final_activation == 'relu' or final_activation == 'linear':
				z = BatchNormalization()(z)	
			z = Activation(dense_activation)(z)
			if args.dropout_prob > 0:
				z = Dropout(args.dropout_prob)(z)
		# Final Prediction Layer
		if args.loss == 'hng':
			z = Dense(num_outputs, init=final_init, W_regularizer=l2(0.001), activity_regularizer=activity_l2(0.001) )(z)
		else:
			z = Dense(args.dense, init=dense_init)(z)
		if final_activation == 'relu' or final_activation == 'linear':
			z = BatchNormalization()(z)
		predictions = Activation(final_activation)(z)
		
	# Model Input/Output	
	if args.tfidf > 0:
		model = Model(input=[sequence, pca_input], output=predictions)
	else:
		model = Model(input=sequence, output=predictions)


# 	if args.model_type == 'cls':
# 		logger.info('Building a CLASSIFICATION model')
# 		sequence = Input(shape=(overal_maxlen,), dtype='int32')
# 		x = Embedding(len(vocab), args.emb_dim, mask_zero=True, init=my_init, trainable=args.embd_train)(sequence)
# 		if args.cnn_dim > 0:
# 			x = Conv1DWithMasking(nb_filter=args.cnn_dim, filter_length=args.cnn_window_size, border_mode=cnn_border_mode, subsample_length=1)(x)
# 		if args.rnn_dim > 0:
# 			x = RNN(args.rnn_dim, return_sequences=False, dropout_W=dropout_W, dropout_U=dropout_U)(x)
# 		predictions = Dense(num_outputs, activation='softmax')(x)
# 		model = Model(input=sequence, output=predictions)

# 	elif args.model_type == 'clsp':
		
# 	elif args.model_type == 'mlp':
# 		logger.info('Building a linear model with POOLING')
# 		sequence = Input(shape=(overal_maxlen,), dtype='int32')
# 		x = Embedding(len(vocab), args.emb_dim, mask_zero=True, init=my_init, trainable=args.embd_train)(sequence)
# 		if args.dropout_prob > 0:
# 			x = Dropout(args.dropout_prob)(x)
# 		x = MeanOverTime(mask_zero=True)(x)
# 		if args.tfidf > 0:
# 			z = merge([x,pca_input], mode='concat')
# 		else:
# 			z = x
# 		if args.dense > 0:
# 			z = Dense(args.dense, activation='tanh')(z)
# 			if args.dropout_prob > 0:
# 				z = Dropout(args.dropout_prob)(z)
# 		predictions = Dense(num_outputs, activation='softmax')(z)
# 		if args.tfidf > 0:
# 			model = Model(input=[sequence, pca_input], output=predictions)
# 		else:
# 			model = Model(input=sequence, output=predictions)
# 			
# 	elif args.model_type == 'reg':
# 		logger.info('Building a REGRESSION model')
# 		model = Sequential()
# 		model.add(Embedding(len(vocab), args.emb_dim, mask_zero=True, init=my_init, trainable=args.embd_train))
# 		if args.cnn_dim > 0:
# 			model.add(Conv1DWithMasking(nb_filter=args.cnn_dim, filter_length=args.cnn_window_size, border_mode=cnn_border_mode, subsample_length=1))
# 		if args.rnn_dim > 0:
# 			model.add(RNN(args.rnn_dim, return_sequences=False, dropout_W=dropout_W, dropout_U=dropout_U))
# 		if args.dropout_prob > 0:
# 			model.add(Dropout(args.dropout_prob))
# 		model.add(Dense(num_outputs))
# 		if not args.skip_init_bias:
# 			bias_value = (np.log(initial_mean_value) - np.log(1 - initial_mean_value)).astype(K.floatx())
# 			model.layers[-1].b.set_value(bias_value)
# 		model.add(Activation('sigmoid'))
# 	
# 	elif args.model_type == 'regp':
# 		logger.info('Building a REGRESSION model with POOLING')
# 		model = Sequential()
# 		model.add(Embedding(len(vocab), args.emb_dim, mask_zero=True, init=my_init, trainable=args.embd_train))
# 		if args.cnn_dim > 0:
# 			model.add(Conv1DWithMasking(nb_filter=args.cnn_dim, filter_length=args.cnn_window_size, border_mode=cnn_border_mode, subsample_length=1))
# 		if args.rnn_dim > 0:
# 			model.add(RNN(args.rnn_dim, return_sequences=True, dropout_W=dropout_W, dropout_U=dropout_U))
# 		if args.dropout_prob > 0:
# 			model.add(Dropout(args.dropout_prob))
# 		if args.aggregation == 'mot':
# 			model.add(MeanOverTime(mask_zero=True))
# 		elif args.aggregation.startswith('att'):
# 			model.add(Attention(op=args.aggregation, activation='tanh', init_stdev=0.01))
# 		model.add(Dense(num_outputs))
# 		if not args.skip_init_bias:
# 			bias_value = (np.log(initial_mean_value) - np.log(1 - initial_mean_value)).astype(K.floatx())
# # 			model.layers[-1].b.set_value(bias_value)
# 			K.set_value(model.layers[-1].b, bias_value)
# 		model.add(Activation('sigmoid'))
# 	
# 	elif args.model_type == 'breg':
# 		logger.info('Building a BIDIRECTIONAL REGRESSION model')
# 		sequence = Input(shape=(overal_maxlen,), dtype='int32')
# 		output = Embedding(len(vocab), args.emb_dim, mask_zero=True, init=my_init, trainable=args.embd_train)(sequence)
# 		if args.cnn_dim > 0:
# 			output = Conv1DWithMasking(nb_filter=args.cnn_dim, filter_length=args.cnn_window_size, border_mode=cnn_border_mode, subsample_length=1)(output)
# 		if args.rnn_dim > 0:
# 			forwards = RNN(args.rnn_dim, return_sequences=False, dropout_W=dropout_W, dropout_U=dropout_U)(output)
# 			backwards = RNN(args.rnn_dim, return_sequences=False, dropout_W=dropout_W, dropout_U=dropout_U, go_backwards=True)(output)
# 		if args.dropout_prob > 0:
# 			forwards = Dropout(args.dropout_prob)(forwards)
# 			backwards = Dropout(args.dropout_prob)(backwards)
# 		merged = merge([forwards, backwards], mode='concat', concat_axis=-1)
# 		densed = Dense(num_outputs)(merged)
# 		if not args.skip_init_bias:
# 			raise NotImplementedError
# 		score = Activation('sigmoid')(densed)
# 		model = Model(input=sequence, output=score)
# 	
# 	elif args.model_type == 'bregp':
# 		logger.info('Building a BIDIRECTIONAL REGRESSION model with POOLING')
# 		sequence = Input(shape=(overal_maxlen,), dtype='int32')
# 		output = Embedding(len(vocab), args.emb_dim, mask_zero=True, init=my_init, trainable=args.embd_train)(sequence)
# 		if args.cnn_dim > 0:
# 			output = Conv1DWithMasking(nb_filter=args.cnn_dim, filter_length=args.cnn_window_size, border_mode=cnn_border_mode, subsample_length=1)(output)
# 		if args.rnn_dim > 0:
# 			forwards = RNN(args.rnn_dim, return_sequences=True, dropout_W=dropout_W, dropout_U=dropout_U)(output)
# 			backwards = RNN(args.rnn_dim, return_sequences=True, dropout_W=dropout_W, dropout_U=dropout_U, go_backwards=True)(output)
# 		if args.dropout_prob > 0:
# 			forwards = Dropout(args.dropout_prob)(forwards)
# 			backwards = Dropout(args.dropout_prob)(backwards)
# 		forwards_mean = MeanOverTime(mask_zero=True)(forwards)
# 		backwards_mean = MeanOverTime(mask_zero=True)(backwards)
# 		merged = merge([forwards_mean, backwards_mean], mode='concat', concat_axis=-1)
# 		densed = Dense(num_outputs)(merged)
# 		if not args.skip_init_bias:
# 			raise NotImplementedError
# 		score = Activation('sigmoid')(densed)
# 		model = Model(input=sequence, output=score)
	
	logger.info('  Model Done')		
	return model

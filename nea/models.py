import numpy as np
import logging

logger = logging.getLogger(__name__)

def create_model(args, initial_mean_value, overal_maxlen, vocab, pca_len=50):
	
	import keras.backend as K
	from keras.layers.embeddings import Embedding
	from keras.models import Sequential, Model
	from keras.layers.core import Dense, Dropout, Activation
	from nea.my_layers import Attention, MeanOverTime, Conv1DWithMasking
	from keras.engine.topology import Input, merge
	
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
	
	dropout_W = args.dropout_prob		# default=0.5
	dropout_U = args.dropout_prob		# default=0.1
	cnn_border_mode='same'
	if "reg" in args.model_type:
		if initial_mean_value.ndim == 0:
			initial_mean_value = np.expand_dims(initial_mean_value, axis=1)
		num_outputs = len(initial_mean_value)
	else:
		num_outputs = initial_mean_value

	###############################################################################################################################
	## Initialize embeddings if requested
	#
	my_trainable = True
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
	
	if args.model_type == 'cls':
		logger.info('Building a CLASSIFICATION model')
		sequence = Input(shape=(overal_maxlen,), dtype='int32')
		x = Embedding(args.vocab_size, args.emb_dim, mask_zero=True, init=my_init, trainable=my_trainable)(sequence)
		if args.cnn_dim > 0:
			x = Conv1DWithMasking(nb_filter=args.cnn_dim, filter_length=args.cnn_window_size, border_mode=cnn_border_mode, subsample_length=1)(x)
		if args.rnn_dim > 0:
			x = RNN(args.rnn_dim, return_sequences=True, dropout_W=dropout_W, dropout_U=dropout_U)(x)
		predictions = Dense(num_outputs, activation='softmax')(x)
		model = Model(input=sequence, output=predictions)

	elif args.model_type == 'clsp':
		logger.info('Building a CLASSIFICATION model with POOLING')
		sequence = Input(shape=(overal_maxlen,), dtype='int32')
		x = Embedding(args.vocab_size, args.emb_dim, mask_zero=True, init=my_init, trainable=my_trainable)(sequence)
		if args.cnn_dim > 0:
			x = Conv1DWithMasking(nb_filter=args.cnn_dim, filter_length=args.cnn_window_size, border_mode=cnn_border_mode, subsample_length=1)(x)
		if args.rnn_dim > 0:
			forwards = RNN(args.rnn_dim, return_sequences=True, dropout_W=dropout_W, dropout_U=dropout_U)(x)
# 			backwards = RNN(args.rnn_dim, return_sequences=True, dropout_W=dropout_W, dropout_U=dropout_U, go_backwards=True)(output)
		if args.dropout_prob > 0:
			forwards = Dropout(args.dropout_prob)(forwards)
# 			backwards = Dropout(args.dropout_prob)(backwards)
		forwards_mean = MeanOverTime(mask_zero=True)(forwards)
# 		backwards_mean = MeanOverTime(mask_zero=True)(backwards)
# 		merged = merge([forwards_mean, backwards_mean], mode='concat', concat_axis=-1)
# 		densed = Dense(32, activation='tanh')(forwards_mean)
# 		droped = Dropout(args.dropout_prob)(densed)
		predictions = Dense(num_outputs, activation='softmax')(forwards_mean)
		model = Model(input=sequence, output=predictions)

	elif args.model_type == 'mlp':
		logger.info('Building a linear model with POOLING')
		sequence = Input(shape=(overal_maxlen,), dtype='int32')
		pca_input = Input(shape=(pca_len,), dtype='float32')
		x = Embedding(args.vocab_size, args.emb_dim, mask_zero=True, init=my_init, trainable=my_trainable)(sequence)
		x = MeanOverTime(mask_zero=True)(x)
		z = merge([x,pca_input], mode='concat')
		z = Dense(32, activation='tanh')(z)
		z = Dropout(args.dropout_prob)(z)
		predictions = Dense(num_outputs, activation='softmax')(z)
		model = Model(input=[sequence, pca_input], output=predictions)
				
	elif args.model_type == 'reg':
		logger.info('Building a REGRESSION model')
		model = Sequential()
		model.add(Embedding(args.vocab_size, args.emb_dim, mask_zero=True, init=my_init, trainable=my_trainable))
		if args.cnn_dim > 0:
			model.add(Conv1DWithMasking(nb_filter=args.cnn_dim, filter_length=args.cnn_window_size, border_mode=cnn_border_mode, subsample_length=1))
		if args.rnn_dim > 0:
			model.add(RNN(args.rnn_dim, return_sequences=False, dropout_W=dropout_W, dropout_U=dropout_U))
		if args.dropout_prob > 0:
			model.add(Dropout(args.dropout_prob))
		model.add(Dense(num_outputs))
		if not args.skip_init_bias:
			bias_value = (np.log(initial_mean_value) - np.log(1 - initial_mean_value)).astype(K.floatx())
			model.layers[-1].b.set_value(bias_value)
		model.add(Activation('sigmoid'))
	
	elif args.model_type == 'regp':
		logger.info('Building a REGRESSION model with POOLING')
		model = Sequential()
		model.add(Embedding(args.vocab_size, args.emb_dim, mask_zero=True, init=my_init, trainable=my_trainable))
		if args.cnn_dim > 0:
			model.add(Conv1DWithMasking(nb_filter=args.cnn_dim, filter_length=args.cnn_window_size, border_mode=cnn_border_mode, subsample_length=1))
		if args.rnn_dim > 0:
			model.add(RNN(args.rnn_dim, return_sequences=True, dropout_W=dropout_W, dropout_U=dropout_U))
		if args.dropout_prob > 0:
			model.add(Dropout(args.dropout_prob))
		if args.aggregation == 'mot':
			model.add(MeanOverTime(mask_zero=True))
		elif args.aggregation.startswith('att'):
			model.add(Attention(op=args.aggregation, activation='tanh', init_stdev=0.01))
		model.add(Dense(num_outputs))
		if not args.skip_init_bias:
			bias_value = (np.log(initial_mean_value) - np.log(1 - initial_mean_value)).astype(K.floatx())
# 			model.layers[-1].b.set_value(bias_value)
			K.set_value(model.layers[-1].b, bias_value)
		model.add(Activation('sigmoid'))
	
	elif args.model_type == 'breg':
		logger.info('Building a BIDIRECTIONAL REGRESSION model')
		sequence = Input(shape=(overal_maxlen,), dtype='int32')
		output = Embedding(args.vocab_size, args.emb_dim, mask_zero=True, init=my_init, trainable=my_trainable)(sequence)
		if args.cnn_dim > 0:
			output = Conv1DWithMasking(nb_filter=args.cnn_dim, filter_length=args.cnn_window_size, border_mode=cnn_border_mode, subsample_length=1)(output)
		if args.rnn_dim > 0:
			forwards = RNN(args.rnn_dim, return_sequences=False, dropout_W=dropout_W, dropout_U=dropout_U)(output)
			backwards = RNN(args.rnn_dim, return_sequences=False, dropout_W=dropout_W, dropout_U=dropout_U, go_backwards=True)(output)
		if args.dropout_prob > 0:
			forwards = Dropout(args.dropout_prob)(forwards)
			backwards = Dropout(args.dropout_prob)(backwards)
		merged = merge([forwards, backwards], mode='concat', concat_axis=-1)
		densed = Dense(num_outputs)(merged)
		if not args.skip_init_bias:
			raise NotImplementedError
		score = Activation('sigmoid')(densed)
		model = Model(input=sequence, output=score)
	
	elif args.model_type == 'bregp':
		logger.info('Building a BIDIRECTIONAL REGRESSION model with POOLING')
		sequence = Input(shape=(overal_maxlen,), dtype='int32')
		output = Embedding(args.vocab_size, args.emb_dim, mask_zero=True, init=my_init, trainable=my_trainable)(sequence)
		if args.cnn_dim > 0:
			output = Conv1DWithMasking(nb_filter=args.cnn_dim, filter_length=args.cnn_window_size, border_mode=cnn_border_mode, subsample_length=1)(output)
		if args.rnn_dim > 0:
			forwards = RNN(args.rnn_dim, return_sequences=True, dropout_W=dropout_W, dropout_U=dropout_U)(output)
			backwards = RNN(args.rnn_dim, return_sequences=True, dropout_W=dropout_W, dropout_U=dropout_U, go_backwards=True)(output)
		if args.dropout_prob > 0:
			forwards = Dropout(args.dropout_prob)(forwards)
			backwards = Dropout(args.dropout_prob)(backwards)
		forwards_mean = MeanOverTime(mask_zero=True)(forwards)
		backwards_mean = MeanOverTime(mask_zero=True)(backwards)
		merged = merge([forwards_mean, backwards_mean], mode='concat', concat_axis=-1)
		densed = Dense(num_outputs)(merged)
		if not args.skip_init_bias:
			raise NotImplementedError
		score = Activation('sigmoid')(densed)
		model = Model(input=sequence, output=score)
	
	logger.info('  Done')
		
	return model

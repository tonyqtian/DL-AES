'''
Created on Feb 26, 2017

@author: tonyq
'''
import matplotlib
matplotlib.use('Agg')

import logging
import numpy as np
import nea.utils as U
import pickle as pk
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping

logger = logging.getLogger(__name__)

def train(args):
	out_dir = args.out_dir_path	
	U.mkdir_p(out_dir + '/preds')
	timestr = U.set_logger(onscreen=args.onscreen, out_dir=out_dir)
	U.print_args(args)
	
# 	assert args.model_type in {'mlp', 'cls', 'clsp', 'reg', 'regp', 'breg', 'bregp'}
	assert args.model_type in {'cls', 'reg'}
	assert args.algorithm in {'rmsprop', 'sgd', 'adagrad', 'adadelta', 'adam', 'adamax'}
	assert args.loss in {'mse', 'mae', 'cnp', 'hng'}
	assert args.recurrent_unit in {'lstm', 'gru', 'simple'}
	assert args.aggregation in {'mot', 'sot'}
	
	if args.seed > 0:
		np.random.seed(args.seed)
	
	from nea.asap_evaluator import Evaluator
	import nea.asap_reader as dataset
	
	###############################################################################################################################
	## Prepare data
	#
	
	from keras.preprocessing import sequence
	
	if args.valid_split > 0:
		(train_x, train_y, train_pmt), (test_x, test_y, test_pmt), vocab, overal_maxlen = dataset.get_data(
			(args.train_path, args.test_path), args.prompt_id, args.vocab_size, args.maxlen, 
			tokenize_text=True, to_lower=True, sort_by_len=False, vocab_path=args.vocab_path)
	else:
	# data_x is a list of lists
		(train_x, train_y, train_pmt), (dev_x, dev_y, dev_pmt), (test_x, test_y, test_pmt), vocab, overal_maxlen = dataset.get_data(
			(args.train_path, args.dev_path, args.test_path), args.prompt_id, args.vocab_size, args.maxlen, 
			tokenize_text=True, to_lower=True, sort_by_len=False, vocab_path=args.vocab_path)
	
	if args.tfidf > 0:
		train_pca, TfIdf, Pca = dataset.get_tfidf(args.train_path, args.prompt_id, pca_dim=args.tfidf, training_material=True)
		if args.valid_split == 0:
			dev_pca, _, _ = dataset.get_tfidf(args.dev_path, args.prompt_id, pca_dim=args.tfidf, tfidf=TfIdf, pca=Pca, training_material=False)
		test_pca, _, _ = dataset.get_tfidf(args.test_path, args.prompt_id, pca_dim=args.tfidf, tfidf=TfIdf, pca=Pca, training_material=False)
	else:
		dev_pca = None
		test_pca = None
		
	if args.features:
		train_ftr = dataset.get_features(args.train_path, args.train_feature_path, args.prompt_id)
		if args.valid_split == 0:
			valid_ftr = dataset.get_features(args.dev_path, args.dev_feature_path, args.prompt_id)
		test_ftr = dataset.get_features(args.test_path, args.test_feature_path, args.prompt_id)
	else:
		test_ftr = None
	
	if not args.vocab_path:
		# Dump vocab
		with open(out_dir + '/vocab.pkl', 'wb') as vocab_file:
			pk.dump(vocab, vocab_file)
	
	# Pad sequences for mini-batch processing
# 	if args.model_type in {'breg', 'bregp', 'clsp', 'cls', 'mlp'}:
	# 	assert args.rnn_dim > 0
	# 	assert args.recurrent_unit == 'lstm'
	train_x = sequence.pad_sequences(train_x, maxlen=overal_maxlen)
	if args.valid_split == 0:
		dev_x = sequence.pad_sequences(dev_x, maxlen=overal_maxlen)
	test_x = sequence.pad_sequences(test_x, maxlen=overal_maxlen)
# 	else:
# 		train_x = sequence.pad_sequences(train_x)
# 		dev_x = sequence.pad_sequences(dev_x)
# 		test_x = sequence.pad_sequences(test_x)
	
	###############################################################################################################################
	## Some statistics
	#
	
	import keras.backend as K
	
	train_y = np.array(train_y, dtype=K.floatx())
	if args.valid_split == 0:
		dev_y = np.array(dev_y, dtype=K.floatx())
	test_y = np.array(test_y, dtype=K.floatx())
	
	if args.prompt_id >= 0:
		train_pmt = np.array(train_pmt, dtype='int32')
		if args.valid_split == 0:
			dev_pmt = np.array(dev_pmt, dtype='int32')
		test_pmt = np.array(test_pmt, dtype='int32')
	
	# count score distribution
	bincounts, mfs_list = U.bincounts(train_y)
	with open('%s/bincounts.txt' % out_dir, 'w') as output_file:
		for bincount in bincounts:
			output_file.write(str(bincount) + '\n')
	
	train_mean = train_y.mean(axis=0)
	train_std = train_y.std(axis=0)
	train_max = train_y.max(axis=0)
	train_min = train_y.min(axis=0)
# 	dev_mean = dev_y.mean(axis=0)
# 	dev_std = dev_y.std(axis=0)
# 	test_mean = test_y.mean(axis=0)
# 	test_std = test_y.std(axis=0)
	
	logger.info('Statistics:')
	
	logger.info('  train_x shape: ' + str(np.array(train_x).shape))
	if args.valid_split == 0:
		logger.info('  dev_x shape:   ' + str(np.array(dev_x).shape))
	logger.info('  test_x shape:  ' + str(np.array(test_x).shape))
	
	logger.info('  train_y shape: ' + str(train_y.shape))
	if args.valid_split == 0:
		logger.info('  dev_y shape:   ' + str(dev_y.shape))
	logger.info('  test_y shape:  ' + str(test_y.shape))
	
	logger.info('  train_y max: %d, min: %d, mean: %.2f, stdev: %.3f, MFC: %s' % (train_max, train_min, train_mean, train_std, str(mfs_list)))
	logger.info('  train_y statistic: %s' % (str(bincounts[0]),))
	
	# We need the dev and test sets in the original scale for evaluation
# 	if args.valid_split == 0:
# 		dev_y_org = dev_y.astype(dataset.get_ref_dtype())
	test_y_org = test_y.astype(dataset.get_ref_dtype())
	
	if "reg" in args.model_type:
		if args.normalize:
			logger.info('  normalize score to range (0,1)')
			# Convert scores to boundary of [0 1] for training and evaluation (loss calculation)
			train_y = dataset.get_model_friendly_scores(train_y, train_pmt)
			if args.valid_split == 0:
				dev_y = dataset.get_model_friendly_scores(dev_y, dev_pmt)
			test_y = dataset.get_model_friendly_scores(test_y, test_pmt)
	else:
		logger.info('  covert train_y to one hot shape')
		assert len(bincounts) == 1, "support only one y value"
		categ = int(max(bincounts[0].keys())) + 1
		# covert to np array to minus 1 to get zero based value
		train_y = to_categorical(train_y, categ)
		if args.valid_split == 0:
			dev_y = to_categorical(dev_y, categ)
		test_y = to_categorical(test_y, categ)
			
	###############################################################################################################################
	## Optimizer algorithm
	#
	
	from nea.optimizers import get_optimizer	
	optimizer = get_optimizer(args)
	
	###############################################################################################################################
	## Building model
	#
	
	if "reg" in args.model_type:
		logger.info('  use regression model')
		final_categ = train_y.mean(axis=0)
		if args.loss == 'mae':
			loss = 'mean_absolute_error'
			metric = 'mean_squared_error'
		elif args.loss == 'mse':
			loss = 'mean_squared_error'
			metric = 'mean_absolute_error'
		else:
			raise NotImplementedError
	else:
		logger.info('  use classification model')
		final_categ = categ
		if args.loss == 'cnp':
			loss = 'categorical_crossentropy'
			metric = 'categorical_accuracy'
		elif args.loss == 'hng':
			loss = 'hinge'
			metric = 'squared_hinge'
		else:
			raise NotImplementedError
	
	from nea.models import create_model		
	model = create_model(args, final_categ, overal_maxlen, vocab)
	model.compile(loss=loss, optimizer=optimizer, metrics=[metric])
	
	if args.onscreen: model.summary()
	
	###############################################################################################################################
	## Plotting model
	#
	
	from keras.utils.visualize_util import plot	
	plot(model, to_file = out_dir + '/' + timestr + 'model_plot.png')
	
	###############################################################################################################################
	## Save model architecture
	#
	
	logger.info('Saving model architecture')
	with open(out_dir + '/'+ timestr + 'model_config.json', 'w') as arch:
		arch.write(model.to_json(indent=2))
	logger.info('  Done')
		
	###############################################################################################################################
	## Initialize Evaluator
	#
	
	evl = Evaluator(args, out_dir, timestr, metric, test_x, test_y, test_y_org, test_pmt, test_pca=test_pca, test_ftr=test_ftr)
	earlystop = EarlyStopping(patience = args.earlystop, verbose=1, mode='auto' )
	
	###############################################################################################################################
	## Training
	#
	
	logger.info('------------------------------------------------------------------------------------------')
	logger.info('Initial Evaluation:')
	evl.eval(model, -1, print_info=True)
	
	model_train_x = [train_x,]
	if not args.valid_split:
		model_dev_x = [dev_x,]
	if args.tfidf > 0:
		model_train_x.append(train_pca)
		if not args.valid_split:
			model_dev_x.append(dev_pca)
	if args.features:
		model_train_x.append(train_ftr)
		if not args.valid_split:
			model_dev_x.append(valid_ftr)
		
	if args.valid_split > 0:
		model.fit(model_train_x, train_y, validation_split=args.valid_split, batch_size=args.batch_size, nb_epoch=args.epochs, verbose=args.verbose, callbacks=[earlystop, evl])
	else:	
		model.fit(model_train_x, train_y, validation_data=(model_dev_x, dev_y), batch_size=args.batch_size, nb_epoch=args.epochs, verbose=args.verbose, callbacks=[earlystop, evl])
	
	return evl.print_final_info()
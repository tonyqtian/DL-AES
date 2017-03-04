'''
Created on Feb 26, 2017

@author: tonyq
'''
import logging
import numpy as np
# from time import time
import nea.utils as U
import pickle as pk
from keras.utils.np_utils import to_categorical

import matplotlib
from keras.callbacks import EarlyStopping
from keras.metrics import hinge
matplotlib.use('Agg')

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
	
# 	evl = Evaluator(args, out_dir, dev_x, test_x, dev_y, test_y, dev_y_org, test_y_org, dev_pmt, test_pmt, dev_pca=dev_pca, test_pca=test_pca)
	evl = Evaluator(args, out_dir, timestr, metric, test_x, test_y, test_y_org, test_pmt, test_pca=test_pca)
	earlystop = EarlyStopping(monitor=metric, patience = args.earlystop, verbose=1, mode='auto' )
	
	###############################################################################################################################
	## Training
	#
	
	logger.info('------------------------------------------------------------------------------------------')
	logger.info('Initial Evaluation:')
	evl.eval(model, -1, print_info=True)
	
# 	total_train_time = 0
# 	total_eval_time = 0
	
# 	if args.plot:
# 		training_epochs = []
# 		training_losses = []
# 		training_accuracy = []
# 		dev_losses = []
# 		dev_accuracy = []
# 		dev_qwks = []
# 		test_qwks = []

	if args.valid_split > 0:
		if args.tfidf > 0:
			model.fit([train_x, train_pca], train_y, validation_split=args.valid_split, batch_size=args.batch_size, nb_epoch=args.epochs, verbose=args.verbose, callbacks=[earlystop, evl])
		else:
			model.fit(train_x, train_y, validation_split=args.valid_split, batch_size=args.batch_size, nb_epoch=args.epochs, verbose=args.verbose, callbacks=[earlystop, evl])
	else:	
		if args.tfidf > 0:
			model.fit([train_x, train_pca], train_y, validation_data=([dev_x, dev_pca], dev_y), batch_size=args.batch_size, nb_epoch=args.epochs, verbose=args.verbose, callbacks=[earlystop, evl])
		else:
			model.fit(train_x, train_y, validation_data=(dev_x, dev_y), batch_size=args.batch_size, nb_epoch=args.epochs, verbose=args.verbose, callbacks=[earlystop, evl])
	
# 	for ii in range(args.epochs):
# 		# Training
# 		t0 = time()
# 		if args.tfidf > 0:
# 			train_history = model.fit([train_x, train_pca], train_y, batch_size=args.batch_size, nb_epoch=1, verbose=args.verbose)
# 		else:
# 			train_history = model.fit(train_x, train_y, batch_size=args.batch_size, nb_epoch=1, verbose=args.verbose)
# 		tr_time = time() - t0
# 		total_train_time += tr_time
# 		
# 		# Evaluate
# 		t0 = time()
# 		dev_loss, dev_acc, dev_qwk, test_qwk = evl.eval(model, ii)
# 		evl_time = time() - t0
# 		total_eval_time += evl_time
# 		
# 		# Print information
# 		train_loss = train_history.history['loss'][0]
# 		train_metric = train_history.history[metric][0]
# 		logger.info('Epoch %d, train: %is, evaluation: %is' % (ii, tr_time, evl_time))
# 		logger.info('[Train] loss: %.4f, metric: %.4f' % (train_loss, train_metric))
# 		evl.print_info()
# 		
# 		if args.plot:
# 			training_epochs.append(ii)
# 			training_losses.append(train_loss)
# 			training_accuracy.append(train_metric)
# 			dev_losses.append(dev_loss)
# 			dev_accuracy.append(dev_acc)
# 			dev_qwks.append(dev_qwk)
# 			test_qwks.append(test_qwk)
# 		
# 		if args.earlystop > 0:
# 			if dev_loss / train_loss > args.earlystop:
# 				logger.info('Early stop >>> dev/train loss rate: %0.2f ' % (dev_loss/train_loss,))
# 				break
			
	###############################################################################################################################
	## Summary of the results
	#
	
# 	logger.info('Training:   %i seconds in total' % total_train_time)
# 	logger.info('Evaluation: %i seconds in total' % total_eval_time)
	
# 	if args.plot:
# 		import matplotlib.pyplot as plt
# 		
# 		plt.plot(training_epochs, training_losses, 'b', label='Train Loss')
# 		plt.plot(training_epochs, training_accuracy, 'r.', label='Train Accuracy')
# 		plt.plot(training_epochs, dev_losses, 'g', label='Dev Loss')
# 		plt.plot(training_epochs, dev_accuracy, 'y.', label='Dev Accuracy')
# 		plt.legend()
# 		plt.xlabel('epochs')
# 		plt.savefig(out_dir + '/' + timestr + 'LossAccuracy.png')
# 		# plt.show()
# 		plt.close()
# 		
# 		plt.plot(training_epochs, training_accuracy, 'b', label='Train Accuracy')
# 		plt.plot(training_epochs, dev_accuracy, 'g', label='Dev Accuracy')
# 		plt.plot(training_epochs, dev_qwks, 'r.', label='Dev QWK')
# 		plt.plot(training_epochs, test_qwks, 'y.', label='Test QWK')
# 		plt.xlabel('epochs')
# 		plt.legend()
# 		plt.savefig(out_dir + '/' + timestr + 'QWK.png')
# 		# plt.show()
# 		plt.close()

	return evl.print_final_info()
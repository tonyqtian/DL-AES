import logging
import numpy as np
from nea.my_kappa_calculator import quadratic_weighted_kappa as qwk
# from nea.my_kappa_calculator import linear_weighted_kappa as lwk
# from nea.my_kappa_calculator import cohen_kappa as lwk
import nea.asap_reader as dataset
from keras.callbacks import Callback
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

class Evaluator(Callback):
	
	def __init__(self, arg, out_dir, timestr, metric, test_x, test_y, test_y_org, test_pmt, test_pca=None, test_ftr=None):
		self.arg = arg
		self.test_pmt = test_pmt
		self.out_dir = out_dir
		self.test_x = [test_x,]
		self.test_y = test_y
		self.test_y_org = test_y_org
		self.test_mean = self.test_y_org.mean()
		self.test_std = self.test_y_org.std()
		self.best_test = -1
		self.best_epoch = -1
		if arg.batch_size > len(test_y):
			self.batch_size = 256
		else:
			self.batch_size = arg.batch_size
		self.dump_ref_scores()
		if self.arg.tfidf > 0:
			self.test_x.append(test_pca)
		if self.arg.features:
			self.test_x.append(test_ftr)
		
		self.metric = metric
		self.val_metric = 'val_' + metric
		self.timestr = timestr
		self.losses = []
		self.accs = []
		self.val_accs = []
		self.val_losses = []
		self.test_qwks = []
	
	def dump_ref_scores(self):
		np.savetxt(self.out_dir + '/preds/test_ref.txt', self.test_y_org, fmt='%i')
	
	def dump_predictions(self, test_pred, epoch):
		np.savetxt(self.out_dir + '/preds/test_pred_' + str(epoch) + '.txt', test_pred, fmt='%.8f')

	def calc_qwk(self, test_pred):
		# Kappa only supports integer values
		test_pred_int = np.rint(test_pred).astype('int32')
		test_qwk = qwk(self.test_y_org, test_pred_int)
		return test_qwk
		
	def eval(self, model, epoch, print_info=False):
		self.test_loss, self.test_metric = model.evaluate(self.test_x, self.test_y, batch_size=self.batch_size, verbose=self.arg.verbose)
		self.test_pred = model.predict(self.test_x, batch_size=self.batch_size).squeeze()
		
		if "reg" in self.arg.model_type:
			if self.arg.normalize:
				self.test_pred = dataset.convert_to_dataset_friendly_scores(self.test_pred, self.test_pmt)
		else:
			self.test_pred = dataset.convert_1hot_to_score(self.test_pred)

		self.dump_predictions(self.test_pred, epoch)
		test_qwk = self.calc_qwk(self.test_pred)
	
		if test_qwk > self.best_test:
			self.best_test = test_qwk
			self.best_epoch = epoch
			if epoch > 5:
				model.save_weights(self.out_dir + '/best_model_weights.h5', overwrite=True)
	
		if print_info:
			self.print_info(epoch, test_qwk)
		return test_qwk

	def on_epoch_end(self, epoch, logs={}):
		self.losses.append(logs.get('loss'))
		self.val_losses.append(logs.get('val_loss'))
		self.accs.append(logs.get(self.metric))		
		self.val_accs.append(logs.get(self.val_metric))	
		self.test_qwks.append( self.eval(self.model, epoch, print_info=True) )
		if self.arg.plot:
			self.plothem()
		return

	def plothem(self):
		training_epochs = [i for i in range(len(self.losses))]
		plt.plot(training_epochs, self.losses, 'b', label='Train Loss')
		plt.plot(training_epochs, self.accs, 'r.', label='Train Metric')
		plt.plot(training_epochs, self.val_losses, 'g', label='Valid Loss')
		plt.plot(training_epochs, self.val_accs, 'y.', label='Valid Metric')
		plt.legend()
		plt.xlabel('epochs')
		plt.savefig(self.out_dir + '/' + self.timestr + 'LossAccuracy.png')
		plt.close()
		
		plt.plot(training_epochs, self.accs, 'b', label='Train Metric')
		plt.plot(training_epochs, self.val_accs, 'g', label='Valid Metric')
		plt.plot(training_epochs, self.test_qwks, 'r.', label='Test QWK')
		plt.xlabel('epochs')
		plt.legend()
		plt.savefig(self.out_dir + '/' + self.timestr + 'QWK.png')
		plt.close()	
			
	def print_info(self, epoch, test_qwk):
		logger.info('\n')
		logger.info('[Test]  Epoch: %i' % epoch)
		logger.info('[Test]  loss: %.4f, metric: %.4f, mean: %.3f (%.3f), stdev: %.3f (%.3f)' % (
			self.test_loss, self.test_metric, self.test_pred.mean(), self.test_mean, self.test_pred.std(), self.test_std))
		logger.info('[TEST]  QWK:  %.3f (Best @ %i: %.3f)' % (test_qwk, self.best_epoch, self.best_test))				
		logger.info('--------------------------------------------------------------------------------------------')
	
	def print_final_info(self):
		logger.info('\n')
		logger.info('--------------------------------------------------------------------------------------------')
		logger.info('Best @ Epoch %i:' % self.best_epoch)
		logger.info('  [TEST] QWK: %.3f' % (self.best_test))
		return self.best_test

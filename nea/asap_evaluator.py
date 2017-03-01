# from scipy.stats import pearsonr, spearmanr, kendalltau
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
	
	def __init__(self, arg, out_dir, timestr, metric, test_x, test_y, test_y_org, test_pmt, test_pca=None):
		self.arg = arg
		self.test_pmt = test_pmt
		self.out_dir = out_dir
		self.test_x = test_x
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
		self.test_pca = test_pca
		
		self.metric = metric
		self.val_metric = 'val_' + metric
		self.timestr = timestr
		self.losses = []
		self.accs = []
		self.val_accs = []
		self.val_losses = []
		self.test_qwks = []
	
	def dump_ref_scores(self):
# 		np.savetxt(self.out_dir + '/preds/dev_ref.txt', self.dev_y_org, fmt='%i')
		np.savetxt(self.out_dir + '/preds/test_ref.txt', self.test_y_org, fmt='%i')
	
# 	def dump_predictions(self, dev_pred, test_pred, epoch):
# 		np.savetxt(self.out_dir + '/preds/dev_pred_' + str(epoch) + '.txt', dev_pred, fmt='%.8f')
# 		np.savetxt(self.out_dir + '/preds/test_pred_' + str(epoch) + '.txt', test_pred, fmt='%.8f')

	def dump_predictions(self, test_pred, epoch):
		np.savetxt(self.out_dir + '/preds/test_pred_' + str(epoch) + '.txt', test_pred, fmt='%.8f')
			
# 	def calc_correl(self, dev_pred, test_pred):
# 		dev_prs, _ = pearsonr(dev_pred, self.dev_y_org)
# 		test_prs, _ = pearsonr(test_pred, self.test_y_org)
# 		dev_spr, _ = spearmanr(dev_pred, self.dev_y_org)
# 		test_spr, _ = spearmanr(test_pred, self.test_y_org)
# 		dev_tau, _ = kendalltau(dev_pred, self.dev_y_org)
# 		test_tau, _ = kendalltau(test_pred, self.test_y_org)
# 		return dev_prs, test_prs, dev_spr, test_spr, dev_tau, test_tau
	
# 	def calc_qwk(self, dev_pred, test_pred):
# 		# Kappa only supports integer values
# 		dev_pred_int = np.rint(dev_pred).astype('int32')
# 		test_pred_int = np.rint(test_pred).astype('int32')
# 		dev_qwk = qwk(self.dev_y_org, dev_pred_int)
# 		test_qwk = qwk(self.test_y_org, test_pred_int)
# 		dev_lwk = lwk(self.dev_y_org, dev_pred_int)
# 		test_lwk = lwk(self.test_y_org, test_pred_int)
# 		return dev_qwk, test_qwk, dev_lwk, test_lwk

	def calc_qwk(self, test_pred):
		# Kappa only supports integer values
		test_pred_int = np.rint(test_pred).astype('int32')
		test_qwk = qwk(self.test_y_org, test_pred_int)
		return test_qwk
		
	def eval(self, model, epoch, print_info=False):
		if self.arg.tfidf > 0:
# 			self.dev_loss, self.dev_metric = model.evaluate([self.dev_x, self.dev_pca], self.dev_y, batch_size=self.batch_size, verbose=self.arg.verbose)
			self.test_loss, self.test_metric = model.evaluate([self.test_x, self.test_pca], self.test_y, batch_size=self.batch_size, verbose=self.arg.verbose)
		else:
# 			self.dev_loss, self.dev_metric = model.evaluate(self.dev_x, self.dev_y, batch_size=self.batch_size, verbose=self.arg.verbose)
			self.test_loss, self.test_metric = model.evaluate(self.test_x, self.test_y, batch_size=self.batch_size, verbose=self.arg.verbose)

		if self.arg.tfidf > 0:
# 			self.dev_pred = model.predict([self.dev_x, self.dev_pca], batch_size=self.batch_size).squeeze()
			self.test_pred = model.predict([self.test_x, self.test_pca], batch_size=self.batch_size).squeeze()
		else:		
# 			self.dev_pred = model.predict(self.dev_x, batch_size=self.batch_size).squeeze()
			self.test_pred = model.predict(self.test_x, batch_size=self.batch_size).squeeze()
		
		if "reg" in self.arg.model_type:
			if self.arg.normalize:
# 				self.dev_pred = dataset.convert_to_dataset_friendly_scores(self.dev_pred, self.dev_pmt)
				self.test_pred = dataset.convert_to_dataset_friendly_scores(self.test_pred, self.test_pmt)
		else:
# 			self.dev_pred = dataset.convert_1hot_to_score(self.dev_pred)
			self.test_pred = dataset.convert_1hot_to_score(self.test_pred)
		
# 		self.dump_predictions(self.dev_pred, self.test_pred, epoch)
		self.dump_predictions(self.test_pred, epoch)
		
# 		self.dev_prs, self.test_prs, self.dev_spr, self.test_spr, self.dev_tau, self.test_tau = self.calc_correl(self.dev_pred, self.test_pred)
# 		self.test_prs, self.test_spr, self.test_tau = self.calc_correl(self.test_pred)
		
# 		self.dev_qwk, self.test_qwk, self.dev_lwk, self.test_lwk = self.calc_qwk(self.dev_pred, self.test_pred)
# 		self.test_qwk = self.calc_qwk(self.test_pred)
		test_qwk = self.calc_qwk(self.test_pred)
	
# 		if self.dev_qwk > self.best_dev[0]:
		if test_qwk > self.best_test:
# 			self.best_dev = [self.dev_qwk, self.dev_lwk, self.dev_prs, self.dev_spr, self.dev_tau]
# 			self.best_test = [self.test_qwk, self.test_lwk, self.test_prs, self.test_spr, self.test_tau]
			self.best_test = test_qwk
			self.best_epoch = epoch
			if epoch > 5:
				model.save_weights(self.out_dir + '/best_model_weights.h5', overwrite=True)
	
# 		if self.test_qwk > self.best_test_missed:
# 			self.best_test_missed = self.test_qwk
# 			self.best_test_missed_epoch = epoch

		if print_info:
			self.print_info(test_qwk)
# 		return (self.dev_loss, self.dev_metric, self.dev_qwk, self.test_qwk)
		return test_qwk

	def on_epoch_end(self, epoch, logs={}):
		self.losses.append(logs.get('loss'))
		self.val_losses.append(logs.get('val_loss'))
		self.accs.append(logs.get(self.metric))		
		self.val_accs.append(logs.get(self.metric))	
		self.test_qwks.append( self.eval(self.model, epoch, print_info=True) )
		if self.arg.plot:
			self.plothem()
		return

	def plothem(self):
		training_epochs = [i for i in range(len(self.losses))]
		plt.plot(training_epochs, self.losses, 'b', label='Train Loss')
		plt.plot(training_epochs, self.accs, 'r.', label='Train Accuracy')
		plt.plot(training_epochs, self.val_losses, 'g', label='Valid Loss')
		plt.plot(training_epochs, self.val_accs, 'y.', label='Valid Accuracy')
		plt.legend()
		plt.xlabel('epochs')
		plt.savefig(self.out_dir + '/' + self.timestr + 'LossAccuracy.png')
		plt.close()
		
		plt.plot(training_epochs, self.accs, 'b', label='Train Accuracy')
		plt.plot(training_epochs, self.val_accs, 'g', label='Valid Accuracy')
		plt.plot(training_epochs, self.test_qwks, 'r.', label='Test QWK')
		plt.xlabel('epochs')
		plt.legend()
		plt.savefig(self.out_dir + '/' + self.timestr + 'QWK.png')
		plt.close()	
			
	def print_info(self, test_qwk):
# 		logger.info('[Dev]   loss: %.4f, metric: %.4f, mean: %.3f (%.3f), stdev: %.3f (%.3f)' % (
# 			self.dev_loss, self.dev_metric, self.dev_pred.mean(), self.dev_mean, self.dev_pred.std(), self.dev_std))
		logger.info('\n')
		logger.info('[Test]  loss: %.4f, metric: %.4f, mean: %.3f (%.3f), stdev: %.3f (%.3f)' % (
			self.test_loss, self.test_metric, self.test_pred.mean(), self.test_mean, self.test_pred.std(), self.test_std))
# 		logger.info('[DEV]   QWK:  %.3f, LWK: %.3f, PRS: %.3f, SPR: %.3f, Tau: %.3f (Best @ %i: {{%.3f}}, %.3f, %.3f, %.3f, %.3f)' % (
# 			self.dev_qwk, self.dev_lwk, self.dev_prs, self.dev_spr, self.dev_tau, self.best_dev_epoch,
# 			self.best_dev[0], self.best_dev[1], self.best_dev[2], self.best_dev[3], self.best_dev[4]))
# 		logger.info('[TEST]  QWK:  %.3f, LWK: %.3f, PRS: %.3f, SPR: %.3f, Tau: %.3f (Best @ %i: {{%.3f}}, %.3f, %.3f, %.3f, %.3f)' % (
# 			self.test_qwk, self.test_lwk, self.test_prs, self.test_spr, self.test_tau, self.best_dev_epoch,
# 			self.best_test[0], self.best_test[1], self.best_test[2], self.best_test[3], self.best_test[4]))
		logger.info('[TEST]  QWK:  %.3f (Best @ %i: %.3f)' % (test_qwk, self.best_epoch, self.best_test))				
		logger.info('--------------------------------------------------------------------------------------------')
	
	def print_final_info(self):
		logger.info('\n')
		logger.info('--------------------------------------------------------------------------------------------')
# 		logger.info('Missed @ Epoch %i:' % self.best_test_missed_epoch)
# 		logger.info('  [TEST] QWK: %.3f' % self.best_test_missed)
		logger.info('Best @ Epoch %i:' % self.best_epoch)
# 		logger.info('  [DEV]  QWK: %.3f, LWK: %.3f, PRS: %.3f, SPR: %.3f, Tau: %.3f' % (self.best_dev[0], self.best_dev[1], self.best_dev[2], self.best_dev[3], self.best_dev[4]))
# 		logger.info('  [TEST] QWK: %.3f, LWK: %.3f, PRS: %.3f, SPR: %.3f, Tau: %.3f' % (self.best_test[0], self.best_test[1], self.best_test[2], self.best_test[3], self.best_test[4]))
		logger.info('  [TEST] QWK: %.3f' % (self.best_test))
# 		return self.best_dev[0]
		return self.best_test

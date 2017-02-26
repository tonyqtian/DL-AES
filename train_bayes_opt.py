'''
Created on Feb 26, 2017

@author: tonyq
'''

from bayes_opt import BayesianOptimization
from train_opt_helper import train_opt

if __name__ == '__main__':
	gp_params = {"alpha": 1e5}
	
# 	print(train_opt(3, 2, 0.1))
# 	raise RuntimeError

# 	svcBO = BayesianOptimization(train_opt, {'convkernel': (0, 16), 'convwin': (1, 5), 'dropout': (0.01, 0.99)})
# 	svcBO.explore({'convkernel': [0, 8, 16], 'convwin': [2, 3, 4],'dropout': [0.4, 0.5, 0.6]})

	svcBO = BayesianOptimization(train_opt, {
# 											 'convkernel': (0, 128),
# 											 'convwin': (2, 5),
											 'rnn_dim': (0, 128),
											 'bi_rmm': (0, 1),
											 'rnn_layers': (0, 4),
											 'embd_train': (0, 1),
 											 'embd_dim': (0, 4),
											 'tfidf': (0, 1),
# 											 'lr': (0.0001, 1),
											 'dropout': (0.01, 0.99)})
	svcBO.explore({
# 				   'convkernel': [0, 32, 0],
# 				   'convwin':    [2, 2,  0], 
				   'rnn_dim':    [0, 0,  32, 32, 32, 16,],
				   'bi_rmm':     [0, 0,  0,   1,  0,  1,],
				   'rnn_layers': [0, 0,  0,   0,  2,  2,],
				   'embd_train': [0, 0,  0,   1,  1,  1,],
 				   'embd_dim':   [0, 0,  1,   1,  2,  2,],
				   'tfidf':      [0, 0,   0,  0,   0, 1,],
# 				   'lr':         [0.001, 0.001],
				   'dropout':    [0.2, 0.4, 0.4, 0.5, 0.4, 0.4]})
	
# 	svcBO.maximize(n_iter=10, acq='ucb', kappa=10, **gp_params)
	svcBO.maximize(n_iter=1, acq="poi", xi=0.1, **gp_params)
	
	print('-'*53)
	print('Final Results')
	print('SVC: %f' % svcBO.res['max']['max_val'])
	print('Params: ', svcBO.res['max']['max_params'])
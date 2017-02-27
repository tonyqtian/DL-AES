'''
Created on Feb 26, 2017

@author: tonyq
'''
import argparse
from time import sleep

from bayes_opt import BayesianOptimization
from train_opt_helper import train_opt

parser = argparse.ArgumentParser()
parser.add_argument("--opt-iter", dest="iter", type=int, metavar='<int>', default=10, help="Iterations")
args1 = parser.parse_args()

gp_params = {"alpha": 1e5}

QWK = BayesianOptimization(train_opt, {
# 										 'bi_rmm': (0, 1),
# 										 'rnn_layers': (0, 3),
# 										 'embd_train': (0, 1),
# 										 'embd_dim': (0, 4),
# 										 'tfidf': (0, 1),
# 										 'lr': (0.0001, 1),
# 										 'convwin': (2, 5),
# 										 'convkernel': (0, 128),
										 'rnn_dim': (0, 192),
										 'dropout': (0.01, 0.99),
										 'dropout_w': (0.01, 0.99),
										 'dropout_u': (0.01, 0.99)})
QWK.explore({
# 			'bi_rmm':     [0, 0,  0,   1,  0,  1,],
# 			'rnn_layers': [0, 0,  0,   0,  2,  2,],
# 			'embd_train': [0, 0,  0,   1,  1,  1,],
# 			'embd_dim':   [0, 0,  1,   1,  2,  2,],
# 			'tfidf':      [0, 0,   0,  0,   0, 1,],
# 			'lr':         [0.001, 0.001],
# 			'convwin':    [2, 2,  0], 
# 			'convkernel': [0, 32, 0],
			'rnn_dim':    [0,     0,  32,  32, 128, 128, 192, 192,],
			'dropout':    [0.2, 0.6, 0.2, 0.6, 0.2, 0.6, 0.3, 0.6,],
			'dropout_w':  [0.2, 0.4, 0.4, 0.5, 0.4, 0.4, 0.4, 0.5,],
			'dropout_u':  [0.2, 0.4, 0.4, 0.5, 0.4, 0.4, 0.3, 0.6,] })

#QWK.maximize(n_iter=10, acq='ucb', kappa=10, **gp_params)
QWK.maximize(n_iter=args1.iter, acq="poi", xi=0.1, **gp_params)

print('-'*53)
print('Final Results')
print('QWK: %f' % QWK.res['max']['max_val'])
print('Params: ', QWK.res['max']['max_params'])

print('\a')
sleep(0.3)
print('\a')
sleep(0.3)
print('\a')
sleep(1)
print('\a')
sleep(0.3)
print('\a')
sleep(0.3)
print('\a')
sleep(1)
print('\a')
sleep(0.3)
print('\a')
sleep(0.3)
print('\a')
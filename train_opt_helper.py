'''
Created on Feb 26, 2017

@author: tonyq
'''

def train_opt(convkernel=0, convwin=2, rnn_dim=0, bi_rmm=0, rnn_layers=0, embd_train=0, embd_dim=0, tfidf=0, lr=0.003, dropout=0.4):
	
	import argparse
	###############################################################################################################################
	## Parse arguments
	#	
	parser = argparse.ArgumentParser()
	parser.add_argument("-tr", "--train", dest="train_path", type=str, metavar='<str>', default='data/train_s4.tsv', help="The path to the training set")
	parser.add_argument("-dv", "--dev", dest="dev_path", type=str, metavar='<str>', default='data/valid_s4.tsv', help="The path to the development set")
	parser.add_argument("-ts", "--test", dest="test_path", type=str, metavar='<str>', default='data/test_s4.tsv', help="The path to the test set")
	parser.add_argument("-o", "--out-dir", dest="out_dir_path", type=str, metavar='<str>', default='output', help="The path to the output directory")
	parser.add_argument("-p", "--prompt", dest="prompt_id", type=int, metavar='<int>', default=4, help="Promp ID for ASAP dataset. '0' means all prompts.")
	parser.add_argument("-m", "--model-type", dest="model_type", type=str, metavar='<str>', default='clsp', help="Model type (cls|clsp|reg|regp|breg|bregp) (default=regp)")
	parser.add_argument("-u", "--rec-unit", dest="recurrent_unit", type=str, metavar='<str>', default='lstm', help="Recurrent unit type (lstm|gru|simple) (default=lstm)")
	parser.add_argument("-a", "--algorithm", dest="algorithm", type=str, metavar='<str>', default='rmsprop', help="Optimization algorithm (rmsprop|sgd|adagrad|adadelta|adam|adamax) (default=rmsprop)")
	parser.add_argument("-l", "--loss", dest="loss", type=str, metavar='<str>', default='mse', help="Loss function (mse|mae|cnp) (default=mse) set to cnp if cls model")
	parser.add_argument("-e", "--embdim", dest="emb_dim", type=int, metavar='<int>', default=50, help="Embeddings dimension (default=50)")
	parser.add_argument("-c", "--cnn-kernel", dest="cnn_dim", type=int, metavar='<int>', default=0, help="CNN output dimension. '0' means no CNN layer (default=0)")
	parser.add_argument("-w", "--cnnwin", dest="cnn_window_size", type=int, metavar='<int>', default=3, help="CNN window size. (default=3)")
	parser.add_argument("-r", "--rnndim", dest="rnn_dim", type=int, metavar='<int>', default=0, help="RNN dimension. '0' means no RNN layer (default=0)")
	parser.add_argument("-b", "--batch-size", dest="batch_size", type=int, metavar='<int>', default=8, help="Batch size (default=32)")
	parser.add_argument("-v", "--vocab-size", dest="vocab_size", type=int, metavar='<int>', default=3733, help="Vocab size (default=4000)")
	parser.add_argument("--aggregation", dest="aggregation", type=str, metavar='<str>', default='mot', help="The aggregation method for regp and bregp types (mot|attsum|attmean) (default=mot)")
	parser.add_argument("--dropout", dest="dropout_prob", type=float, metavar='<float>', default=0.4, help="The dropout probability. To disable, give a negative number (default=0.4)")
	parser.add_argument("--vocab-path", dest="vocab_path", type=str, metavar='<str>', default='data/vocab_p4_glove40W.pkl', help="(Optional) The path to the existing vocab file (*.pkl)")
	parser.add_argument("--emb", dest="emb_path", type=str, metavar='<str>', default='data/glove.6B.50d.simple.txt', help="The path to the word embeddings file (Word2Vec format)")
	parser.add_argument("--skip-init-bias", dest="skip_init_bias", action='store_true', help="Skip initialization of the last layer bias")
	parser.add_argument("--epochs", dest="epochs", type=int, metavar='<int>', default=1, help="Number of epochs (default=50)")
	parser.add_argument("--maxlen", dest="maxlen", type=int, metavar='<int>', default=0, help="Maximum allowed number of words during training. '0' means no limit (default=0)")
	parser.add_argument("--seed", dest="seed", type=int, metavar='<int>', default=1111, help="Random seed (default=1234)")
	parser.add_argument("--tf-idf", dest="tfidf", type=int, metavar='<int>', default=0, help="Concatenate tf-idf matrix with model output (default dim=0)")
	parser.add_argument("--dense", dest="dense", type=int, metavar='<int>', default=0, help="Add dense layer before final full connected layer")
	parser.add_argument("--learning-rate", dest="learning_rate", type=float, metavar='<int>', default=0.0, help="Optimizer learning rate")
	parser.add_argument("--bi", dest="bi", action='store_true', help="Use bi-directional RNN")
	parser.add_argument("--plot", dest="plot", action='store_true', help="Save PNG plot")
	parser.add_argument("--embedding-trainable", dest="embd_train", action='store_true', help="Set embedding layer trainable")
	parser.add_argument("--2layer-rnn", dest="rnn_2l", action='store_true', help="Set 2 layer RNN")
	parser.add_argument("--3layer-rnn", dest="rnn_3l", action='store_true', help="Set 3 layer RNN")
	parser.add_argument("--onscreen", dest="onscreen", action='store_true', help="Show log on stdout")
	args, _ = parser.parse_known_args()
	
	args.cnn_dim = int(round(convkernel))
	args.cnn_window_size = int(round(convwin))
	args.rnn_dim = int(round(rnn_dim))
	if bi_rmm > 0.5:
		args.bi = True
		
	if rnn_layers > 3:
		args.rnn_3l = True
	elif rnn_layers > 2:
		args.rnn_2l = True
	elif rnn_layers < 1:
		args.rnn_dim = 0
	
	if embd_train > 0.5:
		args.embd_train = True

	if embd_dim > 3:
		args.emb_path = 'data/glove.6B.300d.simple.txt'
		args.emb_dim = 300
		if tfidf > 0.5:
			args.tfidf = 300
	elif embd_dim > 2:
		args.emb_path = 'data/glove.6B.200d.simple.txt'
		args.emb_dim = 200
		if tfidf > 0.5:
			args.tfidf = 200
	elif embd_dim > 1:
		args.emb_path = 'data/glove.6B.100d.simple.txt'
		args.emb_dim = 100
		if tfidf > 0.5:
			args.tfidf = 100
	else:
		args.emb_path = 'data/glove.6B.50d.simple.txt'
		args.emb_dim = 50
		if tfidf > 0.5:
			args.tfidf = 50
									
	args.learning_rate = lr
	args.dropout_prob = dropout
	
	args.plot = True
	args.epochs = 80
	
	from train_main import train
	return train(args)

if __name__ == '__main__':
	train_opt(0, 0, 0)
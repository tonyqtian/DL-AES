'''
Created on Feb 26, 2017

@author: tonyq
'''
import argparse
from time import sleep

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
parser.add_argument("-l", "--loss", dest="loss", type=str, metavar='<str>', default='mse', help="Loss function (mse|mae|cnp|hng) (default=mse) set to cnp if cls model")
parser.add_argument("-e", "--embdim", dest="emb_dim", type=int, metavar='<int>', default=50, help="Embeddings dimension (default=50)")
parser.add_argument("-c", "--cnn-kernel", dest="cnn_dim", type=int, metavar='<int>', default=0, help="CNN output dimension. '0' means no CNN layer (default=0)")
parser.add_argument("-w", "--cnnwin", dest="cnn_window_size", type=int, metavar='<int>', default=3, help="CNN window size. (default=3)")
parser.add_argument("-r", "--rnndim", dest="rnn_dim", type=int, metavar='<int>', default=0, help="RNN dimension. '0' means no RNN layer (default=0)")
parser.add_argument("-b", "--batch-size", dest="batch_size", type=int, metavar='<int>', default=8, help="Batch size (default=32)")
parser.add_argument("-v", "--vocab-size", dest="vocab_size", type=int, metavar='<int>', default=3733, help="Vocab size (default=4000)")
parser.add_argument("--aggregation", dest="aggregation", type=str, metavar='<str>', default='mot', help="The aggregation method for regp and bregp types (mot|att) (default=mot)")
parser.add_argument("--dropout", dest="dropout_prob", type=float, metavar='<float>', default=0.4, help="The dropout probability. To disable, give a negative number (default=0.4)")
parser.add_argument("--dropout-w", dest="dropout_w", type=float, metavar='<float>', default=0.0, help="The dropout probability of RNN W. To disable, give a negative number (default=0.4)")
parser.add_argument("--dropout-u", dest="dropout_u", type=float, metavar='<float>', default=0.0, help="The dropout probability of RNN U. To disable, give a negative number (default=0.4)")
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
parser.add_argument("--earlystop", dest="earlystop", type=float, metavar='int', default=4, help="Use early stop")
parser.add_argument("--normalize", dest="normalize", action='store_true', help="Normalize score to 0~1 when use reg model")
parser.add_argument("--verbose", dest="verbose", type=int, metavar='<int>', default=1, help="Show training process bar during train and val")
# parser.add_argument("--dense-activation", dest="dense_activation", type=str, metavar='<str>', default='tanh', help="Dense layer activation type (tanh|sigmoid|relu) (default=tanh)")
parser.add_argument("--valid-split", dest="valid_split", type=float, metavar='<float>', default=0.0, help="Split validation set from training set (default=0.0)")
parser.add_argument("--rnn-optimization", dest="rnn_opt", type=str, metavar='<str>', default='gpu', help="RNN consume_less (cpu|mem|gpu) (default=gpu)")
parser.add_argument("--features", dest="features", action='store_true', help="Get numerical features from essays")
parser.add_argument("--tr-ftr", dest="train_feature_path", type=str, metavar='<str>', default=None, help="The path to the training feature file")
parser.add_argument("--dev-ftr", dest="dev_feature_path", type=str, metavar='<str>', default=None, help="The path to the valid feature file")
parser.add_argument("--test-ftr", dest="test_feature_path", type=str, metavar='<str>', default=None, help="The path to the test feature file")
parser.add_argument("--pre-train-path", dest="pre_train_path", type=str, metavar='<str>', default=None, help="The path to pre-training file")
parser.add_argument("--pre-epochs", dest="pre_epochs", type=int, metavar='<int>', default=0, help="Pre training epoch")
args = parser.parse_args()

from train_main import train

train(args)

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

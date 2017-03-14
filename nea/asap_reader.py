# import random
import codecs
# import sys
from nltk.tokenize import word_tokenize
import logging
import re
import numpy as np
import pickle as pk
from nea.fileparse import features
from nea.token_gen import tokenize_cleaner, sentence_cleaner
from tqdm import tqdm

logger = logging.getLogger(__name__)
num_regex = re.compile('^[+-]?[0-9]+\.?[0-9]*$')
ref_scores_dtype = 'int32'

asap_ranges = {
	0: (0, 60),
	1: (2,12),
	2: (1,6),
	3: (0,3),
	4: (0,3),
	5: (0,4),
	6: (0,4),
	7: (0,30),
	8: (0,60)
}

def get_ref_dtype():
	return ref_scores_dtype

def tokenize(string):
	tokens = word_tokenize(string)
# 	for index, token in enumerate(tokens):
# 		if token == '@' and (index+1) < len(tokens):
# 			tokens[index+1] = '@' + re.sub('[0-9]+.*', '', tokens[index+1])
# 			tokens.pop(index)
	return tokens

def get_score_range(prompt_id):
	return asap_ranges[prompt_id]

def get_model_friendly_scores(scores_array, prompt_id_array):
	arg_type = type(prompt_id_array)
	assert arg_type in {int, np.ndarray}
	if arg_type is int:
		low, high = asap_ranges[prompt_id_array]
		scores_array = (scores_array - low) / (high - low)
	else:
		assert scores_array.shape[0] == prompt_id_array.shape[0]
		dim = scores_array.shape[0]
		low = np.zeros(dim)
		high = np.zeros(dim)
		for ii in range(dim):
			low[ii], high[ii] = asap_ranges[prompt_id_array[ii]]
		scores_array = (scores_array - low) / (high - low)
	assert np.all(scores_array >= 0) and np.all(scores_array <= 1)
	return scores_array

def convert_to_dataset_friendly_scores(scores_array, prompt_id_array):
	arg_type = type(prompt_id_array)
	assert arg_type in {int, np.ndarray}
	if arg_type is int:
		low, high = asap_ranges[prompt_id_array]
		scores_array = scores_array * (high - low) + low
		assert np.all(scores_array >= low) and np.all(scores_array <= high)
	else:
		assert scores_array.shape[0] == prompt_id_array.shape[0]
		dim = scores_array.shape[0]
		low = np.zeros(dim)
		high = np.zeros(dim)
		for ii in range(dim):
			low[ii], high[ii] = asap_ranges[prompt_id_array[ii]]
		scores_array = scores_array * (high - low) + low
	return scores_array

def convert_1hot_to_score(scores_array):
	return np.argmax(scores_array, axis=1)

def is_number(token):
	return bool(num_regex.match(token))

def load_vocab(vocab_path):
	logger.info('Loading vocabulary from: ' + vocab_path)
	with open(vocab_path, 'rb') as vocab_file:
		vocab = pk.load(vocab_file)
	return vocab

def create_vocab(file_path, prompt_id, maxlen, vocab_size, tokenize_text, to_lower):
	logger.info('Creating vocabulary from: ' + file_path)
	if maxlen > 0:
		logger.info('  Removing sequences with more than ' + str(maxlen) + ' words')
	total_words, unique_words = 0, 0
	word_freqs = {}
	with codecs.open(file_path, mode='r', encoding='UTF8') as input_file:
		next(input_file)
		for line in input_file:
			tokens = line.strip().split('\t')
			essay_id = int(tokens[0])
			essay_set = int(tokens[1])
			content = tokens[2].strip()
			score = float(tokens[6])
			if essay_set == prompt_id or prompt_id <= 0:
				content = sentence_cleaner(content, to_lower)
				if tokenize_text:
					content = tokenize(content)
				else:
					content = content.split()
				if maxlen > 0 and len(content) > maxlen:
					continue
				for word in content:
					try:
						word_freqs[word] += 1
					except KeyError:
						unique_words += 1
						word_freqs[word] = 1
					total_words += 1
	logger.info('  %i total words, %i unique words' % (total_words, unique_words))
	import operator
	sorted_word_freqs = sorted(list(word_freqs.items()), key=operator.itemgetter(1), reverse=True)
	if vocab_size <= 0:
		# Choose vocab size automatically by removing all singletons
		vocab_size = 0
		for word, freq in sorted_word_freqs:
			if freq > 1:
				vocab_size += 1
	vocab = {'<pad>':0, '<unk>':1, '<num>':2}
	vcb_len = len(vocab)
	index = vcb_len
	for word, _ in sorted_word_freqs[:vocab_size - vcb_len]:
		vocab[word] = index
		index += 1
	return vocab

# def read_essays(file_path, prompt_id):
# 	logger.info('Reading tsv from: ' + file_path)
# 	essays_list = []
# 	essays_ids = []
# 	with codecs.open(file_path, mode='r', encoding='UTF8') as input_file:
# 		next(input_file)
# 		for line in input_file:
# 			tokens = line.strip().split('\t')
# 			if int(tokens[1]) == prompt_id or prompt_id <= 0:
# 				essays_list.append(tokens[2].strip())
# 				essays_ids.append(int(tokens[0]))
# 	return essays_list, essays_ids

def get_tfidf(file_path, prompt_id=-1, tfidf_dim=3000, pca_dim=50, tfidf=None, pca=None, training_material=True):
	logger.info('<TF/IDF> Reading dataset from: ' + file_path)
	data_x = []
	with codecs.open(file_path, mode='r', encoding='UTF8') as input_file:
		next(input_file)
		for line in input_file:
			tokens = line.strip().split('\t')
			essay_id = int(tokens[0])
			essay_set = int(tokens[1])
			content = tokens[2].strip()
			if essay_set == prompt_id or prompt_id <= 0:
				data_x.append(content)
		if training_material:
			from sklearn.feature_extraction.text import TfidfVectorizer
			tfidf = TfidfVectorizer(preprocessor=tokenize_cleaner, analyzer='word', max_features=tfidf_dim, tokenizer=word_tokenize, stop_words = 'english')
			tfidf_matrix =  tfidf.fit_transform(data_x)
		else:
			tfidf_matrix =  tfidf.transform(data_x)
		logger.info('<TF/IDF> TF/IDF matrix: ' + str(tfidf_matrix.shape))
		if training_material:
			from sklearn.decomposition import TruncatedSVD
			pca = TruncatedSVD(n_components=pca_dim)
			pca.fit(tfidf_matrix)
			pca_matrix = pca.fit_transform(tfidf_matrix)
		else:
			pca_matrix = pca.transform(tfidf_matrix)
# 		feature_names = tfidf.get_feature_names()
		logger.info('<TF/IDF> PCA matrix: ' + str(pca_matrix.shape))
	return pca_matrix, tfidf, pca

def read_dataset(file_path, prompt_id, maxlen, vocab, tokenize_text, to_lower, score_index=6):
	logger.info('Reading dataset from: ' + file_path)
	if maxlen > 0:
		logger.info('  Removing sequences with more than ' + str(maxlen) + ' words')
	data_x, data_y, prompt_ids = [], [], []
	num_hit, unk_hit, total = 0., 0., 0.
	maxlen_x = -1
	with codecs.open(file_path, mode='r', encoding='UTF8') as input_file:
		next(input_file)
		for line in input_file:
			tokens = line.strip().split('\t')
			essay_id = int(tokens[0])
			essay_set = int(tokens[1])
			content = tokens[2].strip()
			score = float(tokens[score_index])
			if essay_set == prompt_id or prompt_id <= 0:
				content = sentence_cleaner(content, to_lower)
				if tokenize_text:
					content = tokenize(content)
				else:
					content = content.split()
				if maxlen > 0 and len(content) > maxlen:
					continue
				indices = []

				for word in content:
					if word in vocab:
						indices.append(vocab[word])
					elif is_number(word):
						indices.append(vocab['<num>'])
						num_hit += 1						
					else:
						indices.append(vocab['<unk>'])
						unk_hit += 1
					total += 1
					
				data_x.append(indices)
				data_y.append(score)
				prompt_ids.append(essay_set)
				if maxlen_x < len(indices):
					maxlen_x = len(indices)
	logger.info('  <num> hit rate: %.2f%%, <unk> hit rate: %.2f%%' % (100*num_hit/total, 100*unk_hit/total))
	return data_x, data_y, prompt_ids, maxlen_x

def get_data(paths, prompt_id, vocab_size, maxlen, tokenize_text=False, to_lower=True, sort_by_len=False, vocab_path=None, score_index=6):
	if len(paths) == 3:
		train_path, dev_path, test_path = paths[0], paths[1], paths[2]
	else:
		train_path, test_path = paths[0], paths[1]
	
	if not vocab_path:
		vocab = create_vocab(train_path, prompt_id, maxlen, vocab_size, tokenize_text, to_lower)
		if len(vocab) < vocab_size:
			logger.warning('The vocabualry includes only %i words (less than %i)' % (len(vocab), vocab_size))
		else:
			assert vocab_size == 0 or len(vocab) == vocab_size
	else:
		vocab = load_vocab(vocab_path)
		if len(vocab) != vocab_size:
			logger.warning('The vocabualry includes %i words which is different from given: %i' % (len(vocab), vocab_size))
	logger.info('  Vocab size: %i' % (len(vocab)))
	
	train_x, train_y, train_prompts, train_maxlen = read_dataset(train_path, prompt_id, maxlen, vocab, tokenize_text, to_lower)
	if len(paths) == 3:
		dev_x, dev_y, dev_prompts, dev_maxlen = read_dataset(dev_path, prompt_id, 0, vocab, tokenize_text, to_lower)
	test_x, test_y, test_prompts, test_maxlen = read_dataset(test_path, prompt_id, 0, vocab, tokenize_text, to_lower)
	
	if len(paths) == 3:
		overal_maxlen = max(train_maxlen, dev_maxlen, test_maxlen)
	else:
		overal_maxlen = max(train_maxlen, test_maxlen)
	
	if len(paths) == 3:
		return ((train_x,train_y,train_prompts), (dev_x,dev_y,dev_prompts), (test_x,test_y,test_prompts), vocab, overal_maxlen)
	else:
		return ((train_x,train_y,train_prompts), (test_x,test_y,test_prompts), vocab, overal_maxlen)
	
def get_features(file_path, prompt_id=-1, norm=False):
	logger.info('<Features> Reading dataset from: ' + file_path)
	with codecs.open(file_path, mode='r', encoding='UTF8') as input_file:
		next(input_file)
		data = []
		for line in tqdm(input_file):
			# get obj for an essay
			send_obj = features(line.split('\n')[0].split('\t'))
			# get numberical features
			send_obj.set_word_count(5)
			# get PoS count: noun,verb,adj,adv
			send_obj.set_pos_features()
			# punctuation count: quote, dot, comma
			send_obj.set_punctuation_features()
			# countable features to vector
			send_obj.set_vectors()
			# top 1/3 text rank phrases
# 			send_obj.get_phrases()
			data.append(send_obj.vector[:-1])
		feature_matrix = np.array(data)
		if norm:
			from sklearn.preprocessing import normalize
			feature_matrix = normalize(feature_matrix, axis=1, norm='l1')
	return feature_matrix
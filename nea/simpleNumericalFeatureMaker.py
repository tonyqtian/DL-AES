'''
Created on Mar 14, 2017

@author: tonyq
'''
import sys
import codecs
from tqdm import tqdm
from fileparse import features
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--source", dest="path", type=str, metavar='<str>', default='../data/train.tsv', help="The path to the source file")
parser.add_argument("--output", dest="output", type=str, metavar='<str>', default='../data/train_feature.tsv', help="The path to the source file")
args = parser.parse_args()

with codecs.open(args.path, 'r', encoding='UTF-8') as fhd:
	with codecs.open(args.output, 'w', encoding='UTF-8') as fwrt:
		for line in fhd:
			next(fhd)
			inputs = fhd.readlines()
			for line in tqdm(inputs, file=sys.stdout):
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
				fwrt.write(' '.join([str(intnum) for intnum in send_obj.vector[:-1]]) + '\n')
		print('Finished')
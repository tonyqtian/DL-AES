'''
Created on Feb 22, 2017

@author: tonyq
'''
import argparse
import pickle

def load_vocab(vocab_path):
    print('Loading vocabulary from: ' + vocab_path)
    with open(vocab_path, 'rb') as vocab_file:
        vocab = pickle.load(vocab_file)
    return vocab

parser = argparse.ArgumentParser()
parser.add_argument("--vocab", dest="vocab_path", type=str, metavar='<str>', default='../output_dir/vocab_glove40W.pkl', help="The path to the vocab pickle")
parser.add_argument("--embd", dest="embd_path", type=str, metavar='<str>', default='../data/glove.6B.50d.40w.txt', help="The path to the embedding file")
args = parser.parse_args()

output_path = args.embd_path.rstrip(".txt") + '.simple.txt'
print('Write pruned file to ', output_path)

myDict = load_vocab(args.vocab_path)
with open(args.embd_path, 'r', encoding='UTF-8') as fhd:
    with open(output_path, 'w', encoding='UTF-8') as fwrt:
        for line in fhd:
            if line.strip().split()[0] in myDict:
                fwrt.write(line)
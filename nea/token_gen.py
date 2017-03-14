from nltk import word_tokenize
import re

def tokenize(data, stops = []):
	if stops == []:
		stops = ['a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any', 'are', "aren't", 'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', "can't", 'cannot', 'could', "couldn't", 'did', "didn't", 'do', 'does', "doesn't", 'doing', "don't", 'down', 'during', 'each', 'few', 'for', 'from', 'further', 'had', "hadn't", 'has', "hasn't", 'have', "haven't", 'having', 'he', "he'd", "he'll", "he's", 'her', 'here', "here's", 'hers', 'herself', 'him', 'himself', 'his', 'how', "how's", 'i', "i'd", "i'll", "i'm", "i've", 'if', 'in', 'into', 'is', "isn't", 'it', "it's", 'its', 'itself', "let's", 'me', 'more', 'most', "mustn't", 'my', 'myself', 'no', 'nor', 'not', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'ought', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 'same', "shan't", 'she', "she'd", "she'll", "she's", 'should', "shouldn't", 'so', 'some', 'such', 'than', 'that', "that's", 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there', "there's", 'these', 'they', "they'd", "they'll", "they're", "they've", 'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up', 'very', 'was', "wasn't", 'we', "we'd", "we'll", "we're", "we've", 'were', "weren't", 'what', "what's", 'when', "when's", 'where', "where's", 'which', 'while', 'who', "who's", 'whom', 'why', "why's", 'with', "won't", 'would', "wouldn't", 'you', "you'd", "you'll", "you're", "you've", 'your', 'yours', 'yourself', 'yourselves']
	endings = ["'t", "'s", "'d", "'m", "'ll", "'ve", "'re"]
	startings = set([x.split("'")[0] for x in stops])
	tokenized = []
	incorrect = []

	for line in data:
		try:
			tokenized.extend(word_tokenize(line))
		except:
			incorrect.append(line)

	temp = []
	PUNCTUATION = [ch for ch in """(){}[]<>!?.:;,`'"@#$%^&*+-|=~/\\_"""]
	OTHER = ['``']
	NER = ['NUM', 'PERSON', 'ORGANISATION', 'LOCATION', 'MONTH', 'DATE', 'TIME', 'MONEY', 'PERCENT', 'CAPS', 'CITY', 'STATE', 'EMAIL', 'DR']
	PUNCTUATION.extend(OTHER)
	for line in incorrect:
		for word in line.split():
			try:
				temp.extend(word_tokenize(word))
			except:
				if word[-1] in PUNCTUATION:
					punc = word[-1]
					word = word[:-1]
					temp.append(punc)
				temp.append(word)

	tokenized.extend(temp)

	final_tokens = []
	i=0
	while i < len(tokenized):
		if i < len(tokenized) - 1 and (tokenized[i+1] == "n't" or (tokenized[i] in startings and tokenized[i+1] in endings)):
			new_token = tokenized[i]+tokenized[i+1]
			final_tokens.append(new_token)
			i += 2
		elif tokenized[i][0] == "'" and (len(tokenized[i]) >= 4 or (len(tokenized[i]) != 1 and tokenized[i].strip("'") not in endings)):
			apostrophe = "'"
			final_tokens.append(apostrophe)
			final_tokens.append(tokenized[i].strip("'"))
			i += 1
		elif tokenized[i][0] == '*':
			temp = tokenized[i].split('*')
			new_token = temp.pop()
			if new_token:
				final_tokens.append(new_token)
			final_tokens.append('*'*len(temp))
			i += 1
		else:
			final_tokens.append(tokenized[i])
			i += 1

	final_tokens = [x for x in final_tokens if x]
	puncs = [x for x in final_tokens if x in PUNCTUATION]
	final_tokens = [x for x in final_tokens if x not in PUNCTUATION]
	for ner in NER:
		final_tokens = [x for x in final_tokens if not x.startswith(ner)]

	temp = []
	for each in final_tokens:
		try:
# 			each.encode('utf8')
			temp.append(each)
		except:
			pass

	return (temp, puncs)

def tokenize_cleaner(text, lower=True):
	if lower:
		text = text.lower()
	
	text = text.replace('can’t', 'cannot')
	text = text.replace('it’s', 'it is')
	text = text.replace('n’t', ' not')
	text = text.replace('’ll', ' will')
	text = text.replace('’s', ' ’s')
	text = text.replace('’ve', ' have')
	text = text.replace('’d', ' would')
	text = text.replace('’m', ' am')
	
	text = text.replace('can`t', 'cannot')
	text = text.replace('it`s', 'it is')
	text = text.replace('n`t', ' not')
	text = text.replace('`ll', ' will')
	text = text.replace('`s', ' ’s')
	text = text.replace('`ve', ' have')
	text = text.replace('`d', ' would')
	text = text.replace('`m', ' am')
	
	text = text.replace("can't", "cannot")
	text = text.replace("it's", "it is")
	text = text.replace("n't", " not")
	text = text.replace("'ll", " will")
	text = text.replace("'s", " ’s")
	text = text.replace("'ve", " have")
	text = text.replace("'d", " would")
	text = text.replace("'m", " am")
	
	#for ASAP text
	text = text.replace("&lt;", " ")
	text = text.replace('â€™', ' ')
	text = text.replace('â€\x9d', ' ')
	text = text.replace('â€œ', ' ')
	text = text.replace('¶', ' ')
	if lower:
		text = text.replace("lüsted", "")
		text = text.replace("minfong", "")
	else:
		text = text.replace("Lüsted", "")
		text = text.replace("Minfong", "")
	text = text.replace(" didnt ", " did not ")
	text = text.replace(" shouldnt ", " should not ")
	text = text.replace(" doesnt ", " does not ")
	text = text.replace(" wasnt ", " was not ")
	text = text.replace(" isnt ", " is not ")
	text = text.replace(" cant ", " cannot ")
	text = text.replace("travelling", "traveling")
	if lower:
		text = re.sub(r'@([A-Za-z])+\d+', lambda pat: pat.group().strip('@1234567890').lower(), text)
	else:
		text = re.sub(r'@([A-Z])+\d+', lambda pat: pat.group().strip('@1234567890').lower(), text)

	text = text.replace('…', ' ')
	text = text.replace('”', ' ')
	text = text.replace('“', ' ')
	text = text.replace('‘', ' ')
	text = text.replace('’', ' ')
	
	text = text.replace(',', ' ')
	text = text.replace('.', ' ')
	text = text.replace('(', ' ')
	text = text.replace(')', ' ')
	text = text.replace('[', ' ')
	text = text.replace(']', ' ')
	text = text.replace(':', ' ')
	text = text.replace("'", " ")
	text = text.replace('?', ' ')
	text = text.replace('!', ' ')
	text = text.replace(';', ' ')
	text = text.replace('&', ' ')
	text = text.replace('/', ' ')
	text = text.replace('-', ' ')
	text = text.replace('+', ' ')
	text = text.replace('#', ' ')    
	text = text.replace('"', ' ')
	
	return text

def sentence_cleaner(text, lower=False):
	if lower:
		text = text.lower()
	
	text = text.replace('can’t', 'cannot')
	text = text.replace('it’s', 'it is')
	text = text.replace('n’t', ' not')
	text = text.replace('’ll', ' will')
	text = text.replace('’s', ' ’s')
	text = text.replace('’ve', ' have')
	text = text.replace('’d', ' would')
	text = text.replace('’m', ' am')
	
	text = text.replace('can`t', 'cannot')
	text = text.replace('it`s', 'it is')
	text = text.replace('n`t', ' not')
	text = text.replace('`ll', ' will')
	text = text.replace('`s', ' ’s')
	text = text.replace('`ve', ' have')
	text = text.replace('`d', ' would')
	text = text.replace('`m', ' am')
	
	text = text.replace("can't", "cannot")
	text = text.replace("it's", "it is")
	text = text.replace("n't", " not")
	text = text.replace("'ll", " will")
	text = text.replace("'s", " ’s")
	text = text.replace("'ve", " have")
	text = text.replace("'d", " would")
	text = text.replace("'m", " am")
	
	#for ASAP text
	text = text.replace("&lt;", " ")
	text = text.replace('â€™', ' ')
	text = text.replace('â€\x9d', ' ')
	text = text.replace('â€œ', ' ')
	text = text.replace('¶', ' ')
	if lower:
		text = text.replace("lüsted", "")
		text = text.replace("minfong", "")
	else:
		text = text.replace("Lüsted", "")
		text = text.replace("Minfong", "")
	text = text.replace(" didnt ", " did not ")
	text = text.replace(" shouldnt ", " should not ")
	text = text.replace(" doesnt ", " does not ")
	text = text.replace(" wasnt ", " was not ")
	text = text.replace(" isnt ", " is not ")
	text = text.replace(" cant ", " cannot ")
	text = text.replace("travelling", "traveling")
	if lower:
		text = re.sub(r'@([A-Za-z])+\d+', lambda pat: pat.group().strip('@1234567890').lower(), text)
	else:
		text = re.sub(r'@([A-Z])+\d+', lambda pat: pat.group().strip('@1234567890').lower(), text)

	text = text.replace('…', ' … ')
	text = text.replace('”', ' ” ')
	text = text.replace('“', ' “ ')
	text = text.replace('/', ' / ')
	text = text.replace('‘', ' ‘ ')
	text = text.replace('’', ' ’ ')
	
	text = text.replace(',', ' , ')
	text = text.replace('.', ' . ')
	text = text.replace('(', ' ( ')
	text = text.replace(')', ' ) ')
	text = text.replace('[', ' ( ')
	text = text.replace(']', ' ) ')
	text = text.replace(':', ' : ')
	text = text.replace("'", " ' ")
	text = text.replace('?', ' ? ')
	text = text.replace('!', ' ! ')
	text = text.replace(';', ' ; ')
	
	text = text.replace('-', ' ')
	text = text.replace('+', ' ')
	text = text.replace('#', ' ')    
	text = text.replace('"', ' ')
	
	return text

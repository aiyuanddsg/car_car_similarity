#!/usr/bin/env python
# encoding: utf-8

from gensim.models import Word2Vec
#import logging
from gensim.models.word2vec import LineSentence, Text8Corpus

inp = '../../data/word2vec/user_doc.txt'
outp1 = '../../data/word2vec/user_doc.text.model'
outp2 = '../../data/word2vec/user_doc.text.vector'

sentences = Text8Corpus(inp)
model = Word2Vec(sentences, sg=1, size=100, window=5, min_count=5, workers=4, iter = 100)
#print model.most_similar(['tag_id3336'])
model.save(outp1)
#print model['fuel_type15']
model.save_word2vec_format(outp2, binary=False)

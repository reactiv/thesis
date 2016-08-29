"""
Save stuff as pickles
Answers as a dict[id, vocab_list]
Datasets as list[dict[question: vocab_list, answer: id_list]]
Vocab as dict[int, string]
"""
import pickle
import numpy as np
from features import tokenise

def get_encoding(string, vocab):
    tokens = tokenise(string)
    return [vocab[t].index for t in tokens if t in vocab]

def encode_questions(w2v, q_bodies, answers, distractors, splits):
    vocabulary = {}
    for word in w2v.vocab:
        vocabulary[w2v.vocab[word].index] = word

    w2v.syn0.save('keras/word2vec_100_dim.embeddings')
    pickle.dump(vocabulary, open('keras/vocab','wb'))
    q_tokens = [get_encoding(q.lower()) for q in q_bodies]
    a_tokens = [get_encoding(a.lower()) for a in answers]
    d_tokens = [get_encoding(d.lower()) for d in distractors]
    a_dict = {i: a for i, a in enumerate(a_tokens + d_tokens)}

    q_list = [{'question': t, 'answers': [i]} for i, t, in enumerate(q_tokens)]
    pickle.dump(a_dict, open('keras/answers', 'wb'))
    for fname, group in splits:
        q_part = [q_list[i] for i in group]
        pickle.dump(q_part, open('keras/answers'.format(fname), 'wb'))
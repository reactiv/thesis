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
    return [vocab[t].index+1 for t in tokens if t in vocab]

def encode_questions(w2v, train_questions, test_questions):
    # print('hi')
    vocabulary = {}
    for word in w2v.vocab:
        vocabulary[w2v.vocab[word].index+1] = word
    emb = w2v.syn0 / np.linalg.norm(w2v.syn0, axis=1)[:,None]
    emb = np.vstack([np.zeros(emb.shape[1]), emb])
    np.save('keras/word2vec_100_dim.embeddings', emb)
    pickle.dump(vocabulary, open('keras/vocabulary','wb'))
    a_index = 0
    a_dict = {}
    q_list = []
    d_list = []
    v_list = []
    true_v_list = []
    disp_v_list = []
    valid_split = 0.1
    valid_ind = int(len(train_questions) * valid_split)

    for q in train_questions[:valid_ind]:
        q_tokens = get_encoding(q.body.lower(), w2v.vocab)
        a_tokens = get_encoding(q.get_correct().lower(), w2v.vocab)
        corr = a_index
        a_index += 1
        d_tokens = [get_encoding(d.lower(), w2v.vocab) for d in q.all_distractors()]
        wrong = range(a_index,a_index+3)
        # print(wrong)
        a_index += 3

        a_dict[corr] = a_tokens
        for ind, w in zip(wrong, d_tokens):
            a_dict[ind] = w

        true_v_list.append({'question': q_tokens, 'answers': [corr]})
        disp_v_list.append({'question': q_tokens, 'good': [corr], 'bad': wrong})

    for q in train_questions[valid_ind:]:
        q_tokens = get_encoding(q.body.lower(), w2v.vocab)
        a_tokens = get_encoding(q.get_correct().lower(), w2v.vocab)
        corr = a_index
        a_index += 1
        d_tokens = [get_encoding(d.lower(), w2v.vocab) for d in q.all_distractors()]
        wrong = range(a_index,a_index+3)
        # print(wrong)
        a_index += 3

        a_dict[corr] = a_tokens
        for ind, w in zip(wrong, d_tokens):
            a_dict[ind] = w

        q_list.append({'question': q_tokens, 'answers': [corr]})
        v_list.append({'question': q_tokens, 'good': [corr], 'bad': wrong})

    for q in test_questions:
        q_tokens = get_encoding(q.body.lower(), w2v.vocab)
        a_tokens = get_encoding(q.get_correct().lower(), w2v.vocab)
        corr = a_index
        a_index += 1
        d_tokens = [get_encoding(d.lower(), w2v.vocab) for d in q.all_distractors()]
        wrong = range(a_index,a_index+3)
        # print(wrong)
        a_index += 3

        a_dict[corr] = a_tokens
        for ind, w in zip(wrong, d_tokens):
            a_dict[ind] = w

        d_list.append({'good': [corr], 'bad': wrong, 'question': q_tokens})

    print('{} answers'.format(len(a_dict)))
    print('{} validation'.format(len(true_v_list)))
    print('{} train'.format(len(q_list)))
    print('{} test'.format(len(d_list)))
    print('{} valid'.format(len(v_list)))
    pickle.dump(a_dict, open('keras/answers', 'wb'))
    pickle.dump(q_list, open('keras/train', 'wb'))
    pickle.dump(d_list, open('keras/dev', 'wb'))
    pickle.dump(v_list, open('keras/valid', 'wb'))
    pickle.dump(disp_v_list, open('keras/valid_disp', 'wb'))
    pickle.dump(true_v_list, open('keras/validation', 'wb'))

def encode_questions_with_gen(w2v, questions, train_ind, generated):
    # print('hi')
    vocabulary = {}
    for word in w2v.vocab:
        vocabulary[w2v.vocab[word].index+1] = word
    emb = w2v.syn0 / np.linalg.norm(w2v.syn0, axis=1)[:,None]
    np.save('keras/word2vec_100_dim.embeddings', emb)
    pickle.dump(vocabulary, open('keras/vocabulary','wb'))
    a_index = 0
    a_dict = {}
    q_list = []
    d_list = []
    for i, q in enumerate(questions):
        q_tokens = get_encoding(q.body.lower(), w2v.vocab)
        a_tokens = get_encoding(q.get_correct().lower(), w2v.vocab)
        corr = a_index
        a_index += 1
        d_tokens = [get_encoding(d.lower(), w2v.vocab) for d in q.all_distractors()]
        wrong = range(a_index,a_index+3)
        # print(wrong)
        a_index += 3

        a_dict[corr] = a_tokens
        for ind, w in zip(wrong, d_tokens):
            a_dict[ind] = w

        if i in train_ind:
            q_list.append({'question': q_tokens, 'answers': [corr]})
        else:
            d_list.append({'good': [corr], 'bad': wrong, 'question': q_tokens})

    g_list = []
    for (q, a) in generated:
        q_tokens = get_encoding(q.lower(), w2v.vocab)
        a_tokens = get_encoding(a.lower(), w2v.vocab)
        a_dict[a_index] = a_tokens
        g_list.append({'question': q_tokens, 'answers': [a_index]})
        a_index += 1


    pickle.dump(a_dict, open('keras/generated_answers', 'wb'))
    pickle.dump(q_list, open('keras/train', 'wb'))
    pickle.dump(d_list, open('keras/dev', 'wb'))
    pickle.dump(g_list, open('keras/gen', 'wb'))

def encode_generated_questions(w2v, questions, answers):
    vocabulary = {}
    for word in w2v.vocab:
        vocabulary[w2v.vocab[word].index+1] = word
    emb = w2v.syn0 / np.linalg.norm(w2v.syn0, axis=1)[:,None]
    np.save('keras/word2vec_100_dim.embeddings', emb)
    pickle.dump(vocabulary, open('keras/vocabulary','wb'))

    q_tokens = [{'question': get_encoding(q.lower(), w2v.vocab), 'answers': [i]} for i, q in enumerate(questions)]
    a_tokens = [get_encoding(a.lower(), w2v.vocab) for a in answers]

    a_dict = {i: a for i, a in enumerate(a_tokens)}

    pickle.dump(a_dict, open('keras/generated_answers', 'wb'))
    pickle.dump(q_tokens, open('keras/gen', 'wb'))

